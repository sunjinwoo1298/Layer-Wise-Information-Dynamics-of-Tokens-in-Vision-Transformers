"""
Phase 2: Progressive multi-layer patch-token pruning (infrastructure only; no budget optimizer).

Layer-numbering convention (the ONLY place it is converted is ``ProgressivePruner.forward``):
    Schedule keys are 1-based human-readable layer numbers, identical to Phase 1's ``layer_idx``.
    Key ``l`` means "after ``l`` Transformer blocks have run (i.e. after 0-based block index ``l-1``),
    keep exactly ``k`` patch tokens"; blocks with 0-based index ``l .. L-1`` then see the shorter sequence.
    Token counts always refer to PATCH tokens only (CLS / prefix tokens are always kept, never counted).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from .token_pruning import TokenPruner
from ..importance.base import TokenImportanceScorer
from ..importance.random_scorer import RandomScorer
from ..importance.prediction_kl import PredictionKLDivergenceScorer
from ..importance.attention_scorer import AttentionScorer

SCORER_NAMES = ("random", "prediction_kl_approx", "cls_attention")
# Phase 1 config labels accepted as aliases (they map to the identical Phase 1 implementations).
_SCORER_ALIASES = {"prediction_kl_approximate": "prediction_kl_approx", "attention_cls": "cls_attention"}


def build_scorer(name: str, seed: Optional[int] = None) -> Tuple[str, TokenImportanceScorer]:
    """Instantiate an unchanged Phase 1 scorer. Returns (canonical_name, scorer)."""
    canonical = _SCORER_ALIASES.get(name, name)
    if canonical == "random":
        return canonical, RandomScorer(seed=seed)
    if canonical == "prediction_kl_approx":
        # Phase 1 implementation: gradient x activation saliency (NOT an exact KL).
        return canonical, PredictionKLDivergenceScorer(mode="approximate")
    if canonical == "cls_attention":
        return canonical, AttentionScorer()
    raise ValueError(f"Unknown scorer '{name}'. Supported: {list(SCORER_NAMES)}")


class ScheduleError(ValueError):
    """Raised for an invalid retention schedule."""


@dataclass(frozen=True)
class RetentionSchedule:
    """Validated {layer (1-based) -> patch tokens retained after that layer}."""
    points: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)  # sorted (layer, k)

    @classmethod
    def from_dict(cls, schedule: Optional[Mapping[Any, Any]]) -> "RetentionSchedule":
        pts = []
        for layer, k in (schedule or {}).items():
            for label, v in (("layer key", layer), ("token count", k)):
                if isinstance(v, bool) or not isinstance(v, int):
                    raise ScheduleError(f"Schedule {label} must be an integer, got {v!r} ({type(v).__name__}).")
            pts.append((layer, k))
        return cls(points=tuple(sorted(pts)))

    def validate(self, num_layers: int, num_patches: int) -> None:
        """Check layers are valid, counts are in [1, current], and counts never increase."""
        current = num_patches
        for layer, k in self.points:
            if layer < 1 or layer > num_layers:
                raise ScheduleError(f"Invalid layer {layer}: must be in [1, {num_layers}] (1-based).")
            if layer == num_layers:
                raise ScheduleError(
                    f"Invalid layer {layer}: pruning after the final block saves no compute "
                    f"(no downstream blocks). Use layers in [1, {num_layers - 1}]."
                )
            if k < 1:
                raise ScheduleError(f"Layer {layer}: token count {k} must be >= 1.")
            if k > current:
                raise ScheduleError(
                    f"Layer {layer}: token count {k} exceeds the current patch-token count {current} "
                    f"(schedule must never increase the token count)."
                )
            current = k

    def as_dict(self) -> Dict[int, int]:
        return dict(self.points)


def prune_patch_tokens(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    k: int,
    num_prefix: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Keep the prefix (CLS) tokens plus the top-k patch tokens, restored to original sequence order.

    tokens: (B, num_prefix + N, d); scores: (B, N) patch-token scores; k: 1 <= k <= N.
    Returns (new_tokens (B, num_prefix + k, d), kept_idx (B, k) indices into the N input patch tokens,
    ascending). Selection is per image. Positional embeddings are already baked into the token
    representations and are NOT re-added.
    """
    prefix, patches = tokens[:, :num_prefix], tokens[:, num_prefix:]
    B, N, _ = patches.shape
    if scores.shape != (B, N):
        raise ValueError(f"scores must have shape {(B, N)}, got {tuple(scores.shape)}")
    if not 1 <= k <= N:
        raise ValueError(f"k={k} must be in [1, {N}]")
    kept, idx = TokenPruner(preserve_spatial_order=True).prune(patches, scores, k)
    return torch.cat([prefix, kept], dim=1), idx


@dataclass
class ProgressiveResult:
    logits: torch.Tensor                                   # (B, num_classes)
    token_trace: List[Dict[str, Any]]                      # one entry per block (after the block / its pruning point)
    kept_indices: Dict[int, torch.Tensor]                  # layer -> (B, k) ORIGINAL patch positions retained
    scorer_time_ms: float = 0.0


class ProgressivePruner:
    """Runs a ViTWrapper forward pass with patch-token pruning at several layers."""

    def __init__(self, wrapper, schedule: Mapping[Any, Any], scorer: str = "random", seed: Optional[int] = 42):
        self.wrapper = wrapper
        self.schedule = schedule if isinstance(schedule, RetentionSchedule) else RetentionSchedule.from_dict(schedule)
        self.schedule.validate(wrapper.num_layers, wrapper.num_original_tokens)
        self.scorer_name, self.scorer = build_scorer(scorer, seed=seed)

    @torch.no_grad()
    def forward_unpruned(self, x: torch.Tensor) -> torch.Tensor:
        """No-pruning path through the same embed/blocks/head pipeline."""
        w = self.wrapper
        tokens = w.forward_layers(w.embed_input(x), 0, w.num_layers)
        return w.forward_head(tokens)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, orig_logits: Optional[torch.Tensor] = None) -> ProgressiveResult:
        import time
        w = self.wrapper
        sched = self.schedule.as_dict()
        npfx = w.num_prefix_tokens
        x = x.to(w.device)

        # Phase 1 semantics: the saliency scorer targets the ORIGINAL (unpruned) model's top class.
        if sched and self.scorer_name == "prediction_kl_approx" and orig_logits is None:
            orig_logits = w(x)

        tokens = w.embed_input(x)
        B = tokens.shape[0]
        orig_pos = torch.arange(w.num_original_tokens, device=tokens.device).unsqueeze(0).expand(B, -1)
        trace: List[Dict[str, Any]] = []
        kept: Dict[int, torch.Tensor] = {}
        scorer_ms = 0.0

        for block_idx in range(w.num_layers):
            tokens = w.forward_layers(tokens, block_idx, block_idx + 1)
            layer = block_idx + 1  # the single 0-based -> 1-based conversion
            n_in = tokens.shape[1] - npfx
            pruned_here = False

            if layer in sched and sched[layer] < n_in:
                k = sched[layer]
                prefix, vis = tokens[:, :npfx], tokens[:, npfx:]
                t0 = time.perf_counter()
                try:
                    scores = self.scorer.score_tokens(w, x, layer, prefix, vis, orig_logits)
                except Exception as e:  # fail clearly; never substitute another scorer
                    raise RuntimeError(
                        f"Scorer '{self.scorer_name}' failed at layer {layer} with {n_in} patch tokens: {e}"
                    ) from e
                if w.device.type == "cuda":
                    torch.cuda.synchronize()
                scorer_ms += (time.perf_counter() - t0) * 1e3
                tokens, idx = prune_patch_tokens(tokens, scores.to(tokens.device), k, npfx)
                orig_pos = torch.gather(orig_pos, 1, idx)
                kept[layer] = orig_pos.clone()
                pruned_here = True

            trace.append({
                "after_block_layer": layer,
                "patch_tokens_in": n_in,
                "patch_tokens_out": tokens.shape[1] - npfx,
                "total_tokens_out": tokens.shape[1],
                "pruned_here": pruned_here,
            })

        logits = w.forward_head(tokens)
        return ProgressiveResult(logits=logits, token_trace=trace, kept_indices=kept, scorer_time_ms=scorer_ms)

    @staticmethod
    def patch_tokens_per_block(trace: List[Dict[str, Any]], num_patches: int) -> List[int]:
        """Patch tokens each block PROCESSED (input count), derived from the trace."""
        return [num_patches] + [t["patch_tokens_out"] for t in trace[:-1]]
