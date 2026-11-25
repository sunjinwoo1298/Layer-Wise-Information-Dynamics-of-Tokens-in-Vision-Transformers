"""
Block-level forward for timm DeiT with a pluggable token compressor and realized-count FLOPs.

Per block: x = x + attn(norm1(x)); [compressor acts here: after attention, before MLP]; x = x + mlp(norm2(x)).
FLOPs convention (frozen protocol): attention (+ half of norm cost) is charged on tokens ENTERING the block, the MLP (+ half of norm cost) on
tokens LEAVING the compression step, patch-embed + head fixed. With no compression this equals the Phase 1-3 ComputeEstimator exactly.
Scoring / merging / fusion cost is NOT part of this figure; it is reported separately (analytic helpers + measured timers).
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

D_DEFAULT, MLP_DEFAULT, HEADS_DEFAULT = 192, 768, 3


class Compressor:
    """Base class: no compression. Subclasses override `compress`."""
    name = "none"
    uses_size = False
    needs_attn_cls = False
    needs_keys = False

    def begin(self, batch: int, tokens: int, device) -> None:
        self.size: Optional[torch.Tensor] = torch.ones(batch, tokens, 1, device=device) if self.uses_size else None
        self.events: List[Dict[str, Any]] = []
        self.aux_seconds = 0.0
        self.aux_flops = 0.0

    def compress(self, i: int, x: torch.Tensor, attn_cls: Optional[torch.Tensor], keys: Optional[torch.Tensor]) -> torch.Tensor:
        return x


def block_attention(blk, h: torch.Tensor, size: Optional[torch.Tensor], want_cls: bool, want_keys: bool):
    """timm Attention re-implemented explicitly (so attention probabilities / keys are available). Optional proportional attention."""
    a = blk.attn
    B, N, C = h.shape
    qkv = a.qkv(h).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = a.q_norm(q), a.k_norm(k)
    attn = (q * a.scale) @ k.transpose(-2, -1)
    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]
    attn = attn.softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    out = a.proj(out)
    cls_row = attn[:, :, 0, 1:].mean(dim=1) if want_cls else None   # (B, N-1): head-mean CLS attention to the other tokens
    keys = k.mean(dim=1) if want_keys else None                      # (B, N, hd): head-mean keys (ToMe metric)
    return out, cls_row, keys


@torch.no_grad()
def run_compressed(model, images: torch.Tensor, comp: Optional[Compressor] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
    comp = comp or Compressor()
    x = model.patch_embed(images)
    x = model._pos_embed(x)
    x = model.patch_drop(x) if hasattr(model, "patch_drop") else x
    x = model.norm_pre(x)
    B, T0 = x.shape[0], x.shape[1]
    comp.begin(B, T0, x.device)
    entering, leaving = [], []
    for i, blk in enumerate(model.blocks):
        h = blk.norm1(x)
        out, cls_row, keys = block_attention(blk, h, comp.size if comp.uses_size else None, comp.needs_attn_cls, comp.needs_keys)
        x = x +blk.drop_path1(blk.ls1(out))
        entering.append(x.shape[1])
        x = comp.compress(i, x, cls_row, keys)
        leaving.append(x.shape[1])
        x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
    x = model.norm(x)
    logits = model.forward_head(x)
    return logits, {"entering": entering, "leaving": leaving, "events": list(comp.events), "aux_seconds": comp.aux_seconds,
                    "aux_flops": comp.aux_flops}


# ---------------- analytic FLOPs on realized counts ----------------

def _attn_f(T, d=D_DEFAULT, H=HEADS_DEFAULT):
    return 6 * T * d * d + 2 * T * T * d + 3 * H * T * T + 2 * T * T * d + 2 * T * d * d + 2 * T * d


def _mlp_f(T, d=D_DEFAULT, mlp=MLP_DEFAULT):
    return 2 * T * d * mlp + 8 * T * mlp + 2 * T * mlp * d + 2 * T * d


def fixed_flops(d=D_DEFAULT, patches=196, patch=16, classes=1000):
    return 2 * 3 * patch * patch * d * patches + 2 * d * classes


def flops_from_counts(entering, leaving, d=D_DEFAULT, H=HEADS_DEFAULT, mlp=MLP_DEFAULT) -> float:
    return float(fixed_flops(d) + sum(_attn_f(a, d, H) + _mlp_f(m, d, mlp) for a, m in zip(entering, leaving)))


def flops_pct(entering, leaving, **kw) -> float:
    base = flops_from_counts([197] * 12, [197] * 12, **kw)
    return 100.0 * flops_from_counts(entering, leaving, **kw) / base
