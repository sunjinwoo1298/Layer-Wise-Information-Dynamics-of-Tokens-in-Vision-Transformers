"""
Prediction-Based Task Sensitivity Token Importance Scorer (Baseline 3).
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TokenImportanceScorer


class PredictionKLDivergenceScorer(TokenImportanceScorer):
    """
    Measures task sensitivity S_l(t_i) = D_KL(p_orig || p_perturbed_i)
    by perturbing/ablating visual tokens at layer l and observing the shift in output prediction.
    
    Modes:
      - 'exact': Rigorous leave-one-out token zeroing / perturbation for every token.
      - 'approximate': Gradient-weighted feature attribution (Saliency: || t_i * grad_t_i ||).
    """

    def __init__(
        self,
        mode: Literal["exact", "approximate"] = "exact",
        perturbation_type: str = "zero",  # "zero" or "mean"
    ):
        super().__init__(name=f"prediction_kl_{mode}")
        self.mode = mode
        self.perturbation_type = perturbation_type

    def score_tokens(
        self,
        model_wrapper: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        prefix_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        orig_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute sensitivity score S_l(t_i) for each visual token.
        """
        B, N, d = visual_tokens.shape
        device = visual_tokens.device

        # Obtain original predictions if needed
        if orig_logits is None:
            with torch.no_grad():
                orig_logits = model_wrapper.forward_from_pruned_tokens(
                    prefix_tokens, visual_tokens, layer_idx
                )
        
        orig_probs = F.softmax(orig_logits, dim=-1)  # (B, num_classes)

        if self.mode == "exact":
            return self._score_exact(
                model_wrapper, layer_idx, prefix_tokens, visual_tokens, orig_probs
            )
        elif self.mode == "approximate":
            return self._score_approximate(
                model_wrapper, x, layer_idx, prefix_tokens, visual_tokens, orig_logits
            )
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _score_exact(
        self,
        model_wrapper: nn.Module,
        layer_idx: int,
        prefix_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        orig_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Exact leave-one-out perturbation: For each visual token i, zero/mean-mask it
        and compute KL divergence of output distribution.
        """
        B, N, d = visual_tokens.shape
        scores = torch.zeros(B, N, device=visual_tokens.device)

        with torch.no_grad():
            for i in range(N):
                perturbed_visual = visual_tokens.clone()
                if self.perturbation_type == "zero":
                    perturbed_visual[:, i, :] = 0.0
                elif self.perturbation_type == "mean":
                    mean_tok = visual_tokens.mean(dim=1, keepdim=True)
                    perturbed_visual[:, i, :] = mean_tok.squeeze(1)
                
                perturbed_logits = model_wrapper.forward_from_pruned_tokens(
                    prefix_tokens, perturbed_visual, layer_idx
                )
                perturbed_log_probs = F.log_softmax(perturbed_logits, dim=-1)

                kl_div = F.kl_div(
                    perturbed_log_probs,
                    orig_probs,
                    reduction="none",
                    log_target=False,
                ).sum(dim=-1)

                scores[:, i] = kl_div

        return scores

    def _score_approximate(
        self,
        model_wrapper: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        prefix_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        orig_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Gradient-activation saliency approximation:
        Attribution = || visual_tokens * d(TargetLogit) / d(visual_tokens) ||_2
        """
        with torch.enable_grad():
            vis_tok = visual_tokens.detach().clone().requires_grad_(True)
            pref_tok = prefix_tokens.detach().clone()

            logits = model_wrapper.forward_from_pruned_tokens(pref_tok, vis_tok, layer_idx)
            
            top_preds = orig_logits.argmax(dim=-1)
            target_logits = logits.gather(1, top_preds.unsqueeze(1)).sum()

            target_logits.backward()

            if vis_tok.grad is not None:
                attribution = (vis_tok * vis_tok.grad).norm(p=2, dim=-1)
            else:
                attribution = vis_tok.norm(p=2, dim=-1)

        return attribution.detach()
