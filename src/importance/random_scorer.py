"""
Random Token Importance Scorer (Baseline 2).
"""

from typing import Optional
import torch
import torch.nn as nn
from .base import TokenImportanceScorer


class RandomScorer(TokenImportanceScorer):
    """
    Baseline 2: Assigns uniformly random importance scores to tokens.
    Crucial for scientific calibration to determine if structured scoring out-performs random selection.
    """

    def __init__(self, seed: Optional[int] = None):
        super().__init__(name="random")
        self.seed = seed
        self.generator = torch.Generator() if seed is not None else None
        if seed is not None:
            self.generator.manual_seed(seed)

    def score_tokens(
        self,
        model_wrapper: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        prefix_tokens: torch.Tensor,
        visual_tokens: torch.Tensor,
        orig_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = visual_tokens.shape
        device = visual_tokens.device
        if self.generator is not None:
            # Generate reproducible random scores on CPU then move to device
            scores = torch.rand(B, N, generator=self.generator).to(device)
        else:
            scores = torch.rand(B, N, device=device)
        return scores
