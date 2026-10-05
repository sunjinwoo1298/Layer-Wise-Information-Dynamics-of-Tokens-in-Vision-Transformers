"""
Abstract Base Class for Token Importance Scorers S_l(t).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class TokenImportanceScorer(ABC):
    """
    Abstract interface for assigning task sensitivity/importance scores S_l(t) 
    to visual patch tokens at transformer layer l.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
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
        Compute importance scores S_l(t) for visual tokens.

        Args:
            model_wrapper: Instrumented ViTWrapper instance.
            x: Input images tensor (B, C, H, W).
            layer_idx: Current 1-indexed layer.
            prefix_tokens: Prefix / [CLS] token representations at layer_idx (B, num_prefix, d).
            visual_tokens: Visual patch token representations at layer_idx (B, N, d).
            orig_logits: Uncompressed model output logits (B, num_classes), cached if available.

        Returns:
            scores: Tensor of shape (B, N) where higher values indicate higher importance.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        return {"scorer_name": self.name}

