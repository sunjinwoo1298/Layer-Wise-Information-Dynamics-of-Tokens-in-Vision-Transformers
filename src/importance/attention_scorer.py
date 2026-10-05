"""
Attention-Based Token Importance Scorer (Baseline 4).
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import TokenImportanceScorer


class AttentionScorer(TokenImportanceScorer):
    """
    Baseline 4: Uses multi-head self-attention weights from the [CLS] prefix token 
    to visual patch tokens at layer l as token importance scores.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__(name="attention_cls")
        self.reduction = reduction

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
        Computes attention score from [CLS] to each visual token in block (layer_idx - 1).
        """
        # Retrieve target transformer block (0-indexed: layer_idx - 1)
        block = model_wrapper.model.blocks[layer_idx - 1]
        attn = block.attn
        
        # Combine prefix and visual tokens to reconstruct layer input
        tokens = torch.cat([prefix_tokens, visual_tokens], dim=1)
        # Apply layer norm 1 if present in block
        norm_tokens = block.norm1(tokens) if hasattr(block, "norm1") else tokens
        
        B, Total_N, d = norm_tokens.shape
        num_heads = attn.num_heads
        head_dim = d // num_heads

        # Calculate Q, K projections
        if hasattr(attn, "qkv"):
            qkv = attn.qkv(norm_tokens).reshape(B, Total_N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
            q, k, _ = qkv[0], qkv[1], qkv[2]  # (B, num_heads, Total_N, head_dim)
        elif hasattr(attn, "q_proj") and hasattr(attn, "k_proj"):
            q = attn.q_proj(norm_tokens).reshape(B, Total_N, num_heads, head_dim).permute(0, 2, 1, 3)
            k = attn.k_proj(norm_tokens).reshape(B, Total_N, num_heads, head_dim).permute(0, 2, 1, 3)
        else:
            # Fallback: token norm
            return visual_tokens.norm(dim=-1)

        # Scale dot product
        scale = getattr(attn, "scale", head_dim ** -0.5)
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, num_heads, Total_N, Total_N)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Attention from CLS (index 0) to visual tokens (indices num_prefix:)
        num_prefix = model_wrapper.num_prefix_tokens
        cls_to_patches = attn_weights[:, :, 0, num_prefix:]  # (B, num_heads, N)

        if self.reduction == "mean":
            scores = cls_to_patches.mean(dim=1)  # (B, N)
        elif self.reduction == "max":
            scores = cls_to_patches.max(dim=1).values
        else:
            scores = cls_to_patches.mean(dim=1)

        return scores.detach()
