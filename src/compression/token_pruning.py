"""
Token Ranking and Pruning Module for Spatial Patch Tokens.
"""

from typing import Tuple
import torch


def prune_tokens_by_indices(
    visual_tokens: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Gathers visual tokens using specified index tensor.
    visual_tokens: (B, N, d)
    indices: (B, k)
    returns: (B, k, d)
    """
    B, _, d = visual_tokens.shape
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, d)
    return torch.gather(visual_tokens, dim=1, index=expanded_indices)


class TokenPruner:
    """
    Ranks visual patch tokens based on importance scores S_l(t)
    and extracts the top-k subset while strictly preserving spatial ordering.
    """

    def __init__(self, preserve_spatial_order: bool = True):
        self.preserve_spatial_order = preserve_spatial_order

    def prune(
        self,
        visual_tokens: torch.Tensor,
        scores: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rank visual tokens and retain top-k.

        Args:
            visual_tokens: Tensor of visual tokens (B, N, d).
            scores: Importance scores S_l(t) of shape (B, N).
            k: Target number of retained visual tokens (1 <= k <= N).

        Returns:
            retained_tokens: Tensor of shape (B, k, d).
            selected_indices: Tensor of shape (B, k) indicating original token positions.
        """
        B, N, d = visual_tokens.shape
        assert 1 <= k <= N, f"Target token count k={k} must be in range [1, {N}]"

        if k == N:
            # Full retention (no pruning)
            indices = torch.arange(N, device=visual_tokens.device).unsqueeze(0).expand(B, -1)
            return visual_tokens, indices

        # Extract top-k indices based on importance scores
        topk_indices = scores.topk(k, dim=1, largest=True, sorted=True).indices  # (B, k)

        if self.preserve_spatial_order:
            # Sort indices along sequence dimension to preserve relative spatial positional ordering
            topk_indices = topk_indices.sort(dim=1).values

        retained_tokens = prune_tokens_by_indices(visual_tokens, topk_indices)
        return retained_tokens, topk_indices
