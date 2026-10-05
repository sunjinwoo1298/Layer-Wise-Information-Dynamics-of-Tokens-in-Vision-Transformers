"""
Unit tests for TokenPruner, top-k selection, and spatial order preservation.
"""

import pytest
import torch
from src.compression.token_pruning import TokenPruner


def test_token_pruner_dimensions_and_order():
    pruner = TokenPruner(preserve_spatial_order=True)
    B, N, d = 2, 196, 64
    k = 50

    visual_tokens = torch.randn(B, N, d)
    # Assign synthetic importance scores
    scores = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(B, -1)

    retained_tokens, selected_indices = pruner.prune(visual_tokens, scores, k)

    assert retained_tokens.shape == (B, k, d)
    assert selected_indices.shape == (B, k)
    
    # Check that indices are strictly ascending (spatial order preserved)
    for b in range(B):
        idx_list = selected_indices[b].tolist()
        assert idx_list == sorted(idx_list)
        # Top 50 highest scored in arange(196) should be 146..195
        assert idx_list == list(range(196 - k, 196))


def test_token_pruner_full_retention():
    pruner = TokenPruner(preserve_spatial_order=True)
    B, N, d = 2, 100, 32
    visual_tokens = torch.randn(B, N, d)
    scores = torch.rand(B, N)

    retained_tokens, selected_indices = pruner.prune(visual_tokens, scores, k=N)
    assert torch.equal(retained_tokens, visual_tokens)
    assert selected_indices.shape == (B, N)
