"""
Unit tests for ViTWrapper, layer extraction, and baseline forward pass consistency.
"""

import pytest
import torch
from src.models.vit_wrapper import ViTWrapper


def test_vit_wrapper_initialization():
    model = ViTWrapper(model_name="deit_tiny_patch16_224", pretrained=False, device="cpu")
    assert model.num_layers == 12
    assert model.hidden_dim == 192
    assert model.num_heads == 3
    assert model.num_original_tokens == 196
    assert model.num_prefix_tokens >= 1


def test_baseline_consistency_at_k_equals_n():
    """Sanity Check: Intercepted forward pass with k=N must equal standard forward pass."""
    model = ViTWrapper(model_name="deit_tiny_patch16_224", pretrained=False, device="cpu")
    dummy_x = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        direct_logits = model(dummy_x)

        for l in [1, 4, 8, 12]:
            prefix, vis = model.extract_tokens_at_layer(dummy_x, l)
            assert prefix.shape == (2, model.num_prefix_tokens, 192)
            assert vis.shape == (2, 196, 192)

            intercepted_logits = model.forward_from_pruned_tokens(prefix, vis, l)
            max_diff = (direct_logits - intercepted_logits).abs().max().item()
            assert max_diff < 1e-4, f"Baseline mismatch at layer {l}: max diff = {max_diff}"
