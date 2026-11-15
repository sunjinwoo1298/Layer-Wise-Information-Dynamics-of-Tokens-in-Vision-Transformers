"""
Unit tests for Phase 2 progressive multi-layer token pruning.
Uses a randomly initialised DeiT-Tiny (pretrained=False) on CPU so no download is needed;
all properties tested are weight-independent.
"""

import pytest
import torch

from src.compression.progressive import (
    ProgressivePruner, RetentionSchedule, ScheduleError, prune_patch_tokens,
)
from src.compression.token_pruning import TokenPruner
from src.importance.random_scorer import RandomScorer
from src.models.vit_wrapper import ViTWrapper


@pytest.fixture(scope="module")
def wrapper():
    torch.manual_seed(0)
    return ViTWrapper("deit_tiny_patch16_224", pretrained=False, device="cpu")


@pytest.fixture(scope="module")
def x():
    return torch.randn(3, 3, 224, 224, generator=torch.Generator().manual_seed(1))


def test_empty_schedule_matches_original_logits(wrapper, x):
    pr = ProgressivePruner(wrapper, {}, scorer="random", seed=0)
    with torch.no_grad():
        orig = wrapper(x)
    res = pr.forward(x)
    assert (orig - res.logits).abs().max().item() <= 1e-4
    assert (orig - pr.forward_unpruned(x)).abs().max().item() <= 1e-4
    assert all(t["patch_tokens_out"] == 196 and not t["pruned_here"] for t in res.token_trace)


def test_all_counts_equal_current_is_noop(wrapper, x):
    pr = ProgressivePruner(wrapper, {2: 196, 4: 196}, scorer="random", seed=0)
    with torch.no_grad():
        orig = wrapper(x)
    assert (orig - pr.forward(x).logits).abs().max().item() <= 1e-4


@pytest.mark.parametrize("scorer", ["random", "cls_attention", "prediction_kl_approx"])
def test_counts_cls_and_order_multi_point(wrapper, x, scorer):
    sched = {2: 98, 4: 49, 8: 10}
    pr = ProgressivePruner(wrapper, sched, scorer=scorer, seed=0)
    res = pr.forward(x)
    B = x.shape[0]
    counts = [t["patch_tokens_out"] for t in res.token_trace]
    # exact requested count after each pruning point; never increases
    for layer, k in sched.items():
        assert res.token_trace[layer - 1]["patch_tokens_out"] == k
        assert res.token_trace[layer - 1]["pruned_here"]
        assert res.token_trace[layer - 1]["total_tokens_out"] == k + 1  # CLS retained
    assert counts == [196, 98, 98, 49, 49, 49, 49, 10, 10, 10, 10, 10]
    assert all(a >= b for a, b in zip(counts, counts[1:]))
    # kept original positions: right size, strictly ascending (order preserved), nested, in range
    prev = None
    for layer, k in sched.items():
        idx = res.kept_indices[layer]
        assert idx.shape == (B, k)
        assert (idx[:, 1:] > idx[:, :-1]).all()
        assert idx.min() >= 0 and idx.max() < 196
        if prev is not None:
            for b in range(B):
                assert set(idx[b].tolist()) <= set(prev[b].tolist())
        prev = idx
    assert res.logits.shape == (B, wrapper.num_classes)


def test_prune_function_keeps_cls_and_order():
    B, N, d = 2, 20, 4
    tokens = torch.randn(B, 1 + N, d)
    scores = torch.rand(B, N, generator=torch.Generator().manual_seed(3))
    out, idx = prune_patch_tokens(tokens, scores, 7, num_prefix=1)
    assert out.shape == (B, 8, d)
    assert torch.equal(out[:, 0], tokens[:, 0])                      # CLS untouched, first
    assert (idx[:, 1:] > idx[:, :-1]).all()
    for b in range(B):                                               # per-image top-k, original features kept
        assert set(idx[b].tolist()) == set(scores[b].topk(7).indices.tolist())
        assert torch.equal(out[b, 1:], tokens[b, 1:][idx[b]])


def test_prune_function_invalid_args():
    tokens, scores = torch.randn(1, 11, 4), torch.rand(1, 10)
    with pytest.raises(ValueError):
        prune_patch_tokens(tokens, scores, 0)
    with pytest.raises(ValueError):
        prune_patch_tokens(tokens, scores, 11)
    with pytest.raises(ValueError):
        prune_patch_tokens(tokens, torch.rand(1, 9), 3)


@pytest.mark.parametrize("bad, msg", [
    ({0: 100}, "Invalid layer 0"),
    ({13: 100}, "Invalid layer 13"),
    ({12: 100}, "final block"),
    ({4: 0}, "must be >= 1"),
    ({4: 197}, "exceeds"),
    ({2: 50, 4: 80}, "never increase"),
    ({4: 98.5}, "integer"),
    ({"4": 98}, "integer"),
    ({4: True}, "integer"),
])
def test_invalid_schedules_raise_clear_errors(wrapper, bad, msg):
    with pytest.raises(ScheduleError, match=msg):
        ProgressivePruner(wrapper, bad)


def test_unknown_scorer_raises(wrapper):
    with pytest.raises(ValueError, match="Unknown scorer"):
        ProgressivePruner(wrapper, {4: 98}, scorer="magic")


def test_same_seed_reproducible_random_selection(wrapper, x):
    sched = {2: 98, 4: 49}
    a = ProgressivePruner(wrapper, sched, "random", seed=7).forward(x)
    b = ProgressivePruner(wrapper, sched, "random", seed=7).forward(x)
    c = ProgressivePruner(wrapper, sched, "random", seed=8).forward(x)
    for layer in sched:
        assert torch.equal(a.kept_indices[layer], b.kept_indices[layer])
    assert torch.equal(a.logits, b.logits)
    assert not torch.equal(a.kept_indices[2], c.kept_indices[2])


@pytest.mark.parametrize("scorer_name, phase1_scorer", [("random", "random"), ("cls_attention", "attn")])
def test_single_layer_matches_phase1(wrapper, x, scorer_name, phase1_scorer):
    """One-layer schedule == Phase 1 extract -> score -> TokenPruner -> forward_from_pruned_tokens."""
    from src.importance.attention_scorer import AttentionScorer
    layer, k = 4, 98
    sc = RandomScorer(seed=5) if phase1_scorer == "random" else AttentionScorer()
    with torch.no_grad():
        orig = wrapper(x)
        pref, vis = wrapper.extract_tokens_at_layer(x, layer)
        scores = sc.score_tokens(wrapper, x, layer, pref, vis, orig)
        kept, _ = TokenPruner(preserve_spatial_order=True).prune(vis, scores, k)
        phase1 = wrapper.forward_from_pruned_tokens(pref, kept, layer)
    prog = ProgressivePruner(wrapper, {layer: k}, scorer_name, seed=5).forward(x)
    assert (phase1 - prog.logits).abs().max().item() <= 1e-5


def test_sequence_is_genuinely_shorter(wrapper, x):
    """Blocks after a pruning point must receive a shorter sequence (no padding/masking)."""
    seen = []
    hooks = [blk.register_forward_pre_hook(lambda m, a: seen.append(a[0].shape[1]))
             for blk in wrapper.model.blocks]
    try:
        ProgressivePruner(wrapper, {2: 98, 4: 49}, "random", seed=0).forward(x)
    finally:
        for h in hooks:
            h.remove()
    assert seen == [197, 197, 99, 99] + [50] * 8
