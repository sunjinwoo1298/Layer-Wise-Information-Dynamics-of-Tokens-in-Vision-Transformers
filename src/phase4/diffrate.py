"""
DiffRate (Chen et al., ICCV 2023, arXiv:2305.17997) inference with the RELEASED DeiT-Tiny compression rates, applied exactly as published.
Independent re-implementation (the official repository has no license file; no code copied). Behavior read from the official code:
  per block (block 0 uncompressed, block 11 compressed like the others):
    attention (with proportional attention: attn += log(size)) -> sort tokens by head-mean CLS attention (CLS pinned at position 0; size
    reordered too) -> PRUNE by slicing to prune_kept_num -> if merge_kept_num < prune_kept_num: MERGE to merge_kept_num: destinations are
    the first merge_kept_num tokens, sources the rest; cosine similarity of the tokens (L2-normalised residual stream), no source may merge
    into CLS, each source -> argmax destination, destination = mean over {itself + its sources} (scatter_reduce 'mean', default include_self),
    size update = sum of (source size * similarity score) and destination size -> MLP.
The published lists are NEVER repaired. A requested merge target larger than the number of available tokens is executed literally
(no sources -> nothing to merge) and recorded as an event; `strict=True` raises instead. (Literal-no-op reading is UNVERIFIED: the official
code was not executed, see implementation_deviations.md.)
"""

import json
import time
from typing import Sequence

import torch

from .engine import Compressor


class IllDefinedMerge(RuntimeError):
    pass


class DiffRateReleased(Compressor):
    name = "diffrate_released"
    uses_size = True
    needs_attn_cls = True

    def __init__(self, prune_kept_num: Sequence[int], merge_kept_num: Sequence[int], strict: bool = False):
        assert len(prune_kept_num) == len(merge_kept_num) == 12
        self.P, self.M, self.strict = list(prune_kept_num), list(merge_kept_num), strict

    @classmethod
    def from_published(cls, path: str, target: str, strict: bool = False):
        d = json.load(open(path))[target]
        return cls(d["prune_kept_num"], d["merge_kept_num"], strict)

    def compress(self, i, x, attn_cls, keys):
        t0 = time.perf_counter()
        B, N, C = x.shape
        order = torch.sort(attn_cls, dim=1, descending=True)[1]
        idx = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), order + 1], dim=1)    # CLS pinned
        x = x.gather(1, idx[..., None].expand(B, N, C))
        size = self.size.gather(1, idx[..., None])
        P, M = self.P[i], self.M[i]
        x, size = x[:, :P], size[:, :P]
        n_now = x.shape[1]
        ev = {"block": i, "prune_kept_num": P, "merge_kept_num": M, "tokens_in": N, "tokens_after_prune": n_now}
        if M < P:
            if M > n_now:
                ev["ill_defined"] = "merge target exceeds available tokens (%d > %d): no sources, executed as no-op" % (M, n_now)
                if self.strict:
                    raise IllDefinedMerge(ev["ill_defined"])
            else:
                metric = x / x.norm(dim=-1, keepdim=True)
                dst_m, src_m = metric[:, :M], metric[:, M:]
                sim = src_m @ dst_m.transpose(-1, -2)
                sim[..., :, 0] = -float("inf")                                    # nothing merges into CLS
                node_max, node_idx = sim.max(dim=-1)
                dst_idx = node_idx[..., None]
                n_src = n_now - M
                dst = x[:, :M].scatter_reduce(1, dst_idx.expand(B, n_src, C), x[:, M:], reduce="mean")
                sz_src = size[:, M:] * node_max[..., None]
                size = size[:, :M].scatter_reduce(1, dst_idx, sz_src, reduce="sum")
                x = dst
                self.aux_flops += 2.0 * n_src * M * C
        self.size = size
        ev["tokens_out"] = x.shape[1]
        self.events.append(ev)
        self.aux_seconds += time.perf_counter() - t0
        return x
