"""
EViT-style, TRAINING-FREE token reorganization (reproduction-inspired; NOT official EViT).
Reference semantics (Liang et al., ICLR 2022, arXiv:2202.07800; github.com/youweiliang/evit, Apache-2.0), read from paper + official code:
  layers 4, 7, 10 (1-indexed), after the attention residual and before the MLP; importance = head-mean CLS-attention row over the other
  tokens; keep k = ceil(kappa * (N - 1)) where N counts CLS and any earlier fused token; top-k kept in descending attention order; the
  remaining tokens are fused into ONE token = sum(attention weight * token) (raw sum as in the official code; the paper text says
  "weighted average" - documented deviation), appended last; CLS stays first; no-op if k == N - 1.
Differences from official EViT: no training, no keep-rate warm-up, DeiT-Tiny weights that were never trained with reorganization.
"""

import math
import time
from typing import Sequence

import torch

from .engine import Compressor


class EViTStyle(Compressor):
    name = "evit_style"
    needs_attn_cls = True

    def __init__(self, kappa: float, locations_1indexed: Sequence[int] = (4, 7, 10), fuse: bool = True):
        self.kappa = float(kappa)
        self.locs = {int(l) - 1 for l in locations_1indexed}
        self.fuse = fuse

    def compress(self, i, x, attn_cls, keys):
        if i not in self.locs:
            return x
        t0 = time.perf_counter()
        B, N, C = x.shape
        left = math.ceil(self.kappa * (N - 1))
        if left >= N - 1 or left < 1:
            self.events.append({"block": i, "noop": True, "N": N})
            self.aux_seconds += time.perf_counter() - t0
            return x
        vals, idx = torch.topk(attn_cls, left, dim=1, largest=True, sorted=True)       # (B, left), indices into tokens 1..N-1
        sel = idx + 1
        kept = x.gather(1, sel[..., None].expand(B, left, C))
        parts = [x[:, :1], kept]
        if self.fuse:
            mask = torch.ones(B, N - 1, dtype=torch.bool, device=x.device)
            mask.scatter_(1, idx, False)
            rest = x[:, 1:][mask[..., None].expand(B, N - 1, C)].reshape(B, N - 1 - left, C)
            rest_w = attn_cls[mask].reshape(B, N - 1 - left)
            parts.append((rest * rest_w[..., None]).sum(dim=1, keepdim=True))
        out = torch.cat(parts, dim=1)
        self.events.append({"block": i, "noop": False, "N_in": N, "kept": left, "N_out": out.shape[1]})
        self.aux_flops += 2.0 * (N - 1 - left) * C + float(N) * math.log2(max(N, 2))      # fusion weighted sum + top-k (analytic estimate)
        self.aux_seconds += time.perf_counter() - t0
        return out
