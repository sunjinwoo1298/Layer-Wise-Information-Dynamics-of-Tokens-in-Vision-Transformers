"""
ToMe (Bolya et al., ICLR 2023, arXiv:2210.09461) bipartite soft matching: independent re-implementation for timm 1.0.x DeiT.
Source repo / license: https://github.com/facebookresearch/ToMe (CC BY-NC 4.0). No source code copied; behavior follows the paper and the
documented semantics (parity against the official code is checked separately in an isolated environment).
Semantics: between attention and MLP; metric = head-mean keys, L2-normalised; alternating A/B split (A = even positions incl. CLS);
CLS never a merge source; at most 50% of the (non-protected) tokens per layer; size-weighted average; proportional attention.
"""

import time
from typing import List, Sequence, Tuple, Union

import torch

from .engine import Compressor

LAYERS = 12


def parse_r(num_layers: int, r: Union[int, Tuple[float, float], Sequence[int]]) -> List[int]:
    """Native r controls: int (constant), (r, inflection) pair (linear from int(r*(1-infl)) to 2r-that, int-truncated), or explicit list."""
    if isinstance(r, (list, tuple)) and len(r) == 2 and isinstance(r[1], float) and not isinstance(r[0], list):
        rr, infl = int(r[0]), float(r[1])
        if infl == 0:
            return [rr] * num_layers
        mn = int(rr * (1 - infl)); mx = 2 * rr - mn
        return [int(mn + (mx - mn) * i / (num_layers - 1)) for i in range(num_layers)]
    if isinstance(r, int):
        return [r] * num_layers
    lst = list(r)
    return lst + [0] * (num_layers - len(lst))


def bipartite_soft_matching(metric: torch.Tensor, r: int, protected: int = 1):
    """Returns (merge_fn, r_eff). metric: (B, T, c). Token 0 (CLS) is protected."""
    T = metric.shape[1]
    r = min(r, (T - protected) // 2)
    if r <= 0:
        return None, 0
    metric = metric / metric.norm(dim=-1, keepdim=True)
    a, b = metric[:, ::2, :], metric[:, 1::2, :]
    scores = a @ b.transpose(-1, -2)
    scores[:, 0, :] = -float("inf")                      # CLS (position 0, set A) is never a merge source
    node_max, node_idx = scores.max(dim=-1)
    edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
    unm_idx = edge_idx[:, r:, :]                          # kept A tokens
    src_idx = edge_idx[:, :r, :]                          # merged A tokens
    dst_idx = node_idx[..., None].gather(dim=1, index=src_idx)
    unm_idx = unm_idx.sort(dim=1)[0]                      # keeps CLS (index 0) first and original order of kept A tokens

    def merge(x: torch.Tensor, mode: str) -> torch.Tensor:
        src, dst = x[:, ::2, :], x[:, 1::2, :]
        n, t1, c = src.shape
        unm = src.gather(dim=1, index=unm_idx.expand(n, t1 - r, c))
        s = src.gather(dim=1, index=src_idx.expand(n, r, c))
        dst = dst.scatter_reduce(1, dst_idx.expand(n, r, c), s, reduce=mode)
        return torch.cat([unm, dst], dim=1)

    return merge, r


class ToMe(Compressor):
    name = "tome"
    uses_size = True
    needs_keys = True

    def __init__(self, r, num_layers: int = LAYERS):
        self.rs = parse_r(num_layers, r)
        self.r_spec = r

    def compress(self, i, x, attn_cls, keys):
        t0 = time.perf_counter()
        merge, r_eff = bipartite_soft_matching(keys, self.rs[i])
        if merge is not None:
            x = merge(x * self.size, "sum")
            self.size = merge(self.size, "sum")
            x = x / self.size
            T = keys.shape[1]
            self.aux_flops += 2.0 * ((T + 1) // 2) * (T // 2) * keys.shape[-1]      # similarity matmul (MACs x2), analytic estimate
        self.events.append({"block": i, "r_requested": self.rs[i], "r_effective": r_eff})
        self.aux_seconds += time.perf_counter() - t0
        return x
