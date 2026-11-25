"""
Phase 3 token-allocation search: model-free numerics.

IMPORTANT (leakage control): nothing in this module receives validation/confirmatory data. The only data-dependent input is the
`SensitivityTables` object (per-image KL values measured on the CALIBRATION set by the experiment module) and a `verify_fn` closure
that returns calibration agreement for a schedule; the selection logic below cannot see any other split.

Method ("LS-Chain + verify"), see results/phase3_algorithm_design/algorithm_design.md:
  proxy   Lhat(K) = D(l1,K1) + sum_{j>=2} [ D2(l_{j-1},K_{j-1}; l_j,K_j) - D(l_{j-1},K_{j-1}) ]      (telescopes to
          D2(l1,K1;l2,K2) + D2(l2,K2;l3,K3) - D(l2,K2) for three stages)
  D       mean_i KL(p_full || p_pruned) when pruning only at layer l to k tokens (calibration images, actual scorer)
  D2      same with two pruning stages (earlier stages unpruned)
  feasible  N > K1 > ... > Km >= floor and  cap - tau <= normalized analytic FLOPs <= cap
  rank    ascending Lhat, M schedules with pairwise L_inf distance >= min_dist
  select  highest VERIFIED calibration agreement (ties < tie_eps -> lower Lhat, then larger final count)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..metrics.compute import ComputeEstimator

DEFAULT_GRID = (16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 164, 184)


# ----------------------------------------------------------------------------- FLOPs (existing accounting, vectorised)

def block_flops_table(est: ComputeEstimator) -> np.ndarray:
    """f[n] = analytic FLOPs of ONE block that processes n patch tokens (+ prefix), n = 0..N (existing estimator)."""
    return np.array([est._block_flops(est.num_prefix + n) for n in range(est.N + 1)], dtype=np.float64)


def fixed_flops(est: ComputeEstimator) -> float:
    return float(2 * 3 * est.patch_size[0] * est.patch_size[1] * est.d * est.N + 2 * est.d * est.num_classes)


def stage_block_counts(layers: Sequence[int], num_layers: int) -> Tuple[int, List[int]]:
    """(# blocks that see all N tokens, [# blocks that see K_j for each stage j]) under 'key l = after l blocks' semantics."""
    ls = list(layers)
    segs = [ls[j + 1] - ls[j] for j in range(len(ls) - 1)] + [num_layers - ls[-1]]
    return ls[0], segs


def normalized_flops(est: ComputeEstimator, layers: Sequence[int], counts: Sequence[Any], f: Optional[np.ndarray] = None):
    """Normalised FLOPs for patch counts `counts` (scalars or broadcastable arrays), identical in value to schedule_flops."""
    f = block_flops_table(est) if f is None else f
    n_full, segs = stage_block_counts(layers, est.num_layers)
    tot = fixed_flops(est) + n_full * f[est.N]
    for s, k in zip(segs, counts):
        tot = tot + s * f[np.asarray(k, dtype=int)]
    return tot / est.baseline_flops


def feasible_schedules(est: ComputeEstimator, layers: Sequence[int], floor: int, cap: float, tau: float = 0.01) -> np.ndarray:
    """All integer (K_1..K_m) with N > K_1 > ... > K_m >= floor and cap - tau <= FLOPs <= cap. Returns int array (n, m).
    Supports m = 2 and m = 3 (exhaustive); larger m needs the DP formulation in the design document (not implemented)."""
    N, m = est.N, len(layers)
    f = block_flops_table(est)
    n_full, segs = stage_block_counts(layers, est.num_layers)
    base = fixed_flops(est) + n_full * f[N]
    lo, hi = (cap - tau) * est.baseline_flops - 1e-6, cap * est.baseline_flops + 1e-6
    ks = np.arange(floor, N)
    rows: List[np.ndarray] = []
    if m == 2:
        for k1 in range(floor + 1, N):
            tot = base + segs[0] * f[k1] + segs[1] * f[ks]
            ok = (ks < k1) & (tot >= lo) & (tot <= hi)
            if ok.any():
                rows.append(np.stack([np.full(ok.sum(), k1), ks[ok]], 1))
    elif m == 3:
        K2, K3 = np.meshgrid(ks, ks, indexing="ij")
        c23 = segs[1] * f[K2] + segs[2] * f[K3]
        m23 = K2 > K3
        for k1 in range(floor + 2, N):
            tot = base + segs[0] * f[k1] + c23
            ok = m23 & (K2 < k1) & (tot >= lo) & (tot <= hi)
            ii, jj = np.nonzero(ok)
            if len(ii):
                rows.append(np.stack([np.full(len(ii), k1), ks[ii], ks[jj]], 1))
    else:
        raise NotImplementedError("exhaustive feasible-set enumeration implemented for 2 or 3 stages")
    return np.concatenate(rows, 0) if rows else np.zeros((0, m), dtype=int)


# ----------------------------------------------------------------------------- sensitivity tables

def _iso_noninc(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators fit that is non-increasing in the index (loss must not grow with more tokens)."""
    from sklearn.isotonic import IsotonicRegression
    x = np.arange(len(y))
    return IsotonicRegression(increasing=False).fit_transform(x, y)


@dataclass
class ProxyModel:
    """Interpolated, monotone-regularised tables on the integer token axis (0..N)."""
    N: int
    d1: Dict[int, np.ndarray]                     # layer -> loss[k], k = 0..N   (loss(N) = 0)
    d2: Dict[Tuple[int, int], np.ndarray]         # (a, b) -> loss[ka, kb]

    def loss(self, layers: Sequence[int], K: np.ndarray) -> np.ndarray:
        K = np.asarray(K, dtype=int).reshape(-1, len(layers))
        tot = np.zeros(len(K))
        if len(layers) == 1:
            return self.d1[layers[0]][K[:, 0]]
        for j in range(1, len(layers)):
            a, b = layers[j - 1], layers[j]
            tot += self.d2[(a, b)][K[:, j - 1], K[:, j]]
            if j >= 2:
                tot -= self.d1[a][K[:, j - 1]]          # chain telescoping: subtract the shared prefix term
        return tot


@dataclass
class SensitivityTables:
    grid: List[int]
    singles: List[int]                            # layers with a single-stage curve
    pairs: List[Tuple[int, int]]                  # consecutive layer pairs with a two-stage table
    n_images: int
    per_image: Dict[str, np.ndarray] = field(default_factory=dict)      # key -> per-image KL on the calibration set
    N: int = 196

    @staticmethod
    def key_single(a: int, k: int) -> str:
        return f"s|{a}|{k}"

    @staticmethod
    def key_pair(a: int, ka: int, b: int, kb: int) -> str:
        return f"p|{a}|{ka}|{b}|{kb}"

    def required_keys(self) -> List[str]:
        keys = [self.key_single(a, k) for a in self.singles for k in self.grid]
        for a, b in self.pairs:
            keys += [self.key_pair(a, ka, b, kb) for ka in self.grid for kb in self.grid if ka > kb]
        return keys

    def build(self, idx: Optional[np.ndarray] = None) -> ProxyModel:
        """Fit the proxy from per-image KL means (optionally over a bootstrap resample `idx` of calibration images)."""
        mean = (lambda v: float(v.mean())) if idx is None else (lambda v: float(v[idx].mean()))
        N = self.N
        ext = np.array(list(self.grid) + [N], dtype=float)
        lg = np.log(ext)
        ks = np.arange(0, N + 1)
        lk = np.log(np.maximum(ks, 1))
        d1: Dict[int, np.ndarray] = {}
        for a in self.singles:
            y = np.array([mean(self.per_image[self.key_single(a, k)]) for k in self.grid] + [0.0])
            y = _iso_noninc(y)
            d1[a] = np.interp(lk, lg, y)
            d1[a][N] = 0.0
        d2: Dict[Tuple[int, int], np.ndarray] = {}
        from scipy.interpolate import RegularGridInterpolator
        for a, b in self.pairs:
            G = len(ext)
            M = np.zeros((G, G))
            for i, ka in enumerate(ext):
                for j, kb in enumerate(ext):
                    if ka == N:
                        M[i, j] = float(np.interp(np.log(kb), lg, self._ysingle(b, idx, mean))) if kb < N else 0.0
                    elif kb >= ka:
                        M[i, j] = float(np.interp(np.log(ka), lg, self._ysingle(a, idx, mean)))
                    else:
                        M[i, j] = mean(self.per_image[self.key_pair(a, int(ka), b, int(kb))])
            for _ in range(2):                                    # alternate monotone projections along both axes
                for i in range(G):
                    M[i, :] = _iso_noninc(M[i, :])
                for j in range(G):
                    M[:, j] = _iso_noninc(M[:, j])
            itp = RegularGridInterpolator((lg, lg), M, bounds_error=False, fill_value=None)
            A, Bk = np.meshgrid(np.maximum(ks, 1), np.maximum(ks, 1), indexing="ij")
            full = itp(np.stack([np.log(A).ravel(), np.log(Bk).ravel()], 1)).reshape(A.shape)
            d2[(a, b)] = full
        return ProxyModel(N=N, d1=d1, d2=d2)

    def _ysingle(self, a: int, idx, mean) -> np.ndarray:
        y = np.array([mean(self.per_image[self.key_single(a, k)]) for k in self.grid] + [0.0])
        return _iso_noninc(y)

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for k in sorted(self.per_image):
            h.update(k.encode()); h.update(np.round(self.per_image[k].astype(np.float64), 9).tobytes())
        return h.hexdigest()


# ----------------------------------------------------------------------------- ranking and selection

def rank_diverse(feasible: np.ndarray, loss: np.ndarray, M: int, min_dist: int) -> List[int]:
    """Indices (into `feasible`) of the M lowest-loss schedules with pairwise L_inf distance >= min_dist (greedy, deterministic)."""
    order = np.argsort(loss, kind="stable")                                  # deterministic: loss, then enumeration order
    chosen: List[int] = []
    for i in order:
        if all(np.abs(feasible[i] - feasible[j]).max() >= min_dist for j in chosen):
            chosen.append(int(i))
            if len(chosen) == M:
                break
    return chosen


def select_verified(cands: List[Dict[str, Any]], tie_eps: float = 0.002) -> Dict[str, Any]:
    """Pick the candidate with the highest verified calibration agreement; ties (within tie_eps) -> lower proxy, larger final count."""
    best = max(c["cal_agreement"] for c in cands)
    tied = [c for c in cands if best - c["cal_agreement"] <= tie_eps + 1e-12]
    return sorted(tied, key=lambda c: (c["proxy_loss"], -c["counts"][-1], c["counts"]))[0]


@dataclass
class SelectionResult:
    status: str                                   # 'selected' | 'infeasible'
    counts: Optional[List[int]]
    candidates: List[Dict[str, Any]]
    n_feasible: int
    reason: str = ""


def select_schedule(proxy: ProxyModel, est: ComputeEstimator, layers: Sequence[int], floor: int, cap: float, tau: float,
                    verify_fn: Callable[[List[int]], float], M: int = 8, min_dist: int = 6, tie_eps: float = 0.002) -> SelectionResult:
    """Full selection for one (pattern, budget, floor). `verify_fn(counts)` must use CALIBRATION data only."""
    feas = feasible_schedules(est, layers, floor, cap, tau)
    if len(feas) == 0:
        return SelectionResult("infeasible", None, [], 0, f"no integer schedule with final count >= {floor} and FLOPs in [{cap - tau:.4f}, {cap:.4f}]")
    loss = proxy.loss(layers, feas)
    pick = rank_diverse(feas, loss, M, min_dist)
    cands = []
    for rank, i in enumerate(pick):
        counts = [int(x) for x in feas[i]]
        cands.append({"rank": rank, "counts": counts, "proxy_loss": float(loss[i]),
                      "flops": float(normalized_flops(est, layers, counts)), "cal_agreement": float(verify_fn(counts))})
    sel = select_verified(cands, tie_eps)
    for c in cands:
        c["selected"] = c["counts"] == sel["counts"]
    return SelectionResult("selected", sel["counts"], cands, int(len(feas)))
