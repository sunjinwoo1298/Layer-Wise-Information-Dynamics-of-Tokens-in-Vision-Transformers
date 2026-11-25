"""
Deterministic development-set configuration selection (frozen protocol, section 5). Pure functions; no data access.
Agreement is the integer count of images agreeing with the unpruned prediction; ties use exact integer arithmetic.
"""

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def kappa_grid() -> List[float]:
    return [round(0.20 + 0.005 * i, 3) for i in range(161)]


def tome_grid() -> List[Tuple[int, float]]:
    return [(r, infl) for r in range(1, 41) for infl in (-1.0, -0.5, 0.0, 0.5, 1.0)]       # generation order: r ascending, then inflection


def feasible(candidates: Sequence[Dict[str, Any]], target_pct: float, tol_pp: float = 1.0) -> List[Dict[str, Any]]:
    return [c for c in candidates if abs(round(c["flops_pct"], 6) - target_pct) <= tol_pp + 1e-9]


def select(candidates: Sequence[Dict[str, Any]], n: int, target_pct: float, method: str) -> Dict[str, Any]:
    """candidates: dicts with keys config (r,infl)|kappa, flops_pct, agree_count, gen_index. Returns dict(chosen, tie_set, ranked)."""
    if not candidates:
        return {"chosen": None, "tie_set": [], "ranked": []}
    best = max(c["agree_count"] for c in candidates)
    tie = [c for c in candidates if (best - c["agree_count"]) * 500 < n]          # < 0.2 pp relative to best, exact integers

    def key(c):
        gap = abs(round(c["flops_pct"], 6) - target_pct)
        if method == "tome":
            r, infl = c["config"]
            return (gap, r, infl, c["gen_index"])
        return (gap, c["kappa"], c["gen_index"])

    ranked = sorted(tie, key=key)
    return {"chosen": ranked[0], "tie_set": ranked, "best_count": best, "n_candidates": len(candidates)}
