"""
Information Retention Curves P_l(k) and Marginal Efficiency (Delta P / Delta C) Calculations.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np


@dataclass
class MarginalEfficiencyInterval:
    layer_idx: int
    k_start: int
    k_end: int
    p_start: float
    p_end: float
    c_start: float
    c_end: float
    delta_p: float
    delta_c: float
    marginal_gain: float  # delta_p / delta_c

    def to_dict(self) -> Dict:
        return {
            "layer_idx": self.layer_idx,
            "k_interval": f"{self.k_start}->{self.k_end}",
            "k_start": self.k_start,
            "k_end": self.k_end,
            "p_start": self.p_start,
            "p_end": self.p_end,
            "c_start": self.c_start,
            "c_end": self.c_end,
            "delta_p": self.delta_p,
            "delta_c": self.delta_c,
            "marginal_gain": self.marginal_gain,
        }


@dataclass
class InformationCurveResult:
    layer_idx: int
    tokens: List[int]
    normalized_compute: List[float]
    performance_values: List[float]
    metric_name: str
    marginal_intervals: List[MarginalEfficiencyInterval]


def calculate_marginal_efficiency(
    layer_df: pd.DataFrame,
    performance_col: str = "prediction_agreement",
    compute_col: str = "normalized_flops",
    is_lower_better: bool = False,
) -> List[MarginalEfficiencyInterval]:
    """
    Computes marginal efficiency intervals (Delta P / Delta C) between adjacent token retention budgets.
    layer_df must contain rows for a single layer sorted by retained_tokens ascending.
    """
    df = layer_df.sort_values(by="retained_tokens", ascending=True).reset_index(drop=True)
    intervals = []

    for i in range(len(df) - 1):
        row_start = df.iloc[i]
        row_end = df.iloc[i + 1]

        k_start = int(row_start["retained_tokens"])
        k_end = int(row_end["retained_tokens"])

        p_start = float(row_start[performance_col])
        p_end = float(row_end[performance_col])

        # If lower is better (e.g. KL divergence), flip difference so improvement is positive
        if is_lower_better:
            delta_p = p_start - p_end
        else:
            delta_p = p_end - p_start

        c_start = float(row_start[compute_col])
        c_end = float(row_end[compute_col])
        delta_c = c_end - c_start

        marginal_gain = delta_p / (delta_c + 1e-12) if abs(delta_c) > 1e-12 else 0.0

        layer_idx = int(row_start["layer"]) if "layer" in row_start else 0

        intervals.append(
            MarginalEfficiencyInterval(
                layer_idx=layer_idx,
                k_start=k_start,
                k_end=k_end,
                p_start=p_start,
                p_end=p_end,
                c_start=c_start,
                c_end=c_end,
                delta_p=delta_p,
                delta_c=delta_c,
                marginal_gain=marginal_gain,
            )
        )

    return intervals
