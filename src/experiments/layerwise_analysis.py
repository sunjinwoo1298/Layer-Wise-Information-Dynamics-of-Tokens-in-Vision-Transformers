"""
Statistical Aggregation and Layer-Wise Empirical Dynamics Analysis.
Implements Wilson Score Confidence Intervals for proportions and McNemar Paired Tests against Random.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Computes Wilson Score Interval for binomial proportion metrics (Agreement, Accuracy).
    Scientifically appropriate for discrete binary outcomes where normal approximation fails near 0/1.
    """
    if total <= 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}

    p_hat = float(successes) / float(total)
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    z2 = z ** 2
    n = float(total)

    denominator = 1.0 + (z2 / n)
    center = (p_hat + (z2 / (2.0 * n))) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * (n ** 2))))

    ci_lower = max(0.0, float(center - margin))
    ci_upper = min(1.0, float(center + margin))

    # Binomial standard error: sqrt(p(1-p)/n)
    sem = float(np.sqrt(p_hat * (1.0 - p_hat) / n))

    return {
        "mean": p_hat,
        "std": float(np.sqrt(p_hat * (1.0 - p_hat))),
        "sem": sem,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": total,
    }


def calculate_continuous_ci(data: List[float], confidence: float = 0.95) -> Dict[str, float]:
    """Calculates Student's t-distribution confidence interval for continuous metrics."""
    arr = np.array(data, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "sem": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    
    mean = float(np.mean(arr))
    if n == 1:
        return {"mean": mean, "std": 0.0, "sem": 0.0, "ci_lower": mean, "ci_upper": mean, "n": 1}

    std = float(np.std(arr, ddof=1))
    sem = float(stats.sem(arr))
    
    t_val = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    ci_lower = float(mean - t_val * sem)
    ci_upper = float(mean + t_val * sem)

    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n": n,
    }


def mcnemar_paired_test(
    strat_agreements: List[float],
    rand_agreements: List[float],
) -> Dict[str, Any]:
    """
    Performs McNemar's Paired Test on matched sample agreement vectors to test if
    the evaluated strategy statistically outperforms the random baseline.
    """
    n = min(len(strat_agreements), len(rand_agreements))
    if n == 0:
        return {"mcnemar_stat": 0.0, "mcnemar_pval": 1.0, "stat_sig_vs_random": False, "b_strat_better": 0, "c_rand_better": 0}

    s_arr = np.array(strat_agreements[:n]) > 0.5
    r_arr = np.array(rand_agreements[:n]) > 0.5

    # b: Strat agreed, Random disagreed
    b = int(np.sum(s_arr & (~r_arr)))
    # c: Strat disagreed, Random agreed
    c = int(np.sum((~s_arr) & r_arr))

    total_discordant = b + c
    if total_discordant == 0:
        return {"mcnemar_stat": 0.0, "mcnemar_pval": 1.0, "stat_sig_vs_random": False, "b_strat_better": b, "c_rand_better": c}

    # Exact binomial test for small sample counts, continuity-corrected chi2 otherwise
    if total_discordant < 25:
        # Exact two-sided binomial test under H0: p = 0.5
        pval = float(2.0 * stats.binom.cdf(min(b, c), total_discordant, 0.5))
        pval = min(1.0, pval)
        stat = float(total_discordant)
    else:
        # Edwards' continuity correction: (|b - c| - 1)^2 / (b + c)
        stat = float(((abs(b - c) - 1.0) ** 2) / (b + c))
        pval = float(1.0 - stats.chi2.cdf(stat, df=1))

    is_sig = bool(pval < 0.05 and b > c)

    return {
        "mcnemar_stat": stat,
        "mcnemar_pval": pval,
        "stat_sig_vs_random": is_sig,
        "b_strat_better": b,
        "c_rand_better": c,
    }


def aggregate_experiment_results(raw_records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Transforms per-evaluation records into a statistical summary DataFrame with Wilson CIs and McNemar tests.
    """
    summary_rows = []

    # Map for finding matched random baseline records
    random_records_map = {}
    for rec in raw_records:
        if rec.get("scorer") == "random":
            key = (rec.get("layer"), rec.get("retained_tokens"))
            random_records_map[key] = rec.get("sample_metrics", {}).get("prediction_agreement", [])

    for rec in raw_records:
        row = {
            "model": rec.get("model"),
            "dataset": rec.get("dataset"),
            "seed": rec.get("seed"),
            "scorer": rec.get("scorer"),
            "layer": rec.get("layer"),
            "original_tokens": rec.get("original_tokens"),
            "retained_tokens": rec.get("retained_tokens"),
            "retention_ratio": rec.get("retention_ratio"),
            "total_flops": rec.get("total_flops"),
            "normalized_flops": rec.get("normalized_flops"),
            "mean_latency_ms": rec.get("mean_latency_ms"),
            "p50_latency_ms": rec.get("p50_latency_ms"),
            "p95_latency_ms": rec.get("p95_latency_ms"),
            "peak_memory_mb": rec.get("peak_memory_mb"),
        }

        sample_metrics = rec.get("sample_metrics", {})
        
        # 1. Binary proportion metrics -> Wilson Score Interval
        proportion_metrics = ["prediction_agreement", "top1_accuracy", "top5_accuracy"]
        for m_name in proportion_metrics:
            if m_name in sample_metrics and len(sample_metrics[m_name]) > 0:
                vals = sample_metrics[m_name]
                successes = int(sum(1 for v in vals if v > 0.5))
                w_stats = wilson_score_interval(successes, len(vals))
                row[f"{m_name}"] = w_stats["mean"]
                row[f"{m_name}_std"] = w_stats["std"]
                row[f"{m_name}_sem"] = w_stats["sem"]
                row[f"{m_name}_ci_lower"] = w_stats["ci_lower"]
                row[f"{m_name}_ci_upper"] = w_stats["ci_upper"]
                row[f"{m_name}_ci_method"] = "wilson"

        # 2. Continuous metrics -> Student's t Interval
        continuous_metrics = ["confidence_retention", "kl_divergence", "js_divergence", "cosine_distance"]
        for m_name in continuous_metrics:
            if m_name in sample_metrics and len(sample_metrics[m_name]) > 0:
                vals = sample_metrics[m_name]
                c_stats = calculate_continuous_ci(vals)
                row[f"{m_name}"] = c_stats["mean"]
                row[f"{m_name}_std"] = c_stats["std"]
                row[f"{m_name}_sem"] = c_stats["sem"]
                row[f"{m_name}_ci_lower"] = c_stats["ci_lower"]
                row[f"{m_name}_ci_upper"] = c_stats["ci_upper"]
                row[f"{m_name}_ci_method"] = "student_t"

        # 3. McNemar's Paired Test against Random Baseline
        scorer = rec.get("scorer")
        layer = rec.get("layer")
        k = rec.get("retained_tokens")
        if scorer != "random" and (layer, k) in random_records_map:
            rand_agreements = random_records_map[(layer, k)]
            strat_agreements = sample_metrics.get("prediction_agreement", [])
            mcn_res = mcnemar_paired_test(strat_agreements, rand_agreements)
            row["mcnemar_stat"] = mcn_res["mcnemar_stat"]
            row["mcnemar_pval"] = mcn_res["mcnemar_pval"]
            row["stat_sig_vs_random"] = mcn_res["stat_sig_vs_random"]
            row["b_strat_better"] = mcn_res["b_strat_better"]
            row["c_rand_better"] = mcn_res["c_rand_better"]
        else:
            row["mcnemar_stat"] = np.nan
            row["mcnemar_pval"] = np.nan
            row["stat_sig_vs_random"] = False
            row["b_strat_better"] = 0
            row["c_rand_better"] = 0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def format_research_summary(df: pd.DataFrame, primary_metric: str = "prediction_agreement") -> str:
    """Generates an objective, non-biased empirical summary with Wilson CIs and McNemar Significance."""
    lines = []
    lines.append("=" * 78)
    lines.append(" EMPIRICAL RESEARCH SUMMARY: LAYER-WISE TOKEN DYNAMICS")
    lines.append("=" * 78)
    lines.append(f"Primary Metric: {primary_metric} (Wilson 95% Confidence Intervals)")
    lines.append("")

    scorers = df["scorer"].unique() if "scorer" in df.columns else ["default"]
    layers = sorted(df["layer"].unique()) if "layer" in df.columns else []

    for scorer in scorers:
        lines.append(f"--- Strategy / Scorer: {scorer} ---")
        sub_df = df[df["scorer"] == scorer] if "scorer" in df.columns else df

        for layer in layers:
            layer_df = sub_df[sub_df["layer"] == layer].sort_values(by="retention_ratio", ascending=False)
            lines.append(f"Layer {layer}:")
            for _, r in layer_df.iterrows():
                ratio = r["retention_ratio"]
                k = int(r["retained_tokens"])
                perf = r.get(primary_metric, np.nan)
                ci_low = r.get(f"{primary_metric}_ci_lower", np.nan)
                ci_high = r.get(f"{primary_metric}_ci_upper", np.nan)
                norm_flops = r.get("normalized_flops", np.nan)
                lat = r.get("mean_latency_ms", np.nan)

                mcn_sig = ""
                if scorer != "random" and not pd.isna(r.get("mcnemar_pval")):
                    pval = r.get("mcnemar_pval")
                    sig_flag = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
                    mcn_sig = f" | vs Rand p={pval:.4f} ({sig_flag})"

                lines.append(
                    f"  {ratio*100:5.1f}% tokens (k={k:3d}) -> "
                    f"{primary_metric}: {perf*100:6.2f}% [95% CI: {ci_low*100:5.1f}% - {ci_high*100:5.1f}%] | "
                    f"Norm FLOPs: {norm_flops*100:5.1f}% | Latency: {lat:5.2f}ms{mcn_sig}"
                )
            lines.append("")

    lines.append("=" * 78)
    return "\n".join(lines)
