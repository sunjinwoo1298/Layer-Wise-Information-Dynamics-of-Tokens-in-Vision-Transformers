"""
Decoupled Evaluation Metrics for Compressed Transformer Assessment.
Strictly separates evaluation metrics from token-selection metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn.functional as F


@dataclass
class EvaluationMetrics:
    top1_accuracy: float
    top5_accuracy: float
    prediction_agreement: float
    confidence_retention: float
    kl_divergence: float
    js_divergence: float
    cosine_distance: float
    sample_count: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "prediction_agreement": self.prediction_agreement,
            "confidence_retention": self.confidence_retention,
            "kl_divergence": self.kl_divergence,
            "js_divergence": self.js_divergence,
            "cosine_distance": self.cosine_distance,
            "sample_count": self.sample_count,
        }


def compute_sample_level_metrics(
    orig_logits: torch.Tensor,
    compressed_logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, List[float]]:
    """
    Computes per-sample metrics to preserve raw measurements and enable statistical confidence intervals.
    """
    orig_logits = orig_logits.detach()
    compressed_logits = compressed_logits.detach()

    orig_probs = F.softmax(orig_logits, dim=-1)
    comp_probs = F.softmax(compressed_logits, dim=-1)

    orig_preds = orig_logits.argmax(dim=-1)
    comp_preds = compressed_logits.argmax(dim=-1)

    # 1. Prediction Agreement
    agreement = (orig_preds == comp_preds).float().cpu().tolist()

    # 2. Top-class Confidence Retention: comp_p(orig_y) / orig_p(orig_y)
    orig_top_prob = orig_probs.gather(1, orig_preds.unsqueeze(1)).squeeze(1)
    comp_top_prob = comp_probs.gather(1, orig_preds.unsqueeze(1)).squeeze(1)
    conf_retention = (comp_top_prob / (orig_top_prob + 1e-12)).clamp(max=2.0).cpu().tolist()

    # 3. Output KL Divergence: D_KL(orig || comp)
    comp_log_probs = F.log_softmax(compressed_logits, dim=-1)
    kl_div = F.kl_div(comp_log_probs, orig_probs, reduction="none", log_target=False).sum(dim=-1).cpu().tolist()

    # 4. Jensen-Shannon Divergence: JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    m_probs = 0.5 * (orig_probs + comp_probs)
    log_m = torch.log(m_probs + 1e-12)
    orig_log_probs = torch.log(orig_probs + 1e-12)
    jsd_orig = F.kl_div(log_m, orig_probs, reduction="none", log_target=False).sum(dim=-1)
    jsd_comp = F.kl_div(log_m, comp_probs, reduction="none", log_target=False).sum(dim=-1)
    js_div = (0.5 * (jsd_orig + jsd_comp)).clamp(min=0.0).cpu().tolist()

    # 5. Cosine Distance
    cosine_sim = F.cosine_similarity(orig_logits, compressed_logits, dim=-1)
    cosine_dist = (1.0 - cosine_sim).clamp(min=0.0).cpu().tolist()

    results: Dict[str, List[float]] = {
        "prediction_agreement": agreement,
        "confidence_retention": conf_retention,
        "kl_divergence": kl_div,
        "js_divergence": js_div,
        "cosine_distance": cosine_dist,
    }

    # 6. Accuracy against ground truth targets (if available)
    if targets is not None:
        targets = targets.to(comp_preds.device)
        top1_acc = (comp_preds == targets).float().cpu().tolist()
        results["top1_accuracy"] = top1_acc
        
        # Top-5 accuracy
        if orig_logits.shape[-1] >= 5:
            top5_preds = compressed_logits.topk(5, dim=-1).indices
            top5_acc = (top5_preds == targets.unsqueeze(1)).any(dim=-1).float().cpu().tolist()
            results["top5_accuracy"] = top5_acc
        else:
            results["top5_accuracy"] = top1_acc

    return results


def compute_prediction_metrics(
    orig_logits: torch.Tensor,
    compressed_logits: torch.Tensor,
    targets: Optional[torch.Tensor] = None,
) -> EvaluationMetrics:
    """Computes aggregated batch/dataset level EvaluationMetrics."""
    sample_metrics = compute_sample_level_metrics(orig_logits, compressed_logits, targets)
    n = len(sample_metrics["prediction_agreement"])

    def _mean(lst: List[float]) -> float:
        return sum(lst) / max(len(lst), 1)

    top1 = _mean(sample_metrics.get("top1_accuracy", [0.0]))
    top5 = _mean(sample_metrics.get("top5_accuracy", [0.0]))

    return EvaluationMetrics(
        top1_accuracy=top1,
        top5_accuracy=top5,
        prediction_agreement=_mean(sample_metrics["prediction_agreement"]),
        confidence_retention=_mean(sample_metrics["confidence_retention"]),
        kl_divergence=_mean(sample_metrics["kl_divergence"]),
        js_divergence=_mean(sample_metrics["js_divergence"]),
        cosine_distance=_mean(sample_metrics["cosine_distance"]),
        sample_count=n,
    )
