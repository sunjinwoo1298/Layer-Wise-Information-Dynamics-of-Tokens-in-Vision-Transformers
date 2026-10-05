"""
Experiment Orchestration Engine: Manages Experiment Lifecycle, Caching, Sanity Checks, and Logging.
"""

from dataclasses import dataclass, field, asdict
import json
import os
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import torch

from ..models.vit_wrapper import ViTWrapper
from ..data.dataset import get_dataloader, DatasetConfig
from ..importance.base import TokenImportanceScorer
from ..importance.random_scorer import RandomScorer
from ..importance.prediction_kl import PredictionKLDivergenceScorer
from ..importance.attention_scorer import AttentionScorer
from ..compression.token_pruning import TokenPruner
from ..metrics.prediction import compute_sample_level_metrics
from ..metrics.compute import ComputeEstimator
from ..metrics.information import calculate_marginal_efficiency
from .layerwise_analysis import aggregate_experiment_results, format_research_summary


@dataclass
class ExperimentConfig:
    model_name: str = "deit_tiny_patch16_224"
    pretrained: bool = True
    dataset_name: str = "imagenet_val"  # "imagenet_val", "cifar100", "synthetic"
    data_dir: Optional[str] = "./data"
    num_samples: int = 250
    batch_size: int = 25
    seed: int = 42
    layers: List[int] = field(default_factory=lambda: [2, 4, 6, 8, 10, 12])
    # Granular sweep of retention ratios (10 points)
    retention_ratios: List[float] = field(
        default_factory=lambda: [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    )
    scorers: List[str] = field(
        default_factory=lambda: ["random", "prediction_kl_approximate", "attention_cls"]
    )
    measure_latency: bool = True
    output_dir: str = "./results/experiment_001"
    device: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentRunner:
    """
    Executes the Phase 1 layer-wise token information dynamics experimental protocol.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._set_seed(config.seed)

        os.makedirs(self.config.output_dir, exist_ok=True)

        # 1. Initialize ViT Model Wrapper
        self.model_wrapper = ViTWrapper(
            model_name=config.model_name,
            pretrained=config.pretrained,
            device=config.device,
        )

        # 2. Initialize DataLoader
        ds_config = DatasetConfig(
            name=config.dataset_name,
            data_dir=config.data_dir,
            num_samples=config.num_samples,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        self.dataloader, self.actual_samples = get_dataloader(ds_config)

        # 3. Initialize Pruner & Compute Estimator
        self.pruner = TokenPruner(preserve_spatial_order=True)
        self.compute_estimator = ComputeEstimator(
            num_layers=self.model_wrapper.num_layers,
            hidden_dim=self.model_wrapper.hidden_dim,
            num_heads=self.model_wrapper.num_heads,
            num_original_tokens=self.model_wrapper.num_original_tokens,
            num_classes=self.model_wrapper.num_classes,
            num_prefix_tokens=self.model_wrapper.num_prefix_tokens,
            patch_size=self.model_wrapper.patch_size,
        )

        # 4. Instantiate Token Importance Scorers
        self.scorers: Dict[str, TokenImportanceScorer] = {}
        for s_name in config.scorers:
            if s_name == "random":
                self.scorers[s_name] = RandomScorer(seed=config.seed)
            elif s_name == "prediction_kl_exact":
                self.scorers[s_name] = PredictionKLDivergenceScorer(mode="exact")
            elif s_name == "prediction_kl_approximate":
                self.scorers[s_name] = PredictionKLDivergenceScorer(mode="approximate")
            elif s_name == "attention_cls":
                self.scorers[s_name] = AttentionScorer()
            else:
                raise ValueError(f"Unknown scorer: {s_name}")

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def run_baseline_sanity_check(self) -> bool:
        """
        Research Sanity Check: Verifies that when k = N (100% token retention),
        the intercepted forward pass matches the unpruned baseline model with zero numerical deviation.
        """
        print("[Sanity Check] Verifying baseline consistency across all layers...")
        sample_batch, _ = next(iter(self.dataloader))
        sample_batch = sample_batch.to(self.model_wrapper.device)

        with torch.no_grad():
            orig_logits = self.model_wrapper(sample_batch)

            for l in range(1, self.model_wrapper.num_layers + 1):
                pref, vis = self.model_wrapper.extract_tokens_at_layer(sample_batch, l)
                intercepted_logits = self.model_wrapper.forward_from_pruned_tokens(pref, vis, l)
                max_diff = (orig_logits - intercepted_logits).abs().max().item()
                if max_diff > 1e-4:
                    raise AssertionError(
                        f"Baseline sanity check failed at layer {l}! Max diff: {max_diff:.6f}"
                    )

        print("[Sanity Check] Baseline consistency verified (max diff < 1e-4).")
        return True

    def run(self) -> Dict[str, Any]:
        """Runs the complete layer-wise token measurement protocol."""
        self.run_baseline_sanity_check()

        N = self.model_wrapper.num_original_tokens
        layers = self.config.layers
        ratios = sorted(self.config.retention_ratios, reverse=True)

        raw_evaluation_records: List[Dict[str, Any]] = []

        # Dummy input for latency benchmarking
        dummy_x = torch.randn(
            1, 3, self.model_wrapper.img_size[0], self.model_wrapper.img_size[1],
            device=self.model_wrapper.device
        )

        print(f"\n[Experiment] Starting Layer-Wise Token Dynamics Evaluation")
        print(f"Model: {self.config.model_name} | Dataset: {self.config.dataset_name} ({self.actual_samples} samples)")
        print(f"Layers: {layers} | Retention Ratios ({len(ratios)} points): {ratios}")
        print(f"Scorers: {list(self.scorers.keys())}\n")

        for scorer_name, scorer in self.scorers.items():
            print(f">>> Evaluating Scorer: {scorer_name}")

            for layer_idx in layers:
                for ratio in ratios:
                    k = max(1, int(round(ratio * N)))
                    
                    # 1. Theoretical FLOPs
                    flops_res = self.compute_estimator.compute_flops(layer_idx, k)

                    # 2. Empirical Full Forward Pass Latency & Memory
                    if self.config.measure_latency:
                        latency_res = self.compute_estimator.measure_latency_and_memory(
                            self.model_wrapper,
                            dummy_x,
                            layer_idx=layer_idx,
                            k=k,
                            scorer=scorer,
                            pruner=self.pruner,
                            num_runs=10,
                        )
                    else:
                        latency_res = None

                    # 3. Model Inference & Decoupled Metrics Collection across Dataset
                    collected_sample_metrics: Dict[str, List[float]] = {}

                    for batch_images, batch_targets in self.dataloader:
                        batch_images = batch_images.to(self.model_wrapper.device)
                        batch_targets = batch_targets.to(self.model_wrapper.device)

                        # Original uncompressed logits
                        with torch.no_grad():
                            orig_logits = self.model_wrapper(batch_images)

                        # Extract intermediate tokens at layer_idx
                        prefix_tok, vis_tok = self.model_wrapper.extract_tokens_at_layer(
                            batch_images, layer_idx
                        )

                        # Token selection scores S_l(t)
                        scores = scorer.score_tokens(
                            self.model_wrapper, batch_images, layer_idx, prefix_tok, vis_tok, orig_logits
                        )

                        # Prune visual tokens to top-k
                        retained_vis, _ = self.pruner.prune(vis_tok, scores, k)

                        # Forward pass with retained tokens
                        with torch.no_grad():
                            comp_logits = self.model_wrapper.forward_from_pruned_tokens(
                                prefix_tok, retained_vis, layer_idx
                            )

                        # Decoupled Evaluation Metrics
                        batch_metrics = compute_sample_level_metrics(
                            orig_logits, comp_logits, batch_targets
                        )

                        for m_key, m_vals in batch_metrics.items():
                            collected_sample_metrics.setdefault(m_key, []).extend(m_vals)

                    # Record evaluation summary entry
                    rec = {
                        "model": self.config.model_name,
                        "dataset": self.config.dataset_name,
                        "seed": self.config.seed,
                        "scorer": scorer_name,
                        "layer": layer_idx,
                        "original_tokens": N,
                        "retained_tokens": k,
                        "retention_ratio": ratio,
                        "total_flops": flops_res.total_flops,
                        "normalized_flops": flops_res.normalized_flops,
                        "mean_latency_ms": latency_res.mean_latency_ms if latency_res else None,
                        "p50_latency_ms": latency_res.p50_latency_ms if latency_res else None,
                        "p95_latency_ms": latency_res.p95_latency_ms if latency_res else None,
                        "peak_memory_mb": latency_res.peak_memory_mb if latency_res else None,
                        "sample_metrics": collected_sample_metrics,
                    }
                    raw_evaluation_records.append(rec)

        # 4. Statistical Aggregation with Wilson CIs and McNemar Paired Tests
        summary_df = aggregate_experiment_results(raw_evaluation_records)

        # 5. Marginal Efficiency Calculations (Delta P / Delta C)
        marginal_intervals_all = []
        for scorer_name in self.scorers.keys():
            scorer_df = summary_df[summary_df["scorer"] == scorer_name]
            for l in layers:
                layer_sub = scorer_df[scorer_df["layer"] == l]
                intervals = calculate_marginal_efficiency(layer_sub, performance_col="prediction_agreement")
                for item in intervals:
                    d = item.to_dict()
                    d["scorer"] = scorer_name
                    marginal_intervals_all.append(d)

        marginal_df = pd.DataFrame(marginal_intervals_all)

        # 6. Save Structured Results
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        metrics_csv_path = os.path.join(self.config.output_dir, "layer_metrics.csv")
        summary_df.to_csv(metrics_csv_path, index=False)

        marginal_csv_path = os.path.join(self.config.output_dir, "marginal_efficiency.csv")
        marginal_df.to_csv(marginal_csv_path, index=False)

        summary_text = format_research_summary(summary_df)
        summary_txt_path = os.path.join(self.config.output_dir, "summary.txt")
        with open(summary_txt_path, "w") as f:
            f.write(summary_text)

        print(summary_text)
        print(f"\n[Results Saved] Results exported to {self.config.output_dir}")

        return {
            "summary_df": summary_df,
            "marginal_df": marginal_df,
            "config": self.config.to_dict(),
        }
