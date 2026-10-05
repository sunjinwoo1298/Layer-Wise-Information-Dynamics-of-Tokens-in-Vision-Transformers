"""
Compute Engine: Disentangles Theoretical FLOPs from Empirical Wall-Clock Latency & Peak Memory.
Measures full end-to-end forward pass including layer interception, scoring, and downstream execution.
"""

from dataclasses import dataclass
import time
from typing import Dict, Tuple, Optional, Any
import torch
import torch.nn as nn


@dataclass
class TheoreticalFLOPsResult:
    layer_idx: int
    retained_tokens: int
    total_tokens: int
    embed_flops: float
    uncompressed_layers_flops: float
    compressed_layers_flops: float
    head_flops: float
    total_flops: float
    baseline_flops: float
    normalized_flops: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "layer_idx": self.layer_idx,
            "retained_tokens": self.retained_tokens,
            "total_tokens": self.total_tokens,
            "embed_flops": self.embed_flops,
            "uncompressed_layers_flops": self.uncompressed_layers_flops,
            "compressed_layers_flops": self.compressed_layers_flops,
            "head_flops": self.head_flops,
            "total_flops": self.total_flops,
            "baseline_flops": self.baseline_flops,
            "normalized_flops": self.normalized_flops,
        }


@dataclass
class EmpiricalLatencyResult:
    mean_latency_ms: float
    std_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    peak_memory_mb: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_latency_ms": self.mean_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "peak_memory_mb": self.peak_memory_mb,
        }


class ComputeEstimator:
    """
    Calculates exact theoretical FLOPs and measures empirical full forward-pass latency/memory.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        num_original_tokens: int,
        num_classes: int = 1000,
        num_prefix_tokens: int = 1,
        patch_size: Tuple[int, int] = (16, 16),
        mlp_ratio: float = 4.0,
    ):
        self.num_layers = num_layers
        self.d = hidden_dim
        self.num_heads = num_heads
        self.N = num_original_tokens
        self.num_classes = num_classes
        self.num_prefix = num_prefix_tokens
        self.patch_size = patch_size
        self.mlp_ratio = mlp_ratio

        # Calculate baseline FLOPs once
        self.baseline_flops = self._calc_total_flops(layer_idx=self.num_layers, k=self.N)["total_flops"]

    def _block_flops(self, seq_len: int) -> float:
        """
        Theoretical FLOPs (Multiply-Accumulate operations * 2) for a single standard ViT block.
        seq_len: T = num_prefix + k
        """
        T = seq_len
        d = self.d
        mlp_dim = int(d * self.mlp_ratio)

        # 1. Multi-Head Self Attention (MSA)
        qkv_flops = 6 * T * d * d
        attn_scores_flops = 2 * (T ** 2) * d
        softmax_flops = 3 * self.num_heads * (T ** 2)
        attn_out_flops = 2 * (T ** 2) * d
        proj_flops = 2 * T * d * d
        msa_flops = qkv_flops + attn_scores_flops + softmax_flops + attn_out_flops + proj_flops

        # 2. MLP Block
        fc1_flops = 2 * T * d * mlp_dim
        act_flops = 8 * T * mlp_dim
        fc2_flops = 2 * T * mlp_dim * d
        mlp_flops = fc1_flops + act_flops + fc2_flops

        # 3. LayerNorms & Residual adds: ~ 4 * T * d
        norm_flops = 4 * T * d

        return float(msa_flops + mlp_flops + norm_flops)

    def _calc_total_flops(self, layer_idx: int, k: int) -> Dict[str, float]:
        """
        Calculates end-to-end FLOPs when visual tokens are pruned to k after layer_idx.
        Layers 1..layer_idx run with full tokens (num_prefix + N).
        Layers (layer_idx+1)..L run with pruned tokens (num_prefix + k).
        """
        embed_flops = 2 * 3 * self.patch_size[0] * self.patch_size[1] * self.d * self.N
        
        full_seq = self.num_prefix + self.N
        pruned_seq = self.num_prefix + k

        uncompressed_layer_flops = layer_idx * self._block_flops(full_seq)
        remaining_layers = max(0, self.num_layers - layer_idx)
        compressed_layer_flops = remaining_layers * self._block_flops(pruned_seq)
        head_flops = 2 * self.d * self.num_classes

        total_flops = embed_flops + uncompressed_layer_flops + compressed_layer_flops + head_flops

        return {
            "embed_flops": float(embed_flops),
            "uncompressed_layers_flops": float(uncompressed_layer_flops),
            "compressed_layers_flops": float(compressed_layer_flops),
            "head_flops": float(head_flops),
            "total_flops": float(total_flops),
        }

    def compute_flops(self, layer_idx: int, k: int) -> TheoreticalFLOPsResult:
        """Calculates theoretical FLOP breakdown and normalized compute ratio."""
        flops_dict = self._calc_total_flops(layer_idx, k)
        normalized = flops_dict["total_flops"] / self.baseline_flops

        return TheoreticalFLOPsResult(
            layer_idx=layer_idx,
            retained_tokens=k,
            total_tokens=self.N,
            embed_flops=flops_dict["embed_flops"],
            uncompressed_layers_flops=flops_dict["uncompressed_layers_flops"],
            compressed_layers_flops=flops_dict["compressed_layers_flops"],
            head_flops=flops_dict["head_flops"],
            total_flops=flops_dict["total_flops"],
            baseline_flops=self.baseline_flops,
            normalized_flops=normalized,
        )

    def _execute_full_forward_pass(
        self,
        model_wrapper: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        scorer: Any,
        pruner: Any,
        k: int,
    ) -> torch.Tensor:
        """
        Executes the COMPLETE end-to-end forward pass:
        Input image x -> Embedding -> Layers 0..l-1 -> Token Scoring -> Top-k Pruning -> Layers l..L-1 -> Head.
        """
        # 1. Embed and run layers up to layer_idx
        prefix_tok, vis_tok = model_wrapper.extract_tokens_at_layer(x, layer_idx)

        # 2. Token Scoring (if k < N)
        if k < self.N and scorer is not None:
            scores = scorer.score_tokens(model_wrapper, x, layer_idx, prefix_tok, vis_tok, orig_logits=None)
            retained_vis, _ = pruner.prune(vis_tok, scores, k)
        else:
            retained_vis = vis_tok

        # 3. Downstream layers + norm + head
        logits = model_wrapper.forward_from_pruned_tokens(prefix_tok, retained_vis, layer_idx)
        return logits

    def measure_latency_and_memory(
        self,
        model_wrapper: nn.Module,
        dummy_x: torch.Tensor,
        layer_idx: int,
        k: int,
        scorer: Optional[Any] = None,
        pruner: Optional[Any] = None,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> EmpiricalLatencyResult:
        """
        Measures real wall-clock latency (ms) and peak memory (MB) covering the FULL forward pass.
        """
        device = model_wrapper.device
        dummy_x = dummy_x.to(device)

        # Warmup
        for _ in range(warmup_runs):
            _ = self._execute_full_forward_pass(model_wrapper, dummy_x, layer_idx, scorer, pruner, k)
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Benchmark runs
        latencies = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        for _ in range(num_runs):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                _ = self._execute_full_forward_pass(model_wrapper, dummy_x, layer_idx, scorer, pruner, k)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                _ = self._execute_full_forward_pass(model_wrapper, dummy_x, layer_idx, scorer, pruner, k)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)

        latencies_tensor = torch.tensor(latencies, dtype=torch.float32)
        mean_lat = float(latencies_tensor.mean())
        std_lat = float(latencies_tensor.std()) if len(latencies) > 1 else 0.0
        p50 = float(latencies_tensor.median())
        p95 = float(torch.quantile(latencies_tensor, 0.95))

        peak_mem = 0.0
        if device.type == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        return EmpiricalLatencyResult(
            mean_latency_ms=mean_lat,
            std_latency_ms=std_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            peak_memory_mb=peak_mem,
        )
