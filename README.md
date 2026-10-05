# Layer-Wise Information Dynamics of Visual Tokens in Vision Transformers

An exploratory research framework to empirically characterize token-level task sensitivity, information retention curves, and marginal computation efficiency across the depth of Vision Transformers (ViTs / DeiT).

---

## 1. Project Context & Objectives

The goal of this research is to investigate how task-relevant visual representations evolve across the depth of a Vision Transformer and quantify the exact trade-off between retained token counts and computational cost.

### Core Research Questions
1. *Does visual token task sensitivity $S_l(t)$ change systematically across transformer depth $l$?*
2. *How does task retention $P_l(k)$ degrade when retaining $k$ visual tokens at layer $l$?*
3. *What is the marginal task performance gained per unit of computation ($\frac{\Delta P_l}{\Delta C_l}$) across different transformer depths?*

> [!IMPORTANT]
> **Research Integrity Principle:** This is an exploratory empirical measurement framework, not a predetermined token-pruning claim. The system does not assume that information decreases monotonically with depth, that later layers are always more compressible, or that attention maps equal information.

---

## 2. Mathematical Formulations

### A. Token Sensitivity / Selection Metric ($S_l(t_i)$)
Visual patch tokens $t_i$ ($i \in \{1, \dots, N\}$) at layer $l$ are scored using decoupled strategies:
* **Prediction Sensitivity (KL Divergence):**
  $$S_l(t_i) = \mathcal{D}_{\text{KL}}\left( p_{\text{orig}} \parallel p_{\text{perturbed}, i} \right)$$
* **Gradient-Weighted Saliency:**
  $$S_l(t_i) = \left\| t_i \odot \nabla_{t_i} \mathcal{L}_{\text{orig}} \right\|_2$$
* **CLS-Attention Saliency:**
  $$S_l(t_i) = \frac{1}{H} \sum_{h=1}^H \mathbf{A}_{h, \text{cls}, i}^{(l)}$$
* **Random Calibration Baseline:**
  $$S_l(t_i) \sim \text{Uniform}(0, 1)$$

### B. Decoupled Evaluation Metrics ($P_l(k)$)
To avoid circular evaluation bias, token selection is evaluated against independent performance metrics:
1. **Prediction Agreement:**
   $$A_l(k) = \frac{1}{M} \sum_{m=1}^M \mathbf{1}\left[ \arg\max p^{(m)}_{\text{orig}} = \arg\max p^{(m)}_{\text{compressed}} \right]$$
2. **Top-1 / Top-5 Task Accuracy:** Evaluated against ground-truth labels.
3. **Top-Class Confidence Retention:**
   $$\text{CR}_l(k) = \frac{1}{M} \sum_{m=1}^M \frac{p^{(m)}_{\text{compressed}}(y_{\text{orig}})}{p^{(m)}_{\text{orig}}(y_{\text{orig}}) + \epsilon}$$
4. **Output KL / JS Divergence:** Measuring full distributional drift.

### C. Computation Modeling ($C_l(k)$)
* **Theoretical End-to-End FLOPs:**
  $$C_l(k) = \text{FLOPs}_{\text{embed}} + \sum_{j=1}^l \text{FLOPs}_{\text{block}}(1 + N) + \sum_{j=l+1}^L \text{FLOPs}_{\text{block}}(1 + k) + \text{FLOPs}_{\text{head}}$$
  $$\hat{C}_l(k) = \frac{C_l(k)}{C_{\text{baseline}}}$$
  where standard Transformer Block FLOPs account for quadratic Multi-Head Attention ($\mathcal{O}(T^2 d)$) and linear MLP projections ($\mathcal{O}(T d^2)$).
* **Empirical Latency & Memory:** Separately measured via CUDA synchronization events and high-precision timers. **FLOPs and latency are never conflated.**

### D. Marginal Efficiency
$$\frac{\Delta P_l}{\Delta C_l} = \frac{P_l(k_{j+1}) - P_l(k_j)}{\hat{C}_l(k_{j+1}) - \hat{C}_l(k_j)}$$

---

## 3. Project Structure

```
.
├── configs/
│   ├── smoke_test.yaml              # Fast test configuration (10 samples)
│   ├── deit_tiny_cifar100_debug.yaml# Debug mode on CIFAR-100
│   └── deit_tiny_imagenet.yaml      # Primary ImageNet research configuration
├── src/
│   ├── models/
│   │   └── vit_wrapper.py           # ViT interception & partial forward execution
│   ├── data/
│   │   └── dataset.py               # ImageNet val, CIFAR-100 (debug), synthetic loaders
│   ├── importance/
│   │   ├── base.py                  # Abstract TokenImportanceScorer interface
│   │   ├── random_scorer.py         # Baseline 2: Uniform random scoring
│   │   ├── prediction_kl.py         # Baseline 3: Prediction KL (Exact & Approx)
│   │   └── attention_scorer.py      # Baseline 4: CLS Attention scoring
│   ├── compression/
│   │   └── token_pruning.py         # Top-k visual token selector (spatial order preserved)
│   ├── metrics/
│   │   ├── prediction.py            # Decoupled agreement, accuracy, KL, confidence
│   │   ├── compute.py               # Theoretical FLOPs and latency benchmark engine
│   │   └── information.py           # Marginal efficiency (Delta P / Delta C)
│   ├── experiments/
│   │   ├── runner.py                # Experiment lifecycle orchestrator & sanity checks
│   │   └── layerwise_analysis.py    # Statistical aggregation (SEM, 95% CIs)
│   └── visualization/
│       └── plots.py                 # 6 publication-ready research plots
├── scripts/
│   ├── run_experiment.py            # CLI for running experiments
│   └── analyze_results.py           # CLI for re-analysis and plotting
├── tests/                           # Comprehensive unit and smoke test suite
└── requirements.txt
```

---

## 4. Quickstart

### Installation
Ensure you have Python 3.10+ and install dependencies:
```bash
pip install -r requirements.txt
```

### 1. Run Unit Tests & Smoke Test
```bash
pytest tests/ -v
```

### 2. Run Smoke Test via CLI
Runs a fast end-to-end evaluation (10 samples, layers 4 and 8, 3 retention ratios):
```bash
python scripts/run_experiment.py --config configs/smoke_test.yaml
```

### 3. Run Primary ImageNet Experiment
```bash
python scripts/run_experiment.py --config configs/deit_tiny_imagenet.yaml --num-samples 500
```

### 4. Analyze Results and Refresh Plots
```bash
python scripts/analyze_results.py --results-dir results/smoke_test
```

---

## 5. Output Deliverables

After running an experiment, the results folder (e.g. `results/smoke_test/`) contains:
* `config.json`: Full experiment parameters and metadata.
* `layer_metrics.csv`: Detailed layer-wise metrics ($\mu, \sigma, \text{SEM}, \text{CI}_{95}$ for agreement, accuracy, FLOPs, latency).
* `marginal_efficiency.csv`: Step-wise $\frac{\Delta P}{\Delta C}$ ratios across token budgets.
* `summary.txt`: Structured empirical observations.
* `plots/`:
  1. `information_vs_tokens.png`: $P_l(k)$ vs $k$ across transformer depth.
  2. `information_vs_compute.png`: $P_l(k)$ vs $\hat{C}_l(k)$.
  3. `compute_vs_tokens.png`: Theoretical FLOPs scaling.
  4. `marginal_information.png`: $\frac{\Delta P}{\Delta C}$ efficiency across layers.
  5. `accuracy_vs_compute.png`: Pareto frontier comparing Random vs Importance vs Attention.
  6. `flops_vs_latency.png`: Discrepancy analysis between theoretical FLOPs and hardware latency.
