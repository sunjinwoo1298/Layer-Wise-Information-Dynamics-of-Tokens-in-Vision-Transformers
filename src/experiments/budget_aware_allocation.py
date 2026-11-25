"""
Phase 3: budget-aware token allocation ("LS-Chain + verify"); evaluation only (frozen model, existing scorer, no fine-tuning).

Stages (each is a separate call; later stages refuse to run if the earlier freeze is missing or altered):
  prepare   calibration/validation split of the 2,000-image subset, fresh confirmatory set, manifests + checksums
  calibrate sensitivity tables (calibration images only), proxy solve, top-M verification on calibration, baselines on calibration,
            proxy-validity check, FREEZE (frozen_schedules.json + sha256)
  validate  frozen schedules + independent baselines on the 1,500 validation images (stats, latency, plots)
  confirm   the same frozen schedules + baselines on the fresh confirmatory set, run ONCE (refuses to repeat)
Geometric, equal-drop (constrained/unconstrained) and single-stage schedules are independent baselines produced by the unchanged Phase 2
generators; the proposed algorithm never receives them as candidates and never falls back to them.
"""

import hashlib
import json
import os
import pickle
import platform
import subprocess
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

from ..compression.progressive import ProgressivePruner
from ..metrics.compute import ComputeEstimator
from ..models.vit_wrapper import ViTWrapper
from ..optimization.schedule_search import (
    DEFAULT_GRID, SensitivityTables, ProxyModel, normalized_flops, select_schedule,
)
from ..data.phase3_splits import (
    assert_split_disjoint, build_confirmatory_subset, make_cal_val_split, split_checksum,
)
from .class_diverse import ClassDiverseConfig, _time_multi, load_subset
from .layerwise_analysis import wilson_score_interval
from .minimum_token import generate_min_token_schedules
from .progressive_eval import (
    LAYER_NUMBERING, _environment, _md_table, _sha256_file, expected_processed_counts, paired_diff_stats, schedule_flops,
)
from .stage_ablation import STAGE_NAME, holm_correction

HAND_MARK = "## Findings and conclusions (hand-written)"
PROPOSED = "proposed"


@dataclass
class AllocConfig:
    model_name: str = "deit_tiny_patch16_224"
    pretrained: bool = True
    parquet_dir: str = "./data/imagenet_val_parquet"
    subset_cache: str = "./data/imagenet_class_diverse_subset.pt"
    confirm_cache: str = "./data/imagenet_confirmatory_2000.pt"
    parent_dir: str = "./results/phase2_minimum_token_ablation"           # saved baseline schedules + parent manifest
    class_diverse_dir: str = "./results/phase2_class_diverse_validation"
    isolation_dir: str = "./results/phase2_final_token_trajectory_isolation"
    output_dir: str = "./results/phase3_budget_aware_allocation"
    seed: int = 42
    split_seed: int = 42
    n_cal: int = 500
    confirm_seed: int = 4242
    batch_size: int = 25
    scorer: str = "cls_attention"
    first_layer: int = 2
    patterns: List[str] = field(default_factory=lambda: ["2-6-10", "2-8-10"])
    budgets: List[float] = field(default_factory=lambda: [0.40, 0.50, 0.60, 0.70])
    floors: List[int] = field(default_factory=lambda: [16, 24, 32])
    match_tolerance: float = 0.01
    grid: List[int] = field(default_factory=lambda: list(DEFAULT_GRID))
    top_m: int = 8
    min_dist: int = 6
    tie_eps: float = 0.002
    bootstrap_stability: int = 10
    proxy_validity_min_spearman: float = 0.6
    latency_warmup: int = 1
    latency_runs: int = 5
    latency_batches: int = 3
    device: Optional[str] = None
    resume: bool = True
    make_plots: bool = True
    max_cal: Optional[int] = None            # smoke / tests only
    max_val: Optional[int] = None
    max_conf: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def hyper_hash(self) -> str:
        keys = ["scorer", "first_layer", "patterns", "budgets", "floors", "match_tolerance", "grid", "top_m", "min_dist", "tie_eps",
                "seed", "split_seed", "n_cal", "confirm_seed", "batch_size", "model_name"]
        d = {k: getattr(self, k) for k in keys}
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def structures_of(cfg: AllocConfig) -> Dict[str, List[Tuple[int, ...]]]:
    out = {"two": [], "three": []}
    for p in cfg.patterns:
        t = tuple(int(x) for x in p.split("-"))
        out["two" if len(t) == 2 else "three"].append(t)
    return out


@dataclass
class DataSet:
    tag: str
    images: torch.Tensor
    labels: torch.Tensor
    ids: np.ndarray
    manifest: pd.DataFrame

    def __len__(self):
        return len(self.images)

    def take(self, n: Optional[int]) -> "DataSet":
        if n is None or n >= len(self):
            return self
        return DataSet(self.tag, self.images[:n], self.labels[:n], self.ids[:n], self.manifest.iloc[:n].copy())


# ----------------------------------------------------------------------------- data

def out_path(cfg: AllocConfig, *parts: str) -> str:
    p = os.path.join(cfg.output_dir, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def prepare_data(cfg: AllocConfig, need_conf: bool = False, data: Optional[Dict[str, Any]] = None) -> Dict[str, DataSet]:
    """Calibration / validation split of the parent subset (+ the fresh confirmatory set if `need_conf`). `data` = test injection."""
    if data is not None:
        return data
    sub = load_subset(ClassDiverseConfig(parquet_dir=cfg.parquet_dir, subset_cache=cfg.subset_cache, images_per_class=2, subset_seed=42), None)
    man = make_cal_val_split(sub["manifest"], cfg.n_cal, cfg.split_seed)
    out = {}
    for tag in ("calibration", "validation"):
        m = man[man.split == tag].reset_index(drop=True)
        idx = man.index[man.split == tag].to_numpy()
        out[tag] = DataSet(tag, sub["images"][idx], sub["labels"][idx], m.image_id.to_numpy(), m)
    summ = {"parent_manifest_sha256": sub["summary"]["manifest_sha256"], "n_calibration": int((man.split == "calibration").sum()),
            "n_validation": int((man.split == "validation").sum()),
            "calibration_sha256": split_checksum(man, "calibration"), "validation_sha256": split_checksum(man, "validation"),
            "calibration_classes": int(man[man.split == "calibration"].label.nunique()),
            "validation_classes": int(man[man.split == "validation"].label.nunique()),
            "split_algorithm": "RandomState(split_seed).permutation(classes)[:n_cal] -> one random image (RandomState(split_seed+1)) per chosen class = calibration; rest = validation",
            "split_seed": cfg.split_seed, "ids_disjoint": True}
    man.to_csv(out_path(cfg, "splits", "split_manifest.csv"), index=False)
    json.dump(summ, open(out_path(cfg, "splits", "split_summary.json"), "w"), indent=2)
    if need_conf:
        c = build_confirmatory_subset(cfg.parquet_dir, sub["manifest"].source_row.to_numpy(), 2, cfg.confirm_seed, cfg.confirm_cache)
        overlap = set(c["manifest"].source_row) & set(sub["manifest"].source_row)
        if overlap:
            raise RuntimeError("confirmatory set overlaps the existing subset")
        c["manifest"].to_csv(out_path(cfg, "splits", "confirmatory_manifest.csv"), index=False)
        json.dump({k: v for k, v in c["summary"].items() if k != "class_histogram"}, open(out_path(cfg, "splits", "confirmatory_summary.json"), "w"), indent=2)
        out["confirmatory"] = DataSet("confirmatory", c["images"], c["labels"], c["manifest"].image_id.to_numpy(), c["manifest"])
    return out


# ----------------------------------------------------------------------------- model runner

class Runner:
    def __init__(self, cfg: AllocConfig, wrapper: Optional[ViTWrapper] = None):
        self.cfg = cfg
        self.wrapper = wrapper or ViTWrapper(cfg.model_name, cfg.pretrained, cfg.device)
        w = self.wrapper
        self.L, self.N = w.num_layers, w.num_original_tokens
        self.est = ComputeEstimator(self.L, w.hidden_dim, w.num_heads, self.N, w.num_classes, w.num_prefix_tokens, w.patch_size)
        self.dev = w.device
        self._base: Dict[str, Tuple[List[torch.Tensor], np.ndarray]] = {}

    def prepare(self, ds: DataSet) -> None:
        bs = self.cfg.batch_size
        batches = [ds.images[i:i + bs] for i in range(0, len(ds), bs)]
        with torch.no_grad():
            logits = [self.wrapper(x.to(self.dev)).cpu() for x in batches]
        self._base[ds.tag] = (logits, torch.cat(logits).argmax(-1).numpy())

    def run(self, schedule: Dict[int, int], ds: DataSet, with_kl: bool = False) -> Dict[str, np.ndarray]:
        if ds.tag not in self._base:
            self.prepare(ds)
        bl, bpred = self._base[ds.tag]
        bs = self.cfg.batch_size
        pruner = ProgressivePruner(self.wrapper, schedule, scorer=self.cfg.scorer, seed=self.cfg.seed)
        preds, top5, kls = [], [], []
        for i, b in enumerate(bl):
            x = ds.images[i * bs:(i + 1) * bs].to(self.dev)
            res = pruner.forward(x, orig_logits=b.to(self.dev))
            lg = res.logits.cpu()
            preds.append(lg.argmax(-1)); top5.append(lg.topk(5, dim=-1).indices)
            if with_kl:
                lp_f, lp_p = Fnn.log_softmax(b, -1), Fnn.log_softmax(lg, -1)
                kls.append((lp_f.exp() * (lp_f - lp_p)).sum(-1))
        preds, top5 = torch.cat(preds).numpy(), torch.cat(top5).numpy()
        y = ds.labels.numpy()
        out = {"pred": preds, "top5": top5, "agree": (preds == bpred).astype(int), "correct": (preds == y).astype(int),
               "top5_correct": (top5 == y[:, None]).any(1).astype(int)}
        if with_kl:
            out["kl"] = torch.cat(kls).numpy().astype(np.float64)
        return out

    def baseline_pred(self, ds: DataSet) -> np.ndarray:
        if ds.tag not in self._base:
            self.prepare(ds)
        return self._base[ds.tag][1]


def rss_mb() -> float:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, "peak_wset", mi.rss)) / 2 ** 20
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- baselines (independent of the proposed pool)

def baseline_entries(est: ComputeEstimator, cfg: AllocConfig) -> Dict[str, Any]:
    g = generate_min_token_schedules(est, cfg.first_layer, cfg.budgets, structures_of(cfg), cfg.match_tolerance, cfg.floors)
    for e in g["entries"]:
        e["role_phase3"] = {"shared": "single_stage", "geometric": "geometric", "equal_drop": "unconstrained_equal_drop",
                            "constrained_equal_drop": "constrained_equal_drop"}[e["family"]]
    return g


def verify_baselines(entries: List[Dict[str, Any]], infeasible: List[Dict[str, Any]], cfg: AllocConfig) -> Dict[str, Any]:
    """Regenerated baselines must equal the saved minimum-token manifest (restricted to this study's patterns and floors)."""
    mp = os.path.join(cfg.parent_dir, "schedule_manifest.json")
    if not os.path.isfile(mp):
        raise RuntimeError(f"Cannot verify baselines: {mp} not found. Stopping.")
    man = json.load(open(mp))
    keep = {s["name"]: {int(k): v for k, v in s["schedule"].items()} for s in man["schedules"]
            if s["role"] != "baseline" and (s["role"] == "anchor" or s["structure"] in cfg.patterns)
            and (s["floor"] is None or s["floor"] in cfg.floors) and any(abs(s["budget_target"] - b) < 1e-9 for b in cfg.budgets)}
    cur = {e["name"]: e["schedule"] for e in entries}
    if cur != keep:
        raise RuntimeError(f"Regenerated baseline schedules differ from the saved manifest: {sorted(set(cur) ^ set(keep))[:6]}")
    pi = {(round(u["budget"], 6), u["structure"], u["floor"]) for u in man["infeasible_combinations"]
          if u["structure"] in cfg.patterns and u["floor"] in cfg.floors and any(abs(u["budget"] - b) < 1e-9 for b in cfg.budgets)}
    ci = {(round(u["budget"], 6), u["structure"], u["floor"]) for u in infeasible}
    if pi != ci:
        raise RuntimeError("Infeasible constrained combinations differ from the saved manifest")
    return {"verified": True, "n_schedules": len(cur), "manifest": mp, "n_infeasible": len(ci)}


# ----------------------------------------------------------------------------- calibration: sensitivity tables

def measure_tables(runner: Runner, cal: DataSet, cfg: AllocConfig, layers_needed: List[int], pairs: List[Tuple[int, int]],
                   ck_path: Optional[str] = None) -> SensitivityTables:
    singles = sorted(set(layers_needed))
    tab = SensitivityTables(grid=list(cfg.grid), singles=singles, pairs=pairs, n_images=len(cal), N=runner.N)
    keys = tab.required_keys()
    sig = hashlib.sha256((json.dumps([cfg.hyper_hash(), len(cal), keys]) + str(cal.ids[:5])).encode()).hexdigest()
    done: Dict[str, np.ndarray] = {}
    if cfg.resume and ck_path and os.path.isfile(ck_path):
        ck = pickle.load(open(ck_path, "rb"))
        if ck["sig"] == sig:
            done = ck["done"]
            print(f"[resume] {len(done)}/{len(keys)} table runs", flush=True)
    t0 = time.time()
    for n, key in enumerate(keys):
        if key in done:
            continue
        p = key.split("|")
        sched = {int(p[1]): int(p[2])} if p[0] == "s" else {int(p[1]): int(p[2]), int(p[3]): int(p[4])}
        done[key] = runner.run(sched, cal, with_kl=True)["kl"]
        if (n + 1) % 20 == 0 or n + 1 == len(keys):
            print(f"[tables {n + 1}/{len(keys)}] {key} ({time.time() - t0:.0f}s)", flush=True)
            if ck_path:
                pickle.dump({"sig": sig, "done": done}, open(ck_path, "wb"))
    tab.per_image = done
    return tab


# ----------------------------------------------------------------------------- calibration stage

def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def run_calibration(cfg: AllocConfig, runner: Optional[Runner] = None, data: Optional[Dict[str, DataSet]] = None) -> Dict[str, Any]:
    t_all, rss0 = time.time(), rss_mb()
    torch.manual_seed(cfg.seed)
    runner = runner or Runner(cfg)
    ds = prepare_data(cfg, need_conf=False, data=data)
    cal = ds["calibration"].take(cfg.max_cal)
    est = runner.est
    g = baseline_entries(est, cfg)
    entries, infeasible = g["entries"], g["infeasible"]
    verification = verify_baselines(entries, infeasible, cfg) if (data is None and cfg.max_cal is None
                                                                    and cfg.budgets == [0.4, 0.5, 0.6, 0.7]) \
        else {"verified": False, "note": "not checked (smoke / custom setup)"}
    structs = [tuple(int(x) for x in p.split("-")) for p in cfg.patterns]
    singles = sorted({l for s in structs for l in s})                      # all stage layers incl. last (needed for ka=N boundary)
    pairs = sorted({(s[j], s[j + 1]) for s in structs for j in range(len(s) - 1)})
    runner.prepare(cal)
    t_tab = time.time()
    tables = measure_tables(runner, cal, cfg, singles, pairs, out_path(cfg, "calibration", "_tables_checkpoint.pkl"))
    t_tab = time.time() - t_tab
    np.savez_compressed(out_path(cfg, "calibration", "sensitivity_per_image_kl.npz"), **{k.replace("|", "__"): v for k, v in tables.per_image.items()})
    proxy = tables.build()

    # ---- selection (calibration only)
    geo = {(e["budget_target"], e["structure"]): e for e in entries if e["family"] == "geometric"}
    cache: Dict[Tuple[int, ...], Dict[str, np.ndarray]] = {}
    t_sel = time.time()
    selections, cand_rows, proposed = [], [], []
    for p in cfg.patterns:
        layers = tuple(int(x) for x in p.split("-"))
        for B in cfg.budgets:
            gref = geo.get((B, p))
            if gref is None:
                for F in cfg.floors:
                    selections.append({"pattern": p, "budget": B, "floor": F, "status": "no_geometric_reference", "reason": "geometric reference unmatched"})
                continue
            cap = gref["normalized_flops"]
            for F in cfg.floors:
                def verify(counts, layers=layers):
                    key = (layers, tuple(counts))
                    if key not in cache:
                        cache[key] = runner.run(dict(zip(layers, counts)), cal)
                    return float(cache[key]["agree"].mean())
                r = select_schedule(proxy, est, layers, F, cap, cfg.match_tolerance, verify, cfg.top_m, cfg.min_dist, cfg.tie_eps)
                stab = _stability(tables, est, layers, F, cap, cfg, r) if r.status == "selected" else {}
                sel = {"pattern": p, "budget": B, "floor": F, "cap_flops": cap, "status": r.status, "reason": r.reason, "n_feasible": r.n_feasible,
                       "counts": r.counts, "geometric_vector": list(gref["schedule"].values()), **stab}
                selections.append(sel)
                for c in r.candidates:
                    cand_rows.append({"pattern": p, "budget": B, "floor": F, "rank": c["rank"], "vector": json.dumps(c["counts"]),
                                      "proxy_loss": c["proxy_loss"], "flops_pct": 100 * c["flops"], "cal_agreement": c["cal_agreement"],
                                      "selected": c["selected"]})
                if r.status == "selected":
                    sched = dict(zip(layers, r.counts))
                    name = f"L{cfg.first_layer}_B{int(round(100 * B))}_{PROPOSED}_f{F}_{p}"
                    proposed.append({"name": name, "schedule": {int(k): int(v) for k, v in sched.items()}, "layers": list(layers), "structure": p,
                                     "budget_target": B, "group": f"L{cfg.first_layer}_B{int(round(100 * B))}", "floor": F, "family": PROPOSED,
                                     "role": "member", "alias_of": None, "stage_count": len(layers), "cap_flops": cap,
                                     "final_patch_tokens": int(r.counts[-1]), **schedule_flops(est, sched)})
    t_sel = time.time() - t_sel
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(out_path(cfg, "calibration", "candidates.csv"), index=False)
    pd.DataFrame(selections).drop(columns=[c for c in ("top_m_overlap",) if False]).to_csv(out_path(cfg, "calibration", "selections.csv"), index=False)

    # ---- baselines on calibration (for calibration->validation gap and proxy validity)
    t_base = time.time()
    base_cal = []
    for e in entries:
        if e["alias_of"]:
            continue
        base_cal.append((e, runner.run(e["schedule"], cal)))
    prop_cal = [(e, cache[(tuple(e["layers"]), tuple(e["schedule"][l] for l in e["layers"]))]) for e in proposed]
    cal_rows = []
    for e, r in base_cal + prop_cal:
        cal_rows.append({"schedule": e["name"], "family": e["family"], "floor": e["floor"], "structure": e["structure"],
                         "budget_target": e["budget_target"], "flops_pct": 100 * e["normalized_flops"],
                         "vector": json.dumps(list(e["schedule"].values())), "cal_agreement": float(r["agree"].mean()),
                         "cal_top1": float(r["correct"].mean()), "n": len(cal)})
    for e in entries:
        if e["alias_of"]:
            src = next(x for x in cal_rows if x["schedule"] == e["alias_of"])
            cal_rows.append({**src, "schedule": e["name"], "family": e["family"], "floor": e["floor"]})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(out_path(cfg, "calibration", "calibration_results.csv"), index=False)
    t_base = time.time() - t_base

    validity = proxy_validity(cfg, proxy, cal, runner, entries, base_cal + prop_cal, cand_df)
    json.dump(validity, open(out_path(cfg, "calibration", "proxy_validity.json"), "w"), indent=2, default=str)

    # ---- freeze
    overhead = {"table_runs": len(tables.required_keys()), "table_seconds": t_tab, "selection_seconds_incl_verification": t_sel,
                "verification_runs": int(len(cache)), "baselines_on_calibration_seconds": t_base, "total_calibration_stage_seconds": time.time() - t_all,
                "peak_rss_mb_before": rss0, "peak_rss_mb_after": rss_mb(), "calibration_images": len(cal), "device": str(runner.dev)}
    json.dump(overhead, open(out_path(cfg, "calibration", "calibration_overhead.json"), "w"), indent=2)
    frozen = {
        "hyperparameters": cfg.to_dict(), "hyperparameter_hash": cfg.hyper_hash(), "code_commit": _git_commit(),
        "layer_numbering": LAYER_NUMBERING, "calibration_ids_sha256": hashlib.sha256(",".join(cal.ids).encode()).hexdigest(),
        "tables_digest": tables.digest(), "baseline_verification": verification,
        "proposed_schedules": proposed, "selections": selections, "infeasible_constrained_baselines": infeasible,
        "baseline_schedules": [{"name": e["name"], "family": e["family"], "floor": e["floor"], "structure": e["structure"],
                                "budget_target": e["budget_target"], "schedule": {str(k): v for k, v in e["schedule"].items()},
                                "alias_of": e["alias_of"], "role": e["role"], "group": e["group"], "layers": e["layers"],
                                "stage_count": e["stage_count"], "final_patch_tokens": e["final_patch_tokens"],
                                "normalized_flops": e["normalized_flops"], "analytic_flops": e["analytic_flops"]} for e in entries]}
    frozen_json = json.dumps(frozen, indent=1, sort_keys=True, default=str)
    fp = out_path(cfg, "frozen_schedules.json")
    open(fp, "w").write(frozen_json)
    open(out_path(cfg, "frozen.sha256"), "w").write(hashlib.sha256(frozen_json.encode()).hexdigest())
    ck = out_path(cfg, "calibration", "_tables_checkpoint.pkl")
    if os.path.isfile(ck):
        os.remove(ck)
    return {"frozen": frozen, "selections": selections, "candidates": cand_df, "calibration_results": cal_df, "validity": validity,
            "overhead": overhead, "tables": tables, "proxy": proxy}


def _stability(tables: SensitivityTables, est, layers, F, cap, cfg: AllocConfig, r) -> Dict[str, Any]:
    """Bootstrap over calibration images: how often is the selected schedule within the top-M of the resampled proxy ranking?"""
    from ..optimization.schedule_search import feasible_schedules, rank_diverse
    if cfg.bootstrap_stability <= 0:
        return {}
    feas = feasible_schedules(est, layers, F, cap, cfg.match_tolerance)
    rs = np.random.RandomState(cfg.seed)
    hits, ranks = 0, []
    sel = np.array(r.counts)
    for b in range(cfg.bootstrap_stability):
        idx = rs.randint(0, tables.n_images, tables.n_images)
        loss = tables.build(idx).loss(layers, feas)
        pick = rank_diverse(feas, loss, cfg.top_m, cfg.min_dist)
        top = feas[pick]
        hits += int(any(np.abs(top - sel).max(1) <= cfg.min_dist - 1))     # within the diversity radius of a resampled top-M member
        pos = int((loss < loss[(feas == sel).all(1)][0]).sum()) if (feas == sel).all(1).any() else -1
        ranks.append(pos)
    return {"bootstrap_top_m_hit_rate": hits / cfg.bootstrap_stability, "bootstrap_median_proxy_rank": float(np.median(ranks))}


def proxy_validity(cfg, proxy: ProxyModel, cal: DataSet, runner: Runner, entries, runs, cand_df) -> Dict[str, Any]:
    """H1: Spearman between proxy loss and MEASURED calibration disagreement over verified candidates, baselines and (saved, calibration-id
    restricted) isolation controls. No validation id is touched."""
    from scipy.stats import spearmanr
    pts = []
    for e, r in runs:
        layers = e["layers"] if e.get("layers") else None
        if not layers or len(layers) != 3 or e.get("family") == "shared":
            continue
        pts.append({"name": e["name"], "family": e["family"], "pattern": e["structure"], "budget": e["budget_target"],
                    "proxy": float(proxy.loss(layers, np.array([[e["schedule"][l] for l in layers]]))[0]), "disagree": 1 - float(r["agree"].mean())})
    for _, c in cand_df.iterrows():
        layers = [int(x) for x in c.pattern.split("-")]
        pts.append({"name": f"cand_{c.pattern}_{c.budget}_{c.floor}_{c['rank']}", "family": "verified_candidate", "pattern": c.pattern, "budget": c.budget,
                    "proxy": float(c.proxy_loss), "disagree": 1 - float(c.cal_agreement)})
    ctrl_n = 0
    pp = os.path.join(cfg.isolation_dir, "per_image_predictions.csv.gz")
    tp = os.path.join(cfg.isolation_dir, "schedule_table.csv")
    if os.path.isfile(pp) and os.path.isfile(tp):
        tab = pd.read_csv(tp)
        tab = tab[(tab.family == "final_count_matched_geometric") & (tab.status == "executed")]
        pi = pd.read_csv(pp, usecols=["schedule", "image_id", "agreement"])
        cal_ids = set(cal.ids)
        pi = pi[pi.image_id.isin(cal_ids)]
        for t in tab.itertuples():
            if t.structure not in cfg.patterns or not any(abs(t.budget_target - b) < 1e-9 for b in cfg.budgets):
                continue
            g = pi[pi.schedule == t.name]
            if len(g) < len(cal) * 0.99:
                continue
            layers = json.loads(t.layers)
            vec = json.loads(t.vector)
            pts.append({"name": t.name, "family": "isolation_control(saved, cal ids only)", "pattern": t.structure, "budget": t.budget_target,
                        "proxy": float(proxy.loss(layers, np.array([vec]))[0]), "disagree": 1 - float(g.agreement.mean())})
            ctrl_n += 1
    df = pd.DataFrame(pts)
    if len(df) < 5:
        return {"n_points": len(df), "note": "too few points"}
    pooled = spearmanr(df.proxy, df.disagree)
    cells = {}
    for (p, b), g in df.groupby(["pattern", "budget"]):
        if len(g) >= 5:
            cells[f"{p}|B{int(round(100 * b))}"] = {"n": len(g), "spearman": float(spearmanr(g.proxy, g.disagree)[0])}
    # also within-cell rank agreement among budgets pooled
    mean_cell = float(np.mean([v["spearman"] for v in cells.values()])) if cells else float("nan")
    df.to_csv(out_path(cfg, "calibration", "proxy_validity_points.csv"), index=False)
    return {"n_points": len(df), "n_isolation_controls_used": ctrl_n, "pooled_spearman": float(pooled[0]), "pooled_p": float(pooled[1]),
            "mean_within_cell_spearman": mean_cell, "per_cell": cells, "threshold": cfg.proxy_validity_min_spearman,
            "h1_met": bool(pooled[0] >= cfg.proxy_validity_min_spearman)}


# ----------------------------------------------------------------------------- freeze checks

def load_frozen(cfg: AllocConfig) -> Dict[str, Any]:
    fp, hp = out_path(cfg, "frozen_schedules.json"), out_path(cfg, "frozen.sha256")
    if not (os.path.isfile(fp) and os.path.isfile(hp)):
        raise RuntimeError("No frozen schedules found: run the calibration stage first. Refusing to evaluate.")
    txt = open(fp).read()
    if hashlib.sha256(txt.encode()).hexdigest() != open(hp).read().strip():
        raise RuntimeError("frozen_schedules.json does not match frozen.sha256 (altered after freezing). Refusing to evaluate.")
    fz = json.loads(txt)
    if fz["hyperparameter_hash"] != cfg.hyper_hash():
        raise RuntimeError("Hyper-parameters changed after freezing. Refusing to evaluate.")
    return fz


def frozen_entries(fz: Dict[str, Any]) -> List[Dict[str, Any]]:
    ents = []
    for b in fz["baseline_schedules"]:
        ents.append({**b, "schedule": {int(k): v for k, v in b["schedule"].items()}, "role_phase3": b["family"]})
    for p in fz["proposed_schedules"]:
        ents.append({**p, "schedule": {int(k): v for k, v in p["schedule"].items()}, "role_phase3": PROPOSED})
    return ents


# ----------------------------------------------------------------------------- evaluation (validation / confirmatory)

def _metric_row(e, split, n_img, agree, correct, t5, cfg):
    row = {"schedule": e["name"], "split": split, "family": e["family"], "floor": e.get("floor"), "structure": e["structure"],
           "budget_target": e["budget_target"], "group": e["group"], "stage_count": e["stage_count"], "scorer": cfg.scorer, "seed": cfg.seed,
           "vector": json.dumps(list(e["schedule"].values())), "final_patch_tokens": e["final_patch_tokens"],
           "normalized_flops": e["normalized_flops"], "flops_pct": 100 * e["normalized_flops"], "analytic_flops": e["analytic_flops"], "n_images": n_img,
           "alias_of": e.get("alias_of")}
    for name, arr in (("agreement", agree), ("top1", correct), ("top5", t5)):
        w = wilson_score_interval(int(arr.sum()), n_img)
        row.update({f"{name}_count": int(arr.sum()), name: w["mean"], f"{name}_ci_low": w["ci_lower"], f"{name}_ci_high": w["ci_upper"]})
    return row


def run_evaluation(cfg: AllocConfig, split: str, runner: Optional[Runner] = None, data: Optional[Dict[str, DataSet]] = None,
                   latency: bool = True) -> Dict[str, Any]:
    if split not in ("validation", "confirmatory"):
        raise ValueError(split)
    fz = load_frozen(cfg)
    sub = split
    done_marker = out_path(cfg, sub, "done.json")
    if split == "validation" and os.path.isfile(done_marker) and not cfg.resume:
        raise RuntimeError("validation already evaluated")
    if split == "confirmatory":
        if not os.path.isfile(out_path(cfg, "validation", "done.json")):
            raise RuntimeError("Run the validation stage before the confirmatory evaluation.")
        if json.load(open(out_path(cfg, "validation", "done.json")))["frozen_sha256"] != open(out_path(cfg, "frozen.sha256")).read().strip():
            raise RuntimeError("Validation was run on a different freeze. Refusing.")
        if os.path.isfile(done_marker):
            raise RuntimeError("The confirmatory evaluation has already been run once and must not be repeated.")
    torch.manual_seed(cfg.seed)
    runner = runner or Runner(cfg)
    ds = prepare_data(cfg, need_conf=(split == "confirmatory"), data=data)[split]
    ds = ds.take(cfg.max_val if split == "validation" else cfg.max_conf)
    est = runner.est
    entries = frozen_entries(fz)
    runner.prepare(ds)
    n = len(ds)
    exec_entries = [e for e in entries if not e.get("alias_of")]
    ck = out_path(cfg, sub, "_checkpoint.pkl")
    rows, pis, done = [], [], set()
    if cfg.resume and os.path.isfile(ck):
        c = pickle.load(open(ck, "rb"))
        if c["sig"] == open(out_path(cfg, "frozen.sha256")).read().strip():
            rows, pis, done = c["rows"], c["pis"], c["done"]
    t0 = time.time()
    base_entry = {"name": "no_pruning", "schedule": {}, "family": "none", "floor": None, "structure": "", "budget_target": 1.0, "group": "baseline",
                  "stage_count": 0, "final_patch_tokens": runner.N, "alias_of": None, **schedule_flops(est, {})}
    for i, e in enumerate([base_entry] + exec_entries):
        if e["name"] in done:
            continue
        r = runner.run(e["schedule"], ds)
        if e["family"] == "none":
            assert (r["agree"] == 1).all(), "no-pruning path must reproduce the unpruned predictions exactly"
        rows.append(_metric_row(e, split, n, r["agree"].astype(bool), r["correct"].astype(bool), r["top5_correct"].astype(bool), cfg))
        pis.append(pd.DataFrame({"schedule": e["name"], "image_idx": np.arange(n), "image_id": ds.ids, "label": ds.labels.numpy(),
                                 "baseline_pred": runner.baseline_pred(ds), "compressed_pred": r["pred"],
                                 "top5_preds": [" ".join(map(str, t)) for t in r["top5"]], "correct": r["correct"], "agreement": r["agree"],
                                 "top5_correct": r["top5_correct"]}))
        done.add(e["name"])
        if len(done) % 10 == 0:
            print(f"[{split} {len(done)}/{len(exec_entries) + 1}] {e['name']} ({time.time() - t0:.0f}s)", flush=True)
            pickle.dump({"sig": open(out_path(cfg, "frozen.sha256")).read().strip(), "rows": rows, "pis": pis, "done": done}, open(ck, "wb"))
    metrics = pd.DataFrame(rows)
    per_img = pd.concat(pis, ignore_index=True)
    lat_df = pd.DataFrame()
    if latency:
        nb = (n + cfg.batch_size - 1) // cfg.batch_size
        pick = sorted({int(round(i * (nb - 1) / max(cfg.latency_batches - 1, 1))) for i in range(cfg.latency_batches)})
        xs = [ds.images[b * cfg.batch_size:(b + 1) * cfg.batch_size].to(runner.dev) for b in pick]
        import statistics
        with torch.no_grad():
            runner.wrapper(xs[0])
            ts = []
            for x in xs:
                for _ in range(cfg.latency_runs):
                    t = time.perf_counter(); runner.wrapper(x); ts.append((time.perf_counter() - t) * 1e3)
        unp = statistics.median(ts)
        lr = []
        for e in [base_entry] + exec_entries:
            p = ProgressivePruner(runner.wrapper, e["schedule"], cfg.scorer, seed=cfg.seed)
            tm = _time_multi(p, xs, cfg.latency_warmup, cfg.latency_runs)
            lr.append({"schedule": e["name"], "family": e["family"], "floor": e.get("floor"), "structure": e["structure"],
                       "budget_target": e["budget_target"], "flops_pct": 100 * e["normalized_flops"], "unpruned_forward_ms_median": unp,
                       "end_to_end_ms_median": tm["total_ms_median"], "end_to_end_ms_mean": tm["total_ms_mean"],
                       "scorer_overhead_ms_median": 0.0 if e["family"] == "none" else tm["scorer_ms_median"],
                       "device": str(runner.dev), "batch_size": len(xs[0]), "warmup": cfg.latency_warmup, "timed_runs": cfg.latency_runs,
                       "num_batches": len(xs)})
        lat_df = pd.DataFrame(lr)
        lat_df.to_csv(out_path(cfg, sub, "latency.csv"), index=False)
    # alias rows
    extra = []
    for e in entries:
        if e.get("alias_of"):
            src = metrics[metrics.schedule == e["alias_of"]]
            if len(src):
                d = src.iloc[0].to_dict()
                d.update({"schedule": e["name"], "family": e["family"], "floor": e["floor"], "alias_of": e["alias_of"],
                          "vector": json.dumps(list(e["schedule"].values()))})
                extra.append(d)
    if extra:
        metrics = pd.concat([metrics, pd.DataFrame(extra)], ignore_index=True)
    # calibration -> validation gap
    cal_res = pd.read_csv(out_path(cfg, "calibration", "calibration_results.csv")).set_index("schedule")
    metrics["cal_agreement"] = [cal_res.cal_agreement.get(s, np.nan) for s in metrics.schedule]
    metrics["cal_top1"] = [cal_res.cal_top1.get(s, np.nan) for s in metrics.schedule]
    metrics.to_csv(out_path(cfg, sub, "metrics.csv"), index=False)
    metrics.to_json(out_path(cfg, sub, "metrics.json"), orient="records", indent=1)
    per_img.to_csv(out_path(cfg, sub, "per_image_predictions.csv.gz"), index=False, compression="gzip")
    stats = build_phase3_tests(per_img, entries, cfg.match_tolerance)
    stats.to_csv(out_path(cfg, sub, "statistical_tests.csv"), index=False)
    json.dump(stats.attrs.get("skipped", []), open(out_path(cfg, sub, "skipped_pairs.json"), "w"), indent=1)
    cross = reproduction_check(cfg, metrics, split)
    out = {"metrics": metrics, "per_image": per_img, "stats": stats, "latency": lat_df, "entries": [base_entry] + entries, "frozen": fz,
           "cross_check": cross, "split": split, "n": n, "runner": runner}
    run_info = {"split": split, "n_images": n, "ids_sha256": hashlib.sha256(",".join(ds.ids).encode()).hexdigest(),
                "environment": _environment(runner.wrapper), "runtime_seconds": time.time() - t0, "peak_rss_mb": rss_mb(),
                "frozen_sha256": open(out_path(cfg, "frozen.sha256")).read().strip(), "cross_check": cross,
                "timing_method": ("time.perf_counter around ProgressivePruner.forward; median over timed runs on evenly spaced batches after warm-up; "
                                  "scorer overhead = time inside scorer calls; CPU only, no GPU claim"), "unpruned_forward_ms": (None if lat_df.empty else float(lat_df.unpruned_forward_ms_median.iloc[0]))}
    json.dump(run_info, open(out_path(cfg, sub, "run_manifest.json"), "w"), indent=2, default=str)
    if cfg.make_plots:
        from ..visualization.phase3_plots import make_all_plots
        make_all_plots(out, cfg, out_path(cfg, sub, "plots", "x.png").rsplit(os.sep, 1)[0])
    write_result(out, cfg)
    json.dump({"frozen_sha256": run_info["frozen_sha256"], "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, open(done_marker, "w"))
    if os.path.isfile(ck):
        os.remove(ck)
    return out


def reproduction_check(cfg, metrics, split) -> Dict[str, Any]:
    """Baseline schedules on the SAME images must reproduce the saved Phase 2 agreement counts (deterministic scorer)."""
    if split != "validation" or cfg.max_val is not None:
        return {"checked": False}
    try:
        pi = pd.read_csv(os.path.join(cfg.parent_dir, "per_image_metrics.csv"), usecols=["schedule", "scorer", "seed", "image_id", "agreement", "correct"])
    except Exception as ex:
        return {"checked": False, "error": str(ex)}
    pi = pi[(pi.scorer == cfg.scorer) & (pi.seed == cfg.seed)]
    val_ids = set(pd.read_csv(out_path(cfg, "splits", "split_manifest.csv")).query("split=='validation'").image_id)
    pi = pi[pi.image_id.isin(val_ids)]
    saved = pi.groupby("schedule").agg(a=("agreement", "sum"), c=("correct", "sum"), n=("agreement", "size"))
    m = metrics[(~metrics.alias_of.notna()) & metrics.schedule.isin(saved.index)]
    bad = [r.schedule for r in m.itertuples() if (saved.loc[r.schedule, "a"] != r.agreement_count or saved.loc[r.schedule, "c"] != r.top1_count)]
    return {"checked": True, "n_compared": int(len(m)), "n_identical": int(len(m) - len(bad)), "mismatches": bad}


# ----------------------------------------------------------------------------- statistics

def _fam(e) -> str:
    return PROPOSED if e["family"] == PROPOSED else (e["family"] if e["family"] != "constrained_equal_drop" else f"constrained_f{e['floor']}")


def build_phase3_tests(per_img: pd.DataFrame, entries: List[Dict[str, Any]], tol: float) -> pd.DataFrame:
    """diff = A - B, A = proposed. Predefined families (Holm within each, per metric):
         PRIMARY    proposed(F) vs geometric              (cells x floors)
         SECONDARY  proposed(F) vs constrained equal-drop at the same floor ;  proposed(F) vs single-stage anchor
         EXPLORATORY proposed(F) vs unconstrained equal-drop (diagnostic)
       pairs with identical vectors are listed in `skipped` and excluded; pairs with |dFLOPs| > tol are excluded."""
    idx = {k: g.sort_values("image_idx") for k, g in per_img.groupby("schedule")}
    rows, skipped = [], []

    def pred(e):
        return idx.get(e.get("alias_of") or e["name"])

    def add(a, b, kind):
        ga, gb = pred(a), pred(b)
        if ga is None or gb is None:
            return
        va, vb = list(a["schedule"].values()), list(b["schedule"].values())
        if va == vb and list(a["schedule"].keys()) == list(b["schedule"].keys()):
            skipped.append({"a": a["name"], "b": b["name"], "reason": "identical vectors", "kind": kind}); return
        d = a["normalized_flops"] - b["normalized_flops"]
        if abs(d) > tol + 1e-12:
            skipped.append({"a": a["name"], "b": b["name"], "reason": "pairwise |dFLOPs| > tolerance", "kind": kind}); return
        for metric, col in (("agreement", "agreement"), ("top1", "correct")):
            s = paired_diff_stats(ga[col].to_numpy(), gb[col].to_numpy())
            rows.append({"comparison": kind, "metric": metric, "budget_target": a["budget_target"], "structure": a["structure"],
                         "floor": a.get("floor"), "schedule_a": a["name"], "schedule_b": b["name"], "family_a": _fam(a), "family_b": _fam(b),
                         "vector_a": json.dumps(va), "vector_b": json.dumps(vb), "final_a": va[-1], "final_b": vb[-1],
                         "flops_pct_a": 100 * a["normalized_flops"], "flops_pct_b": 100 * b["normalized_flops"], "flops_diff_pp": 100 * d,
                         "n_images": s["n"], "count_a": s["a_count"], "count_b": s["b_count"], "diff": s["diff"], "ci_low": s["boot_ci_low"],
                         "ci_high": s["boot_ci_high"], "a_only": s["a_only"], "b_only": s["b_only"], "p_raw": s["mcnemar_p"]})

    byk: Dict[Tuple, List[Dict[str, Any]]] = {}
    anchors = {e["group"]: e for e in entries if e.get("role") == "anchor"}
    for e in entries:
        if e.get("role") == "member":
            byk.setdefault((e["group"], e["structure"]), []).append(e)
    for (grp, st), ents in byk.items():
        geo = next((e for e in ents if e["family"] == "geometric"), None)
        unc = next((e for e in ents if e["family"] == "equal_drop"), None)
        for p in [e for e in ents if e["family"] == PROPOSED]:
            if geo is not None:
                add(p, geo, "prop_vs_geo")
            c = next((e for e in ents if e["family"] == "constrained_equal_drop" and e["floor"] == p["floor"]), None)
            if c is not None:
                add(p, c, "prop_vs_cons")
            if grp in anchors:
                add(p, anchors[grp], "prop_vs_single")
            if unc is not None:
                add(p, unc, "prop_vs_unc")
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped
    if df.empty:
        return df
    lab = {"prop_vs_geo": "PRIMARY", "prop_vs_cons": "SECONDARY", "prop_vs_single": "SECONDARY", "prop_vs_unc": "EXPLORATORY"}
    df["family"] = [f"{lab[c]}_{c}|{m}" for c, m in zip(df.comparison, df.metric)]
    df["family_size"] = df.groupby("family").family.transform("size")
    df["p_holm"] = np.nan
    for _, g in df.groupby("family"):
        df.loc[g.index, "p_holm"] = holm_correction(g.p_raw.tolist())
    df["significant_holm_0.05"] = df.p_holm < 0.05
    df["significant_raw_0.05_uncorrected"] = df.p_raw < 0.05
    return df


# ----------------------------------------------------------------------------- result.md

def write_result(out: Dict[str, Any], cfg: AllocConfig) -> None:
    split, m, st, fz = out["split"], out["metrics"], out["stats"], out["frozen"]
    n = out["n"]
    L = [f"# Phase 3 — Budget-aware allocation: {split} results (generated)\n",
         "*Generated by `src/experiments/budget_aware_allocation.py`. A hand-written conclusion section, if present in `result.md`, is never overwritten "
         "(generated text then goes to `result_generated.md`). Agreement, top-1, analytic FLOPs, calibration overhead and latency are reported separately.*\n",
         "## Setup\n",
         f"- Frozen schedules: `frozen_schedules.json` (SHA-256 `{open(out_path(cfg, 'frozen.sha256')).read().strip()}`), hyper-parameter hash `{fz['hyperparameter_hash'][:16]}…`, code commit `{fz['code_commit'][:10]}`.",
         f"- Split evaluated: **{split}**, n = {n}. Scorer `{cfg.scorer}`, seed {cfg.seed}, first layer {cfg.first_layer}, patterns {cfg.patterns}, budgets {cfg.budgets}, floors {cfg.floors}, FLOPs tolerance ±{100 * cfg.match_tolerance:.0f} pp.",
         "- Proposed method: measured layer-sensitivity chain proxy + exact FLOPs-constrained enumeration, top-8 verified on the calibration set only (see `results/phase3_algorithm_design/algorithm_design.md`). Geometric / equal-drop / single-stage are independent baselines, never candidates or fallbacks.",
         f"- Layer numbering: {LAYER_NUMBERING}\n"]
    sp = out_path(cfg, "splits", "split_summary.json")
    if os.path.isfile(sp):
        s = json.load(open(sp))
        L.append(f"- Split hashes: calibration `{s['calibration_sha256']}` ({s['n_calibration']} images, {s['calibration_classes']} classes); validation `{s['validation_sha256']}` ({s['n_validation']} images, {s['validation_classes']} classes); parent manifest `{s['parent_manifest_sha256']}`.")
    cp = out_path(cfg, "splits", "confirmatory_summary.json")
    if split == "confirmatory" and os.path.isfile(cp):
        c = json.load(open(cp))
        L.append(f"- Confirmatory set: {c['num_images']} images, {c['num_distinct_classes']} classes, manifest SHA-256 `{c['manifest_sha256']}`; disjoint from the 2,000-image subset (asserted).")
    cc = out["cross_check"]
    L.append("- Baseline reproduction vs saved Phase 2 predictions on the same images: " + (f"{cc['n_identical']}/{cc['n_compared']} identical agreement and top-1 counts" + (f"; mismatches {cc['mismatches']}" if cc['mismatches'] else "") if cc.get("checked") else "not applicable for this split"))
    L.append("")

    L.append("## Selected schedules (calibration-only selection) and results\n")
    mm = m[m.family == PROPOSED].sort_values(["budget_target", "structure", "floor"])
    geo = m[m.family == "geometric"].set_index(["budget_target", "structure"])
    rows = []
    for r in mm.itertuples():
        g = geo.loc[(r.budget_target, r.structure)] if (r.budget_target, r.structure) in geo.index else None
        rows.append([f"B{int(100 * r.budget_target)}", r.structure, int(r.floor), r.vector, f"{r.flops_pct:.2f}", f"{100 * r.agreement:.1f} [{100 * r.agreement_ci_low:.1f}, {100 * r.agreement_ci_high:.1f}]",
                     f"{100 * r.top1:.1f}", f"{100 * r.cal_agreement:.1f}" if r.cal_agreement == r.cal_agreement else "",
                     (f"{r.vector_g}" if False else (g.vector if g is not None else "")), (f"{100 * g.agreement:.1f}" if g is not None else "")])
    L.append(_md_table(rows, ["budget", "pattern", "floor", "proposed vector", "FLOPs %", f"{split} agreement % [Wilson 95%]", "top-1 %",
                              "calibration agreement %", "geometric vector", "geometric agreement %"]))
    for e in fz["selections"]:
        if e["status"] != "selected":
            L.append(f"\n- **INFEASIBLE / not run:** B{int(100 * e['budget'])} {e['pattern']} floor {e['floor']}: {e.get('reason', '')}")
    inf = fz.get("infeasible_constrained_baselines", [])
    if inf:
        L.append("\nInfeasible constrained equal-drop baselines (never relaxed): " + "; ".join(f"B{int(100 * u['budget'])} {u['structure']} floor {u['floor']}" for u in inf))

    def tbl(kind, title, extra_cols=True):
        d = st[(st.comparison == kind) & (st.metric == "agreement")] if len(st) else pd.DataFrame()
        if d.empty:
            return
        t1 = {r.schedule_a + "|" + r.schedule_b: r for r in st[(st.comparison == kind) & (st.metric == "top1")].itertuples()}
        L.append(f"\n### {title}\n")
        rr = []
        for r in d.sort_values(["budget_target", "structure", "floor"]).itertuples():
            t = t1.get(r.schedule_a + "|" + r.schedule_b)
            rr.append([f"B{int(100 * r.budget_target)}", r.structure, int(r.floor), r.vector_a, r.vector_b, f"{r.flops_pct_a:.2f} / {r.flops_pct_b:.2f} (Δ{r.flops_diff_pp:+.2f})",
                       f"{r.count_a} vs {r.count_b}", f"{100 * r.diff:+.2f} [{100 * r.ci_low:+.2f}, {100 * r.ci_high:+.2f}]", f"{r.p_raw:.3g}", f"{r.p_holm:.3g}",
                       (f"{100 * t.diff:+.2f} [{100 * t.ci_low:+.2f}, {100 * t.ci_high:+.2f}] (Holm {t.p_holm:.3g})" if t is not None else "")])
        L.append(_md_table(rr, ["budget", "pattern", "floor", "A (proposed) vector", "B vector", "FLOPs % A / B", "agree count A vs B",
                                "Δ agreement pp [95% paired-bootstrap CI]", "raw p", "Holm p", "Δ top-1 pp [CI] (Holm)"]))

    L.append("\n## Paired comparisons (A = proposed, B = baseline; positive = proposed better)\n")
    tbl("prop_vs_geo", "PRIMARY: proposed vs geometric (Holm family: all cells × floors)")
    tbl("prop_vs_cons", "SECONDARY: proposed vs constrained equal-drop (same floor)")
    tbl("prop_vs_single", "SECONDARY: proposed vs single-stage at matched FLOPs")
    tbl("prop_vs_unc", "EXPLORATORY (diagnostic): proposed vs unconstrained equal-drop")
    sk = st.attrs.get("skipped", []) if len(st) else []
    if sk:
        L.append(f"\nExcluded pairs ({len(sk)}): " + "; ".join(f"{s['kind']}: {s['a']} vs {s['b']} ({s['reason']})" for s in sk[:30]))
    if len(st):
        pr = st[(st.comparison == "prop_vs_geo") & (st.metric == "agreement")]
        if len(pr):
            L.append(f"\nDescriptive summary of the primary family (agreement): n={len(pr)}, mean Δ {100 * pr['diff'].mean():+.2f} pp, "
                     f"positive {int((pr['diff'] > 0).sum())}, negative {int((pr['diff'] < 0).sum())}, Holm-significant positive "
                     f"{int(((pr['diff'] > 0) & pr['significant_holm_0.05']).sum())}, Holm-significant negative {int(((pr['diff'] < 0) & pr['significant_holm_0.05']).sum())}. "
                     "By budget: " + ", ".join(f"B{int(100 * b)} {100 * g['diff'].mean():+.2f}" for b, g in pr.groupby("budget_target")) + ".")
    # gap
    if split == "validation":
        g = m[(m.family == PROPOSED)].copy()
        gg = []
        geo2 = m[m.family == "geometric"].set_index(["budget_target", "structure"])
        for r in g.itertuples():
            k = (r.budget_target, r.structure)
            if k in geo2.index:
                cal_d = r.cal_agreement - geo2.loc[k].cal_agreement
                val_d = r.agreement - geo2.loc[k].agreement
                gg.append({"budget": r.budget_target, "pattern": r.structure, "floor": r.floor, "cal_delta_pp": 100 * cal_d, "val_delta_pp": 100 * val_d,
                           "gap_pp": 100 * (cal_d - val_d)})
        if gg:
            gd = pd.DataFrame(gg)
            L.append("\n## Calibration → validation generalisation gap (agreement Δ vs geometric: calibration minus validation)\n")
            L.append(f"Mean calibration Δ {gd.cal_delta_pp.mean():+.2f} pp, mean validation Δ {gd.val_delta_pp.mean():+.2f} pp, mean gap {gd.gap_pp.mean():+.2f} pp (H5 threshold 2 pp).\n")
            L.append(_md_table([[f"B{int(100 * r.budget)}", r.pattern, int(r.floor), f"{r.cal_delta_pp:+.2f}", f"{r.val_delta_pp:+.2f}", f"{r.gap_pp:+.2f}"] for r in gd.itertuples()],
                               ["budget", "pattern", "floor", "calibration Δ pp", "validation Δ pp", "gap pp"]))
    lat = out["latency"]
    L.append("\n## FLOPs, scorer overhead and latency (CPU; separate quantities; no latency inferred from FLOPs)\n")
    if len(lat):
        L.append(f"Device `{lat.device.iloc[0]}`, batch {lat.batch_size.iloc[0]}, warm-up {lat.warmup.iloc[0]}, {lat.timed_runs.iloc[0]} timed runs × {lat.num_batches.iloc[0]} batches; unpruned forward {lat.unpruned_forward_ms_median.iloc[0]:.0f} ms. CPU numbers do not establish GPU speedups.\n")
        L.append(_md_table([[f, len(g), f"{g.flops_pct.mean():.1f}", f"{g.end_to_end_ms_median.mean():.0f}", f"{g.scorer_overhead_ms_median.mean():.1f}"]
                            for f, g in lat[lat.family != "none"].groupby("family")], ["family", "schedules", "mean FLOPs %", "mean end-to-end ms", "mean scorer overhead ms"]))
    oh = out_path(cfg, "calibration", "calibration_overhead.json")
    if os.path.isfile(oh):
        o = json.load(open(oh))
        L.append(f"\nCalibration / search overhead (offline, reported separately; zero allocation overhead at inference since the schedule is a constant vector): "
                 f"{o['table_runs']} table runs in {o['table_seconds'] / 60:.1f} min; selection + {o['verification_runs']} verification runs {o['selection_seconds_incl_verification'] / 60:.1f} min; "
                 f"baselines on calibration {o['baselines_on_calibration_seconds'] / 60:.1f} min; total calibration stage {o['total_calibration_stage_seconds'] / 60:.1f} min on {o['device']}; peak RSS ≈ {o['peak_rss_mb_after']:.0f} MB.")
    vp = out_path(cfg, "calibration", "proxy_validity.json")
    if os.path.isfile(vp):
        v = json.load(open(vp))
        L.append(f"\n## Proxy validity on the calibration set (H1)\n\n" + (f"n = {v['n_points']} points; pooled Spearman(proxy, measured disagreement) = {v['pooled_spearman']:.2f} (p={v['pooled_p']:.2g}); mean within-cell Spearman {v['mean_within_cell_spearman']:.2f}; threshold {v['threshold']} → H1 {'met' if v['h1_met'] else 'NOT met'}." if 'pooled_spearman' in v else str(v)))
    L.append("\n## Limitations\n")
    L += ["- DeiT-Tiny, CLS-attention scorer (weak at layer 2, see evidence ledger row 3), first layer 2, two 3-stage patterns; no architecture generalisation.",
          "- The validation images were part of the earlier Phase 2 design history; they are held out only with respect to Phase 3 schedule selection. Final claims rest on the fresh confirmatory set (evaluated once).",
          "- Calibration n = 500 limits the ability to rank candidates that differ by < ~1–2 pp; proposed schedules may therefore not beat geometric.",
          "- Analytic FLOPs; CPU latency noisy and not comparable across runs; Pred-KL not used.",
          "- Cells where the floor cannot be met within the FLOPs band are reported as infeasible."]
    txt = "\n".join(map(str, L)) + "\n"
    rp = out_path(cfg, "result.md")
    open(out_path(cfg, f"result_{split}.md"), "w", encoding="utf-8").write(txt)      # generated, per split; never touches hand-written text
    if not os.path.isfile(rp):
        open(rp, "w", encoding="utf-8").write(txt)
