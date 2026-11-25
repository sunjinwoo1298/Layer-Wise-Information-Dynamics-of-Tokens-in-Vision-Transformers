#!/usr/bin/env python3
"""
Phase 4 Gate 3: compare OUR ToMe (timm 1.0.14, src/phase4/tome.py) with the official ToMe outputs produced in the isolated timm 0.4.12 environment
(results/phase4_adaptive_method_benchmark/parity/official_tome_*). P4-DEV only. Diagnostic, not a benchmark result.
"""
import json
import os
import sys
import time

import numpy as np
import timm
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.phase4.data import load_split  # noqa: E402
from src.phase4.engine import Compressor, flops_pct, run_compressed  # noqa: E402
from src.phase4.tome import ToMe, parse_r  # noqa: E402

PAR = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark", "parity")
torch.set_num_threads(4)
d = load_split("dev")
X, y = d["images"], d["labels"]
model = timm.create_model("deit_tiny_patch16_224", pretrained=True).eval()
off_counts = json.load(open(os.path.join(PAR, "official_tome_token_counts.json")))
off = torch.load(os.path.join(PAR, "official_tome_logits_first200.pt"), weights_only=False)
meta = json.load(open(os.path.join(PAR, "official_tome_run_meta.json")))
res = {"our_timm": timm.__version__, "official_meta": {k: meta[k] for k in ("tome_commit", "timm", "torch", "python")}}

# 1) token-count parity over the whole (r, inflection) grid
mism, flops_gap = [], []
for key, c in off_counts.items():
    if key == "r=0":
        spec = 0
    else:
        r = int(key.split(",")[0].split("=")[1]); infl = float(key.split("=")[-1])
        spec = r if infl == 0 else (r, infl)
    _, info = run_compressed(model, X[:1], ToMe(spec))
    if info["entering"] != c["entering"] or info["leaving"] != c["leaving"]:
        mism.append({"config": key, "ours_leaving": info["leaving"], "official_leaving": c["leaving"]})
    flops_gap.append(abs(flops_pct(info["entering"], info["leaving"]) - flops_pct(c["entering"], c["leaving"])))
res["count_parity"] = {"configs_compared": len(off_counts), "mismatches": len(mism), "max_flops_pct_gap_between_ours_and_official_counts": max(flops_gap),
                       "examples": mism[:5]}

# 2) unpruned parity across timm versions + logits parity on the first 200 dev images
with torch.no_grad():
    ours_ref = torch.cat([model(X[i:i + 25]) for i in range(0, 200, 25)])
ref = off["logits"]["ref_unpatched_timm0.4.12"]
res["unpruned_timm1.0.14_vs_timm0.4.12_first200"] = {"max_abs_logit_diff": float((ours_ref - ref).abs().max()),
                                                    "pred_agree": int((ours_ref.argmax(1) == ref.argmax(1)).sum()), "n": 200}
specs = {"r=8": 8, "r=16": 16, "r=11,inflection=-0.5": (11, -0.5), "r=7,inflection=-1.0": (7, -1.0), "r=30,inflection=0.5": (30, 0.5), "r=0": 0}
res["logit_parity_first200"] = {}
for k, spec in specs.items():
    lg = torch.cat([run_compressed(model, X[i:i + 25], ToMe(spec))[0] for i in range(0, 200, 25)])
    o = off["logits"][k]
    res["logit_parity_first200"][k] = {"max_abs_logit_diff": float((lg - o).abs().max()), "mean_abs_logit_diff": float((lg - o).abs().mean()),
                                       "pred_agree_with_official": int((lg.argmax(1) == o.argmax(1)).sum()), "n": 200,
                                       "ours_agree_with_ours_unpruned": int((lg.argmax(1) == ours_ref.argmax(1)).sum()),
                                       "official_agree_with_official_unpruned": int((o.argmax(1) == ref.argmax(1)).sum())}

# 3) full P4-DEV for two configs
with torch.no_grad():
    full_ref = torch.cat([model(X[i:i + 25]) for i in range(0, len(X), 25)]).argmax(1)
res["full_dev"] = {"ours_unpruned_top1": int((full_ref == y).sum()), "official_unpruned_top1": meta["full_dev"]["unpruned"]["top1_count"], "n": len(X)}
for k, spec in {"r=16": 16, "r=11,inflection=-0.5": (11, -0.5)}.items():
    t0 = time.time()
    pr = torch.cat([run_compressed(model, X[i:i + 25], ToMe(spec))[0].argmax(1) for i in range(0, len(X), 25)])
    res["full_dev"][k] = {"ours_agree_count": int((pr == full_ref).sum()), "ours_top1": int((pr == y).sum()),
                          "official_agree_count": meta["full_dev"][k]["agree_count_vs_unpruned_timm0.4.12"], "official_top1": meta["full_dev"][k]["top1_count"],
                          "ours_seconds": round(time.time() - t0, 1)}
json.dump(res, open(os.path.join(PAR, "tome_parity_result.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
