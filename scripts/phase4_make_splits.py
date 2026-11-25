#!/usr/bin/env python3
"""
Phase 4 Gate 1: build the P4-DEV (2/class, seed 2026) and P4-CONFIRM (3/class, seed 7331) manifests.
Reads Phase 1-3 caches read-only; writes ONLY under results/phase4_adaptive_method_benchmark/splits/. Refuses to overwrite.
Rows already used in Phases 1-3 (parent subset, Phase 3 confirmatory, Phase 1 dev250 = first 250 rows) are excluded.
Classes with fewer than (2 + 3) unused rows are excluded (this removes the 5 classes exhausted by dev250).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark", "splits")
PARQUET = os.path.join(ROOT, "data", "imagenet_val_parquet", "data", "*.parquet")
N_DEV, N_CONF, SEED_DEV, SEED_CONF = 2, 3, 2026, 7331


def main():
    if os.path.exists(os.path.join(OUT, "split_summary.json")):
        sys.exit("splits already exist; refusing to overwrite")
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(PARQUET))
    labels, enc_sha = [], []
    for f in files:  # one shard at a time to bound memory
        t = pq.read_table(f)
        labels += t.column("label").to_pylist()
        enc_sha += [hashlib.sha1(x["bytes"]).hexdigest() for x in t.column("image").to_pylist()]
        del t
    labels = np.array(labels)
    assert len(labels) == 50000
    used = set(torch.load(os.path.join(ROOT, "data", "imagenet_class_diverse_subset.pt"), weights_only=False)["source_rows"])
    p3 = set(torch.load(os.path.join(ROOT, "data", "imagenet_confirmatory_2000.pt"), weights_only=False)["source_rows"])
    used_all = used | p3 | set(range(250))
    avail = {c: [i for i in np.where(labels == c)[0] if int(i) not in used_all] for c in range(1000)}
    classes = [c for c in range(1000) if len(avail[c]) >= N_DEV + N_CONF]
    excluded = [c for c in range(1000) if c not in classes]
    dev_rows, conf_rows = [], []
    rng_d, rng_c = np.random.RandomState(SEED_DEV), np.random.RandomState(SEED_CONF)
    for c in classes:
        pool = list(avail[c])
        d = [int(x) for x in rng_d.choice(pool, N_DEV, replace=False)]
        rest = [x for x in pool if x not in d]
        k = [int(x) for x in rng_c.choice(rest, N_CONF, replace=False)]
        dev_rows += d
        conf_rows += k
    assert not (set(dev_rows) & set(conf_rows)) and not ((set(dev_rows) | set(conf_rows)) & used_all)
    summary = {"classes_included": len(classes), "classes_excluded": excluded, "unused_rows_total": 50000 - len(used_all),
               "phase1to3_rows_excluded": len(used_all), "seeds": {"dev": SEED_DEV, "confirm": SEED_CONF}}
    for name, rows in (("p4_dev", dev_rows), ("p4_confirm", conf_rows)):
        df = pd.DataFrame({"position": range(len(rows)), "image_id": [f"val_{r:05d}" for r in rows], "source_row": rows,
                           "label": labels[rows], "encoded_image_sha1": [enc_sha[r] for r in rows], "split": name})
        path = os.path.join(OUT, f"{name}_manifest.csv")
        df.to_csv(path, index=False)
        summary[name] = {"n": len(df), "per_class": int(df.groupby("label").size().iloc[0]), "classes": int(df.label.nunique()),
                         "ids_sha256": hashlib.sha256(",".join(df.image_id).encode()).hexdigest(),
                         "manifest_file_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    summary["disjoint_dev_confirm"] = True
    summary["disjoint_from_phase1_3"] = True
    json.dump(summary, open(os.path.join(OUT, "split_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
