#!/usr/bin/env python3
"""
Phase 4 Gate 4, step 0: fix the 500-image P4-DEV calibration subset used for the supplementary Phase 3 re-selection (identified by manifest + hash
BEFORE any selection runs). Uses the unchanged Phase 3 split function on the P4-DEV manifest (seed 42): one image from each of 500 randomly chosen
classes = calibration; the other 1,490 P4-DEV images = held-out. P4-CONFIRM is not read. Refuses to overwrite.
"""
import hashlib
import json
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.data.phase3_splits import make_cal_val_split, split_checksum  # noqa: E402  (read-only reuse)

S = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark", "splits")
out_json = os.path.join(S, "p4_dev_cal500_summary.json")
if os.path.exists(out_json):
    sys.exit("already exists; refusing to overwrite")
man = pd.read_csv(os.path.join(S, "p4_dev_manifest.csv"))
m = man.rename(columns={"encoded_image_sha1": "image_sha1"})
sp = make_cal_val_split(m, 500, 42)
sp["dev_position"] = range(len(sp))                      # position in the P4-DEV manifest (index into the cached tensors)
cal, held = sp[sp.split == "calibration"], sp[sp.split == "validation"]
cal.to_csv(os.path.join(S, "p4_dev_cal500_manifest.csv"), index=False)
held.to_csv(os.path.join(S, "p4_dev_heldout1490_manifest.csv"), index=False)
summ = {"split_function": "src.data.phase3_splits.make_cal_val_split(manifest=P4-DEV, n_cal=500, seed=42) (unchanged Phase 3 code)",
        "n_calibration": len(cal), "n_heldout": len(held), "calibration_classes": int(cal.label.nunique()), "heldout_classes": int(held.label.nunique()),
        "calibration_split_checksum": split_checksum(sp, "calibration"), "heldout_split_checksum": split_checksum(sp, "validation"),
        "calibration_ids_sha256": hashlib.sha256(",".join(cal.image_id).encode()).hexdigest(),
        "heldout_ids_sha256": hashlib.sha256(",".join(held.image_id).encode()).hexdigest(),
        "cal_manifest_file_sha256": hashlib.sha256(open(os.path.join(S, "p4_dev_cal500_manifest.csv"), "rb").read()).hexdigest(),
        "heldout_manifest_file_sha256": hashlib.sha256(open(os.path.join(S, "p4_dev_heldout1490_manifest.csv"), "rb").read()).hexdigest(),
        "parent": "splits/p4_dev_manifest.csv (file sha256 69201fd9589657e21d9d35a12b805740d5ac30dc0c1258e2894589921107f7d2)", "disjoint": True}
assert not (set(cal.image_id) & set(held.image_id))
json.dump(summ, open(out_json, "w"), indent=1)
print(json.dumps(summ, indent=1))
