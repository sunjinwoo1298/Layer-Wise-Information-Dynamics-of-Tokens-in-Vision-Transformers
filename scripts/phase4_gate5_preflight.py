#!/usr/bin/env python3
"""Phase 4 Gate 5 PREFLIGHT (reads no P4-CONFIRM images/labels; only the manifest CSV): verifies freeze integrity, absence of cache/marker/partial outputs,
manifest identity and disjointness, and records the final hashes. Writes gate5_preflight.json. Never calls load_split('confirm')."""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
S = os.path.join(P4, "splits")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


chk = {}
r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "phase4_freeze_protocol.py"), "--verify"], capture_output=True, text=True)
chk["protocol_freeze_verify"] = {"ok": r.returncode == 0, "output": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr.strip()}
fb = open(os.path.join(P4, "frozen_method_configs.json"), "rb").read()
fcfg = json.loads(fb)
chk["frozen_config_hash_matches"] = hashlib.sha256(fb).hexdigest() == open(os.path.join(P4, "frozen_method_configs.sha256")).read().strip()
bad = [k for k, h in fcfg["file_sha256"].items() if sha(os.path.join(ROOT, k)) != h]
chk["frozen_dependencies_unchanged"] = {"ok": not bad, "changed": bad, "n_files": len(fcfg["file_sha256"])}
conf = pd.read_csv(os.path.join(S, "p4_confirm_manifest.csv"))
dev = pd.read_csv(os.path.join(S, "p4_dev_manifest.csv"))
p13 = set(np.asarray(torch.load(os.path.join(ROOT, "data", "imagenet_class_diverse_subset.pt"), weights_only=False)["source_rows"]).tolist()) | \
      set(np.asarray(torch.load(os.path.join(ROOT, "data", "imagenet_confirmatory_2000.pt"), weights_only=False)["source_rows"]).tolist()) | set(range(250))
cs = set(conf.source_row)
chk["confirm_manifest"] = {"n": len(conf), "classes": int(conf.label.nunique()), "file_sha256": sha(os.path.join(S, "p4_confirm_manifest.csv")),
                           "equals_frozen": sha(os.path.join(S, "p4_confirm_manifest.csv")) == fcfg["split_identity"]["confirm_manifest_sha256"],
                           "ids_sha256": hashlib.sha256(",".join(conf.image_id).encode()).hexdigest(),
                           "disjoint_from_dev": not (cs & set(dev.source_row)), "disjoint_from_phase1_3": not (cs & p13), "n_phase1_3_images_checked": len(p13),
                           "excluded_classes_absent": all(c not in set(conf.label) for c in range(5))}
out = os.path.join(P4, "per_image_results")
chk["clean_state"] = {"confirm_cache_absent": not os.path.exists(os.path.join(ROOT, "data", "p4_confirm_cache.pt")),
                      "marker_absent": not os.path.exists(os.path.join(P4, "confirmatory_run_done.marker")),
                      "no_partial_outputs": not (os.path.isdir(out) and os.listdir(out)), "no_confirmatory_environment_json": not os.path.exists(os.path.join(P4, "confirmatory_environment.json"))}
g = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, r'%s'); import os; os.environ.pop('PHASE4_CONFIRMATORY', None); from src.phase4 import data; data.load_split('confirm')" % ROOT],
                   capture_output=True, text=True)
chk["loader_locked_without_env_var"] = "ConfirmatoryGuardError" in g.stderr and not os.path.exists(os.path.join(ROOT, "data", "p4_confirm_cache.pt"))
rec = {"original_protocol_freeze_sha256_file": open(os.path.join(P4, "protocol_freeze.sha256")).read().strip(),
       "frozen_method_configs_sha256": hashlib.sha256(fb).hexdigest(), "protocol_amendments_md_sha256": sha(os.path.join(P4, "protocol_amendments.md")),
       "gate5_runner_sha256": sha(os.path.join(ROOT, "scripts", "phase4_gate5_confirmatory.py")), "development_results_md_sha256": sha(os.path.join(P4, "development_results.md")),
       "checks": chk, "command": "set PHASE4_CONFIRMATORY=1 && conda run --no-capture-output -n minor python -u scripts\\phase4_gate5_confirmatory.py"}
ok = chk["protocol_freeze_verify"]["ok"] and chk["frozen_config_hash_matches"] and chk["frozen_dependencies_unchanged"]["ok"] and chk["loader_locked_without_env_var"] and \
    all(chk["clean_state"].values()) and all(v for k, v in chk["confirm_manifest"].items() if k.startswith(("equals", "disjoint", "excluded")))
rec["all_checks_pass"] = bool(ok)
json.dump(rec, open(os.path.join(P4, "gate5_preflight.json"), "w"), indent=1)
print(json.dumps(rec, indent=1))
