#!/usr/bin/env python3
"""
Phase 4 protocol freeze. `--create` writes protocol_freeze.json (full SHA-256 of every frozen file) and protocol_freeze.sha256
(SHA-256 of that JSON file's bytes); refuses to overwrite. `--verify` recomputes everything and exits non-zero on any mismatch.
The confirmatory runner must call verify() first.
"""
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
FILES = [
    "results/phase4_adaptive_method_benchmark/benchmark_protocol.md",
    "results/phase4_adaptive_method_benchmark/benchmark_config.yaml",
    "results/phase4_adaptive_method_benchmark/method_config_matrix.csv",
    "results/phase4_adaptive_method_benchmark/splits/p4_dev_manifest.csv",
    "results/phase4_adaptive_method_benchmark/splits/p4_confirm_manifest.csv",
    "results/phase4_adaptive_method_benchmark/splits/split_summary.json",
    "results/phase4_adaptive_method_benchmark/diffrate_deit_tiny_rates_as_published.json",
    "results/phase4_adaptive_method_benchmark/diffrate_flops_interpretations.csv",
    "results/phase4_adaptive_method_benchmark/phase3_replication_mapping.csv",
    "results/phase4_adaptive_method_benchmark/analytic_candidates.csv",
    "results/phase4_adaptive_method_benchmark/analytic_feasibility_summary.csv",
    "results/phase4_adaptive_method_benchmark/phase3_documentation_errata.md",
    "results/phase3_budget_aware_allocation/frozen_schedules.json",
    "results/phase3_budget_aware_allocation/frozen.sha256",
    "scripts/phase4_make_splits.py",
    "scripts/phase4_analytic_feasibility.py",
    "scripts/phase4_gate2_tables.py",
    "scripts/phase4_freeze_protocol.py",
]


def sha(path):
    return hashlib.sha256(open(os.path.join(ROOT, path), "rb").read()).hexdigest()


def compute():
    d = {f: sha(f) for f in FILES if not f.endswith("phase4_freeze_protocol.py")}
    return d


def create():
    jp, sp = os.path.join(P4, "protocol_freeze.json"), os.path.join(P4, "protocol_freeze.sha256")
    if os.path.exists(jp):
        sys.exit("protocol_freeze.json exists; changes require a dated entry in protocol_amendments.md and a new freeze file")
    fz = json.load(open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")))
    txt = open(os.path.join(ROOT, "results", "phase3_budget_aware_allocation", "frozen_schedules.json")).read()
    obj = {"protocol_version": "phase4-gate2-v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "Full SHA-256 hashes. frozen_schedules.json content hash (load_frozen convention) is "
                   "hashlib.sha256(text.encode()) == contents of frozen.sha256; the raw-bytes hash is listed under files.",
           "phase3_frozen_content_sha256": hashlib.sha256(txt.encode()).hexdigest(), "phase3_code_commit_recorded": fz["code_commit"],
           "files_sha256": compute(), "git_commits_made": False}
    s = json.dumps(obj, indent=1, sort_keys=True)
    open(jp, "wb").write(s.encode("utf-8"))  # exact bytes: the recorded hash equals sha256sum of the file
    open(sp, "wb").write(hashlib.sha256(s.encode("utf-8")).hexdigest().encode("ascii"))
    print(s)
    print("protocol_freeze.sha256 =", open(sp).read())


def verify():
    jp, sp = os.path.join(P4, "protocol_freeze.json"), os.path.join(P4, "protocol_freeze.sha256")
    raw = open(jp, "rb").read()
    s = raw.decode("utf-8")
    bad = []
    if hashlib.sha256(raw).hexdigest() != open(sp).read().strip():
        bad.append("protocol_freeze.json does not match protocol_freeze.sha256")
    obj = json.loads(s)
    cur = compute()
    for f, h in obj["files_sha256"].items():
        if cur.get(f) != h:
            bad.append(f"MISMATCH {f}")
    if bad:
        sys.exit("FREEZE VERIFICATION FAILED:\n" + "\n".join(bad))
    print("freeze verified:", len(cur), "files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    create() if a.create else verify()
