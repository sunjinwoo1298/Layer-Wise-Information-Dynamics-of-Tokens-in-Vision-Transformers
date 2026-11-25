"""
Phase 4 data: builds/caches tensors for the frozen P4 manifests (995 classes; classes 0-4 excluded as documented), verifying every encoded-image SHA-1
against the manifest. P4-CONFIRM is GUARDED: it cannot be loaded unless (a) the protocol freeze verifies, (b) frozen_method_configs.json exists,
(c) the environment variable PHASE4_CONFIRMATORY=1 is set deliberately, and (d) no previous confirmatory run marker exists.
"""

import glob
import hashlib
import io
import os
import subprocess
import sys
from typing import Dict

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
P4 = os.path.join(ROOT, "results", "phase4_adaptive_method_benchmark")
CACHE = {"dev": os.path.join(ROOT, "data", "p4_dev_cache.pt"), "confirm": os.path.join(ROOT, "data", "p4_confirm_cache.pt")}
MANIFEST = {"dev": os.path.join(P4, "splits", "p4_dev_manifest.csv"), "confirm": os.path.join(P4, "splits", "p4_confirm_manifest.csv")}


class ConfirmatoryGuardError(RuntimeError):
    pass


def _verify_frozen_config() -> None:
    """The confirmatory loader verifies the ACTUAL frozen_method_configs.json bytes against its .sha256 and every dependency file hash listed in it."""
    import hashlib
    import json as _json
    fj = os.path.join(P4, "frozen_method_configs.json")
    fs = os.path.join(P4, "frozen_method_configs.sha256")
    if not os.path.isfile(fs):
        raise ConfirmatoryGuardError("frozen_method_configs.sha256 missing")
    raw = open(fj, "rb").read()
    if hashlib.sha256(raw).hexdigest() != open(fs).read().strip():
        raise ConfirmatoryGuardError("frozen_method_configs.json does not match its recorded SHA-256")
    for rel, h in _json.loads(raw).get("file_sha256", {}).items():
        if hashlib.sha256(open(os.path.join(ROOT, rel), "rb").read()).hexdigest() != h:
            raise ConfirmatoryGuardError("frozen dependency changed after freeze: " + rel)


def _guard_confirm() -> None:
    if os.environ.get("PHASE4_CONFIRMATORY") != "1":
        raise ConfirmatoryGuardError("P4-CONFIRM is locked: PHASE4_CONFIRMATORY=1 is required and is only set at Gate 5.")
    if not os.path.isfile(os.path.join(P4, "frozen_method_configs.json")):
        raise ConfirmatoryGuardError("P4-CONFIRM is locked until frozen_method_configs.json exists (Gate 4 freeze).")
    if os.path.isfile(os.path.join(P4, "confirmatory_run_done.marker")):
        raise ConfirmatoryGuardError("P4-CONFIRM was already evaluated once; a second run is refused.")
    _verify_frozen_config()
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "phase4_freeze_protocol.py"), "--verify"], capture_output=True, text=True)
    if r.returncode != 0:
        raise ConfirmatoryGuardError("protocol freeze verification failed:\n" + r.stdout + r.stderr)


def load_split(name: str) -> Dict[str, object]:
    """name in {'dev', 'confirm'}. Returns dict(images, labels, ids, source_rows)."""
    if name not in ("dev", "confirm"):
        raise ValueError(name)
    if name == "confirm":
        _guard_confirm()
    man = pd.read_csv(MANIFEST[name])
    rows = man.source_row.to_numpy()
    if os.path.isfile(CACHE[name]):
        d = torch.load(CACHE[name], map_location="cpu", weights_only=False)
        if np.array_equal(np.asarray(d["source_rows"]), rows):
            return d
    import pyarrow.parquet as pq
    from PIL import Image
    from src.data.dataset import get_transforms  # read-only reuse of the Phase 1-3 preprocessing
    tf = get_transforms(224)
    files = sorted(glob.glob(os.path.join(ROOT, "data", "imagenet_val_parquet", "**", "val-*.parquet"), recursive=True))
    offs, off = [], 0
    for f in files:
        offs.append(off); off += pq.ParquetFile(f).metadata.num_rows
    want: Dict[int, list] = {}
    for pos, r in enumerate(rows):
        want.setdefault(int(r), []).append(pos)
    images = torch.empty(len(rows), 3, 224, 224)
    enc = [""] * len(rows)
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        r0 = 0
        for g in range(pf.num_row_groups):
            n_g = pf.metadata.row_group(g).num_rows
            hit = [offs[fi] + r0 + j for j in range(n_g) if (offs[fi] + r0 + j) in want]
            if hit:
                imgs = pf.read_row_group(g, columns=["image"]).column("image").to_pylist()
                for gr in hit:
                    raw = imgs[gr - offs[fi] - r0]["bytes"]
                    im = Image.open(io.BytesIO(raw)).convert("RGB")
                    for pos in want[gr]:
                        images[pos] = tf(im)
                        enc[pos] = hashlib.sha1(raw).hexdigest()
            r0 += n_g
    if enc != man.encoded_image_sha1.tolist():
        raise RuntimeError("encoded image SHA-1 mismatch vs manifest")
    d = {"images": images, "labels": torch.tensor(man.label.to_numpy(), dtype=torch.long), "ids": man.image_id.tolist(), "source_rows": rows,
         "encoded_sha1": enc}
    torch.save(d, CACHE[name])
    return d
