"""
Phase 3 data splits (deterministic, model-output independent).

1. Calibration / validation split of the existing 2,000-image class-diverse subset (2 images per class):
     RandomState(seed) permutes the class ids; the first `n_cal` classes contribute ONE image each (chosen by RandomState(seed+1)) to the
     calibration set; every other image is validation. => calibration = 500 images from 500 distinct classes; validation = 1,500 images
     covering all 1,000 classes. Disjoint by construction (asserted). Image order inside each split follows the parent manifest order.
2. Fresh confirmatory set: 2 images per class drawn (seeded, class-balanced, same algorithm as the class-diverse subset) from the ImageNet-val
   rows NOT used by the 2,000-image subset (source_row exclusion, asserted disjoint). Same preprocessing (`get_transforms(224)`).
"""

import glob
import hashlib
import io
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .class_diverse import _finish, build_manifest, class_coverage_summary, manifest_hash, select_balanced_rows
from .dataset import get_transforms


def make_cal_val_split(manifest: pd.DataFrame, n_cal: int = 500, seed: int = 42) -> pd.DataFrame:
    """Return a copy of `manifest` with a `split` column in {'calibration','validation'} (see module docstring)."""
    m = manifest.copy().reset_index(drop=True)
    classes = np.array(sorted(m.label.unique()))
    if n_cal > len(classes):
        raise ValueError(f"n_cal={n_cal} exceeds the number of classes ({len(classes)})")
    chosen = np.random.RandomState(seed).permutation(classes)[:n_cal]
    rs2 = np.random.RandomState(seed + 1)
    m["split"] = "validation"
    for c in sorted(chosen.tolist()):
        idx = np.flatnonzero(m.label.to_numpy() == c)
        m.loc[idx[rs2.randint(len(idx))], "split"] = "calibration"
    assert_split_disjoint(m)
    return m


def assert_split_disjoint(m: pd.DataFrame) -> None:
    cal, val = set(m[m.split == "calibration"].image_id), set(m[m.split == "validation"].image_id)
    if cal & val:
        raise AssertionError("calibration and validation image ids overlap")
    if len(cal) + len(val) != len(m) or len(set(m.image_id)) != len(m):
        raise AssertionError("split does not partition the manifest (duplicate or missing ids)")


def split_checksum(m: pd.DataFrame, split: str) -> str:
    """SHA-256 over (position, source_row, label, image_sha1) of the split's rows in manifest order."""
    h = hashlib.sha256()
    for r in m[m.split == split][["position", "source_row", "label", "image_sha1"]].itertuples(index=False):
        h.update(f"{r.position},{r.source_row},{r.label},{r.image_sha1}\n".encode())
    return h.hexdigest()


def build_confirmatory_subset(parquet_dir: str, exclude_rows: np.ndarray, per_class: int, seed: int, cache_path: str,
                              num_classes: int = 1000) -> Dict[str, Any]:
    """Fresh class-balanced set from ImageNet-val rows not in `exclude_rows` (cached; manifest + checksum like the parent subset)."""
    import pyarrow.parquet as pq
    from PIL import Image
    files = sorted(glob.glob(os.path.join(parquet_dir, "**", "val-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No val-*.parquet shards under {parquet_dir}")
    shard_labels, offsets, off = [], [], 0
    for f in files:
        t = pq.read_table(f, columns=["label"]).column("label").to_numpy()
        shard_labels.append(t); offsets.append(off); off += len(t)
    all_labels = np.concatenate(shard_labels)
    masked = all_labels.copy()
    masked[np.asarray(exclude_rows, dtype=int)] = -1                    # excluded rows can never be selected
    rows = select_balanced_rows(masked, per_class, seed, num_classes)
    if set(rows.tolist()) & set(np.asarray(exclude_rows).tolist()):
        raise AssertionError("confirmatory rows overlap the existing subset")
    meta = {"parquet_files": [os.path.basename(f) for f in files], "total_source_rows": int(len(all_labels)), "per_class": per_class,
            "seed": seed, "excluded_source_rows": int(len(exclude_rows)), "role": "fresh confirmatory set (disjoint from the 2,000-image subset)"}
    if os.path.isfile(cache_path):
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        if d.get("source_rows") is not None and np.array_equal(np.asarray(d["source_rows"]), rows):
            return _finish_confirm(d["images"], d["labels"], rows, d["encoded_sha1"], meta, num_classes)
    tf = get_transforms(224)
    images = torch.empty(len(rows), 3, 224, 224)
    labels = torch.tensor(all_labels[rows], dtype=torch.long)
    enc = [""] * len(rows)
    want: Dict[int, list] = {}
    for pos, r in enumerate(rows):
        want.setdefault(int(r), []).append(pos)
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        base, r0 = offsets[fi], 0
        for g in range(pf.num_row_groups):
            n_g = pf.metadata.row_group(g).num_rows
            hit = [base + r0 + j for j in range(n_g) if (base + r0 + j) in want]
            if hit:
                imgs = pf.read_row_group(g, columns=["image", "label"]).column("image").to_pylist()
                for gr in hit:
                    raw = imgs[gr - base - r0]["bytes"]
                    im = Image.open(io.BytesIO(raw))
                    im = im.convert("RGB") if im.mode != "RGB" else im
                    for pos in want[gr]:
                        images[pos] = tf(im)
                        enc[pos] = hashlib.sha1(raw).hexdigest()
            r0 += n_g
    if any(s == "" for s in enc):
        raise RuntimeError("Some selected confirmatory rows were not found in the parquet shards.")
    torch.save({"images": images, "labels": labels, "source_rows": rows, "encoded_sha1": enc, "meta": meta}, cache_path)
    return _finish_confirm(images, labels, rows, enc, meta, num_classes)


def _finish_confirm(images, labels, rows, enc, meta, num_classes) -> Dict[str, Any]:
    man = build_manifest(images, labels, rows, enc)
    man["split"] = "confirmatory"
    summ = {**class_coverage_summary(labels.tolist(), num_classes), "manifest_sha256": manifest_hash(man), "sampling": meta,
            "preprocessing": "get_transforms(224): Resize 256 bicubic, CenterCrop 224, ToTensor, ImageNet mean/std"}
    return {"images": images, "labels": labels, "manifest": man, "summary": summ, "source_rows": rows}
