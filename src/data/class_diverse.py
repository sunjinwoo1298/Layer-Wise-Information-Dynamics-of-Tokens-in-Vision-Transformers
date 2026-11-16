"""
Deterministic class-balanced ImageNet-val subset construction (Phase 2 class-diverse validation).

Sampling algorithm (documented, seeded, independent of any model output):
  1. Read the `label` column of every validation parquet shard (source order = shard order, then row order).
  2. For every class c in 0..999, collect its source rows (50 per class in ImageNet-1k val).
  3. With numpy RandomState(seed): for each class in ascending order draw `per_class` rows WITHOUT replacement
     (a seeded permutation of that class's rows, first `per_class` taken). Classes with fewer rows contribute all rows.
  4. The selected (class, row) pairs are put in a seeded random order (RandomState(seed + 1)) so that every evaluation
     batch mixes many classes. That order is the fixed evaluation order for every schedule/scorer/seed.
  5. Images are decoded and preprocessed with the project's `get_transforms` (Resize 256 bicubic, CenterCrop 224,
     ImageNet mean/std) -- identical to Phase 1 -- and cached as one tensor file.
The subset is class-balanced by construction but is a sample of the validation set, not the full distribution.
"""

import glob
import hashlib
import io
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .dataset import get_transforms


def class_coverage_summary(labels: List[int], num_classes: int = 1000) -> Dict[str, Any]:
    """Counts of images/classes and per-class count statistics (over classes that are present AND over all classes)."""
    lab = np.asarray(labels, dtype=int)
    hist = np.bincount(lab, minlength=num_classes)
    present = hist[hist > 0]
    return {
        "num_images": int(len(lab)), "num_classes_total": int(num_classes), "num_distinct_classes": int((hist > 0).sum()),
        "classes_missing": int((hist == 0).sum()),
        "per_class_count_over_all_classes": {"min": int(hist.min()), "median": float(np.median(hist)), "max": int(hist.max())},
        "per_class_count_over_present_classes": ({"min": int(present.min()), "median": float(np.median(present)),
                                                  "max": int(present.max())} if len(present) else {}),
        "count_histogram": {str(int(k)): int(v) for k, v in zip(*np.unique(hist, return_counts=True))},
        "class_histogram": {str(i): int(c) for i, c in enumerate(hist)},
    }


def select_balanced_rows(source_labels: np.ndarray, per_class: int, seed: int, num_classes: int = 1000) -> np.ndarray:
    """Return source-row indices (in evaluation order) per the algorithm in the module docstring. Pure function of its inputs."""
    rs = np.random.RandomState(seed)
    chosen: List[int] = []
    for c in range(num_classes):
        rows = np.flatnonzero(source_labels == c)
        if len(rows) == 0:
            continue
        perm = rs.permutation(len(rows))
        chosen.extend(rows[np.sort(perm[:per_class])].tolist())
    chosen = np.asarray(chosen, dtype=int)
    order = np.random.RandomState(seed + 1).permutation(len(chosen))
    return chosen[order]


def manifest_hash(manifest: pd.DataFrame) -> str:
    """SHA-256 over the ordered (position, source_row, label, image_sha1) records -- changes if the subset or order changes."""
    h = hashlib.sha256()
    for r in manifest[["position", "source_row", "label", "image_sha1"]].itertuples(index=False):
        h.update(f"{r.position},{r.source_row},{r.label},{r.image_sha1}\n".encode())
    return h.hexdigest()


def build_manifest(images: torch.Tensor, labels: torch.Tensor, source_rows: np.ndarray, raw_sha1: Optional[List[str]] = None
                   ) -> pd.DataFrame:
    n = len(images)
    return pd.DataFrame({
        "position": range(n), "image_id": [f"val_row_{int(r):06d}" for r in source_rows], "source_row": [int(r) for r in source_rows],
        "label": labels.tolist(), "image_sha1": [hashlib.sha1(images[i].numpy().tobytes()).hexdigest() for i in range(n)],
        "encoded_image_sha1": raw_sha1 if raw_sha1 is not None else [""] * n})


def build_class_diverse_subset(parquet_dir: str, per_class: int, seed: int, cache_path: str,
                               num_classes: int = 1000) -> Dict[str, Any]:
    """Build (or load from cache) the subset tensors. Returns dict(images, labels, manifest, summary)."""
    import pyarrow.parquet as pq
    from PIL import Image
    files = sorted(glob.glob(os.path.join(parquet_dir, "**", "val-*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No val-*.parquet shards under {parquet_dir}")
    # label column only (cheap) -> global row index across shards
    shard_labels, offsets, off = [], [], 0
    for f in files:
        t = pq.read_table(f, columns=["label"]).column("label").to_numpy()
        shard_labels.append(t); offsets.append(off); off += len(t)
    all_labels = np.concatenate(shard_labels)
    rows = select_balanced_rows(all_labels, per_class, seed, num_classes)

    meta = {"parquet_files": [os.path.basename(f) for f in files], "total_source_rows": int(len(all_labels)),
            "per_class": per_class, "seed": seed}
    if os.path.isfile(cache_path):
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        if d.get("source_rows") is not None and np.array_equal(np.asarray(d["source_rows"]), rows):
            return _finish(d["images"], d["labels"], rows, d["encoded_sha1"], meta, num_classes)

    tf = get_transforms(224)
    images = torch.empty(len(rows), 3, 224, 224)
    labels = torch.tensor(all_labels[rows], dtype=torch.long)
    enc_sha1 = [""] * len(rows)
    want: Dict[int, List[int]] = {}
    for pos, r in enumerate(rows):
        want.setdefault(int(r), []).append(pos)
    for fi, f in enumerate(files):
        pf = pq.ParquetFile(f)
        base = offsets[fi]
        r0 = 0
        for g in range(pf.num_row_groups):
            n_g = pf.metadata.row_group(g).num_rows
            hit = [base + r0 + j for j in range(n_g) if (base + r0 + j) in want]
            if hit:
                tbl = pf.read_row_group(g, columns=["image", "label"])
                imgs = tbl.column("image").to_pylist()
                for gr in hit:
                    rec = imgs[gr - base - r0]
                    raw = rec["bytes"]
                    im = Image.open(io.BytesIO(raw))
                    im = im.convert("RGB") if im.mode != "RGB" else im
                    for pos in want[gr]:
                        images[pos] = tf(im)
                        enc_sha1[pos] = hashlib.sha1(raw).hexdigest()
            r0 += n_g
    if any(s == "" for s in enc_sha1):
        raise RuntimeError("Some selected rows were not found in the parquet shards.")
    torch.save({"images": images, "labels": labels, "source_rows": rows, "encoded_sha1": enc_sha1, "meta": meta}, cache_path)
    return _finish(images, labels, rows, enc_sha1, meta, num_classes)


def _finish(images, labels, rows, enc_sha1, meta, num_classes) -> Dict[str, Any]:
    manifest = build_manifest(images, labels, rows, enc_sha1)
    summary = {**class_coverage_summary(labels.tolist(), num_classes), "manifest_sha256": manifest_hash(manifest),
               "sampling": {**meta, "algorithm": __doc__.split("Sampling algorithm")[1].split("The subset is")[0].strip(),
                            "preprocessing": "get_transforms(224): Resize 256 bicubic, CenterCrop 224, ToTensor, ImageNet mean/std"}}
    return {"images": images, "labels": labels, "manifest": manifest, "summary": summary}
