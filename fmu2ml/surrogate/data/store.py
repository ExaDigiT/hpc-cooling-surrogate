"""
Out-of-core memmap substrate for multi-rate surrogate training.

The parallel FMU generator writes one self-contained output file per chunk
(``chunk_<id>/fmu_output_*_operational.parquet``, carrying both input and output
columns). Loading all chunks at full resolution to RAM is what crashes training,
even though every Dataset only ever strides the data down afterward.

This module converts each chunk, **one at a time** (so peak RAM is a single
chunk), into anti-aliased, decimated ``.npy`` arrays — one per requested sampling
rate — and records a manifest. Datasets then ``np.load(..., mmap_mode="r")`` and
slice windows on demand, so the full dataset never resides in RAM and each
decoder group can read its own rate.

Key properties
--------------
* **Anti-aliased decimation** (block-mean by default) replaces plain striding,
  which folds sub-Nyquist energy back into the passband (the suspected cause of
  the ``V_flow_sec`` degradation). Block-mean is a boxcar low-pass + decimate.
* **Multi-rate**: a separate decimated array is written per rate. Rates must be
  integer multiples of the smallest rate so window origins align across grids
  (enforced here and relied on by the dataset).
* **Idempotent**: existing, up-to-date arrays are skipped unless ``overwrite``.
"""

from typing import Dict, List, Tuple, Optional, Any
import glob
import json
import os
import re

import numpy as np
import pandas as pd


CHUNK_GLOB = "fmu_output_*_operational.parquet"
MANIFEST_NAME = "store_manifest.json"


# ─────────────────────────────────────────────────────────────────────────────
# Chunk discovery (shared with the per-file loader)
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_id(path: str) -> int:
    """Parse the integer id from a ``chunk_<id>`` path; -1 if absent."""
    m = re.search(r"chunk_(\d+)", os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else -1


def discover_chunk_files(
    data_dir: str,
    pattern: str = CHUNK_GLOB,
) -> List[Tuple[int, str]]:
    """
    Return ``[(chunk_id, output_parquet_path)]`` in natural chunk-id order.

    Looks for ``chunk_<id>/`` subdirectories and, inside each, the
    lexicographically first file matching ``pattern``. Chunks without a match
    are skipped.
    """
    chunk_dirs = sorted(glob.glob(os.path.join(data_dir, "chunk_*")), key=_chunk_id)
    out: List[Tuple[int, str]] = []
    for d in chunk_dirs:
        matches = sorted(glob.glob(os.path.join(d, pattern)))
        if matches:
            out.append((_chunk_id(d), matches[0]))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Anti-aliased decimation
# ─────────────────────────────────────────────────────────────────────────────

def block_mean_decimate(arr: np.ndarray, rate: int) -> np.ndarray:
    """
    Boxcar low-pass + decimate by ``rate`` along axis 0.

    Trailing rows that do not fill a full block are dropped. ``rate == 1`` is a
    no-op (returns a float32 copy). This is a crude but adequate anti-aliasing
    filter — unlike plain striding it does not alias sub-Nyquist energy into the
    passband.
    """
    arr = np.asarray(arr, dtype=np.float32)
    if rate <= 1:
        return arr.astype(np.float32, copy=True)
    n = (arr.shape[0] // rate) * rate
    if n == 0:
        return np.empty((0, arr.shape[1]), dtype=np.float32)
    blocks = arr[:n].reshape(n // rate, rate, arr.shape[1])
    return blocks.mean(axis=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Store construction
# ─────────────────────────────────────────────────────────────────────────────

def _validate_rates(rates: List[int]) -> List[int]:
    """Dedupe, sort, and require every rate to be a multiple of the smallest."""
    rates = sorted({int(r) for r in rates})
    if not rates or rates[0] < 1:
        raise ValueError(f"rates must be positive integers, got {rates!r}.")
    base = rates[0]
    bad = [r for r in rates if r % base != 0]
    if bad:
        raise ValueError(
            f"all rates must be integer multiples of the smallest ({base}); "
            f"offending: {bad}. Cross-rate window origins require a common base."
        )
    return rates


def _array_path(out_dir: str, chunk_id: int, rate: int) -> str:
    return os.path.join(out_dir, f"chunk_{chunk_id}_rate{rate}.npy")


def build_chunk_store(
    data_dir: str,
    out_dir: str,
    input_cols: List[str],
    dynamic_cols: List[str],
    rates: List[int],
    pattern: str = CHUNK_GLOB,
    time_col: str = "time",
    reduce: str = "mean",
    overwrite: bool = False,
    chunk_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Convert each ``chunk_<id>/`` output parquet into per-rate decimated memmaps.

    For every chunk (read one at a time → peak RAM ≈ one chunk), select the
    feature columns ``input_cols + dynamic_cols`` (sorted by ``time_col`` when
    present), and for each rate write an anti-aliased, decimated ``float32`` array
    ``chunk_<id>_rate<r>.npy`` whose columns are ``input_cols`` followed by
    ``dynamic_cols``. A ``store_manifest.json`` records the column layout, rates,
    and per-chunk row counts.

    Parameters
    ----------
    data_dir : str
        Directory of ``chunk_<id>/`` folders.
    out_dir : str
        Destination for the ``.npy`` arrays and the manifest (created if needed).
    input_cols, dynamic_cols : List[str]
        Feature columns; stored concatenated, with ``n_input = len(input_cols)``.
    rates : List[int]
        Distinct sampling rates to materialize (must share a common base; see
        :func:`_validate_rates`).
    reduce : str
        Decimation method. Only ``"mean"`` (block-mean) is implemented.
    chunk_ids : List[int], optional
        Process only these chunk ids. ``None`` (default) processes all discovered
        chunks.
    overwrite : bool
        Rebuild arrays even if present.

    Returns
    -------
    Dict[str, Any]
        The manifest (also written to ``out_dir/store_manifest.json``).
    """
    if reduce != "mean":
        raise ValueError(f"reduce={reduce!r} not supported (only 'mean').")
    rates = _validate_rates(rates)
    cols = list(input_cols) + list(dynamic_cols)
    if len(set(cols)) != len(cols):
        raise ValueError("input_cols and dynamic_cols must be disjoint.")

    os.makedirs(out_dir, exist_ok=True)
    files = discover_chunk_files(data_dir, pattern)
    if not files:
        raise ValueError(f"No chunk_*/{pattern} files found under {data_dir!r}.")
    if chunk_ids is not None:
        allowed = set(chunk_ids)
        files = [(cid, p) for cid, p in files if cid in allowed]
        if not files:
            raise ValueError(
                f"chunk_ids={chunk_ids!r} matched none of the discovered chunks.")

    import pyarrow.parquet as pq

    chunks_meta: List[Dict[str, Any]] = []

    for cid, path in files:
        rate_meta: Dict[str, Any] = {}
        # Skip the parquet read entirely if every rate file already exists.
        need = [r for r in rates
                if overwrite or not os.path.exists(_array_path(out_dir, cid, r))]
        if need:
            # Validate against the schema (no data read), then pushdown only the
            # columns we keep — one chunk resident at a time.
            present = set(pq.ParquetFile(path).schema.names)
            missing = [c for c in cols if c not in present]
            if missing:
                raise ValueError(
                    f"chunk_{cid}: output parquet is missing {len(missing)} "
                    f"expected feature column(s), e.g. {missing[:3]}."
                )
            has_time = bool(time_col) and time_col in present
            read_cols = (cols + [time_col]) if has_time else cols
            try:
                df = pd.read_parquet(path, columns=read_cols, engine="pyarrow")
            except Exception as e:
                raise OSError(
                    f"chunk_{cid}: failed to read {path!r} — the parquet is likely "
                    f"truncated or corrupt (generation job killed mid-write, "
                    f"interrupted transfer, or still being written). Regenerate or "
                    f"re-copy it, or exclude it via chunk_ids=. "
                    f"({type(e).__name__}: {e})"
                ) from e
            if has_time:
                df = df.sort_values(time_col).reset_index(drop=True)
            full = df[cols].to_numpy(dtype=np.float32)
            del df
            for r in need:
                np.save(_array_path(out_dir, cid, r), block_mean_decimate(full, r))
            del full

        for r in rates:
            p = _array_path(out_dir, cid, r)
            # Read length cheaply from the memmap header without loading data.
            n_rows = int(np.load(p, mmap_mode="r").shape[0])
            rate_meta[str(r)] = {"file": os.path.basename(p), "n_rows": n_rows}
        chunks_meta.append({"chunk_id": cid, "rates": rate_meta})

    manifest = {
        "data_dir": os.path.abspath(data_dir),
        "reduce": reduce,
        "rates": rates,
        "base_rate": rates[0],
        "n_input": len(input_cols),
        "input_cols": list(input_cols),
        "dynamic_cols": list(dynamic_cols),
        "chunk_ids": [cid for cid, _ in files],
        "chunks": chunks_meta,
    }
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_store_manifest(out_dir: str) -> Dict[str, Any]:
    """Load the store manifest written by :func:`build_chunk_store`."""
    with open(os.path.join(out_dir, MANIFEST_NAME)) as f:
        return json.load(f)


def open_chunk_array(out_dir: str, chunk_id: int, rate: int) -> np.ndarray:
    """Memory-map a single chunk's decimated array at the given rate (read-only)."""
    return np.load(_array_path(out_dir, chunk_id, rate), mmap_mode="r")


# ─────────────────────────────────────────────────────────────────────────────
# Explicit chunk hold-out for val/test
# ─────────────────────────────────────────────────────────────────────────────

def assign_chunk_ids_explicit(
    chunk_ids: List[int],
    val_chunk_ids: Optional[List[int]],
    test_chunk_ids: Optional[List[int]],
) -> Dict[str, List[int]]:
    """
    Split chunk ids into train/val/test using user-named val/test ids.

    Train is the complement. Validates that named ids exist and that val/test are
    disjoint. Either list may be ``None`` (treated as empty).
    """
    val = list(val_chunk_ids or [])
    test = list(test_chunk_ids or [])
    known = set(chunk_ids)
    unknown = [c for c in val + test if c not in known]
    if unknown:
        raise ValueError(
            f"val/test_chunk_ids reference chunks not in the store: {unknown}. "
            f"Available: {sorted(known)}."
        )
    overlap = set(val) & set(test)
    if overlap:
        raise ValueError(f"val and test chunk ids overlap: {sorted(overlap)}.")
    held = set(val) | set(test)
    train = [c for c in chunk_ids if c not in held]
    return {"train": train, "val": val, "test": test}
