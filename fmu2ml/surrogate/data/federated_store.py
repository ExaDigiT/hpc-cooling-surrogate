"""
Multi-rate, memmap-backed dataset and loaders for the federated surrogate.

Reads the per-rate decimated arrays written by :mod:`fmu2ml.surrogate.data.store`
and slices windows on demand, so the full dataset never resides in RAM and each
decoder group can use its own sampling rate and look-back.

Window-origin alignment
------------------------
A "prediction origin" ``o`` is an index on the **native** (rate-1) grid. For a
group/branch sampled at rate ``r`` the corresponding row in its decimated array is
``a = o // r`` (exact because every rate divides ``o``). To make this exact for
*all* rates at once, origins are stepped by ``L = lcm(rates)`` and bounded so every
branch has a full history and every group has a full ``K``-step horizon. No window
crosses a chunk boundary (one store array per chunk).

Two key contracts
------------------
* **single-rate** (``config.multi_rate == False``): emits exactly the legacy
  ``FederatedDataset`` keys (``u_hist``, ``y_hist``, ``y_delta``, ``y_delta_{T,V,
  p,Vs,ps,W}``, ``last_dynamic``, ``future_dynamic`` [+ ``raw_inputs_last`` for
  Phase 6]), so the existing trainer / loss / evaluator run unchanged.
* **multi-rate**: emits per-branch ``u_hist__{b}`` / ``y_hist__{b}`` and per-group
  ``y_delta_{suf}`` / ``last_{suf}`` / ``future_{suf}`` at each group's own rate
  (+ ``raw_inputs_last`` / ``last_dynamic_phys`` on the shared physics grid for
  Phase 6). The multi-rate trainer/eval branch (W6) consumes these.
"""

from typing import Dict, List, Tuple, Optional, Any
from math import gcd

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import store as _store
from .normalizer import FederatedNormalizer


# head name -> (batch-key suffix, column_info index key)
_HEAD_META = {
    'G_T':  ('T',  'temp_indices'),
    'G_V':  ('V',  'flow_indices'),
    'G_p':  ('p',  'pressure_indices'),
    'G_Vs': ('Vs', 'flow_sec_indices'),
    'G_ps': ('ps', 'pressure_sec_indices'),
    'G_W':  ('W',  'power_indices'),
}


def _lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b)


def _lcm_all(values: List[int]) -> int:
    out = 1
    for v in values:
        out = _lcm(out, int(v))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Spec extraction from the config
# ─────────────────────────────────────────────────────────────────────────────

def _branch_specs(config) -> Dict[str, Tuple[int, int]]:
    """branch id -> (rate, history). Heads on a branch share these (validated)."""
    specs: Dict[str, Tuple[int, int]] = {}
    for h in config.head_outputs:
        b = config.encoder_groups[h]
        specs[b] = (config.head_subsample[h], config.head_history_steps[h])
    return specs


def _group_specs(config, column_info) -> Dict[str, Dict[str, Any]]:
    """head -> {rate, K, branch, suffix, col_idx (positions within dynamic_cols)}."""
    specs: Dict[str, Dict[str, Any]] = {}
    for h in config.head_outputs:
        suffix, idx_key = _HEAD_META[h]
        specs[h] = {
            'rate': config.head_subsample[h],
            'K': config.head_prediction_steps[h],
            'branch': config.encoder_groups[h],
            'suffix': suffix,
            'col_idx': np.asarray(column_info[idx_key], dtype=np.int64),
        }
    return specs


# ─────────────────────────────────────────────────────────────────────────────
# Origin grid (per chunk)
# ─────────────────────────────────────────────────────────────────────────────

def origin_grid(
    lens_by_rate: Dict[int, int],
    branch_specs: Dict[str, Tuple[int, int]],
    group_specs: Dict[str, Dict[str, Any]],
    physics: Optional[Tuple[int, int]],
) -> Tuple[int, int, int]:
    """
    Return ``(o_lo, L, n_samples)`` for one chunk.

    ``o`` ranges over ``o_lo, o_lo+L, ...`` (native-grid origins). ``o_lo`` is the
    smallest origin giving every branch a full history; the upper bound keeps every
    group's ``K``-step horizon and the physics grid inside the chunk.
    """
    rates = {r for (r, _) in branch_specs.values()}
    rates |= {g['rate'] for g in group_specs.values()}
    if physics is not None:
        rates.add(physics[0])
    L = _lcm_all(sorted(rates))

    lo = 1  # at least one step so "last" (a-1) exists
    for (r, H) in branch_specs.values():
        lo = max(lo, H * r)               # a = o//r >= H
    if physics is not None:
        lo = max(lo, physics[0])          # a_phys >= 1

    hi = None
    for g in group_specs.values():
        r, K = g['rate'], g['K']
        n = lens_by_rate[r]
        cap = (n - K) * r                 # a + K <= n
        hi = cap if hi is None else min(hi, cap)
    if physics is not None:
        rp, Kp = physics
        cap = (lens_by_rate[rp] - Kp) * rp
        hi = cap if hi is None else min(hi, cap)

    # snap bounds onto the L grid
    o_lo = ((lo + L - 1) // L) * L
    o_hi = (hi // L) * L if hi is not None else -1
    n_samples = (o_hi - o_lo) // L + 1 if (o_hi >= o_lo) else 0
    return o_lo, L, n_samples


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FederatedStoreDataset(Dataset):
    """Memmap-backed, multi-rate windows for one split (a list of chunk ids)."""

    def __init__(
        self,
        store_dir: str,
        manifest: Dict[str, Any],
        chunk_ids: List[int],
        config,
        column_info: Dict[str, Any],
        normalizer: FederatedNormalizer,
        include_raw_inputs: bool = False,
        origins_by_chunk: Optional[Dict[int, np.ndarray]] = None,
    ):
        self.store_dir = store_dir
        self.manifest = manifest
        self.config = config
        self.column_info = column_info
        self.normalizer = normalizer
        self.include_raw_inputs = include_raw_inputs
        self.n_input = int(manifest['n_input'])
        self.input_cols = manifest['input_cols']
        self.dynamic_cols = manifest['dynamic_cols']
        self.multi_rate = bool(config.multi_rate)

        self.branch_specs = _branch_specs(config)
        self.group_specs = _group_specs(config, column_info)
        self.physics = (
            (int(config.physics_rate), int(config.physics_K))
            if include_raw_inputs else None
        )

        # per-chunk row counts by rate, from the manifest
        lens = {c['chunk_id']: {int(r): m['n_rows'] for r, m in c['rates'].items()}
                for c in manifest['chunks']}

        # Flat index over (chunk, origin). Two modes:
        #   * full grid (origins_by_chunk is None): origins are the arithmetic
        #     sequence o_lo, o_lo+L, ... stored compactly as (cid, o_lo, L).
        #   * explicit origins (segment split): the caller supplies the exact
        #     origins per chunk; we store the array and index into it directly.
        self._explicit = origins_by_chunk is not None
        self._chunks: List[Tuple[int, int, int]] = []         # (cid, o_lo, L)
        self._origins: List[np.ndarray] = []                  # used iff _explicit
        self._cum: List[int] = [0]
        for cid in chunk_ids:
            o_lo, L, n = origin_grid(
                lens[cid], self.branch_specs, self.group_specs, self.physics)
            if self._explicit:
                origins = np.asarray(origins_by_chunk.get(cid, []), dtype=np.int64)
                if origins.size == 0:
                    continue
                self._chunks.append((cid, o_lo, L))
                self._origins.append(origins)
                self._cum.append(self._cum[-1] + int(origins.size))
            elif n > 0:
                self._chunks.append((cid, o_lo, L))
                self._cum.append(self._cum[-1] + n)
        self._n = self._cum[-1]
        # lazily-opened memmaps, keyed (chunk_id, rate); reopened per worker
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}
        # touched-page reclaim: after this many __getitem__ calls, advise the OS
        # it may drop the memmap pages we have faulted in, so they stop counting
        # against this process's RSS (the real driver of the 1 Hz OOM-during-train).
        self._reads = 0
        self._reclaim_every = 2048

    def __len__(self) -> int:
        return self._n

    def _arr(self, chunk_id: int, rate: int) -> np.ndarray:
        key = (chunk_id, rate)
        a = self._cache.get(key)
        if a is None:
            a = _store.open_chunk_array(self.store_dir, chunk_id, rate)
            self._cache[key] = a
        return a

    def _reclaim(self) -> None:
        """Drop faulted-in memmap pages so RSS plateaus instead of growing.

        Striding the long 1 Hz arrays faults their pages into the kernel page
        cache, and because the files are mmap'd they count toward our RSS and
        climb across the epoch until the OOM-killer fires. Re-opening the memmaps
        releases our references to the old mappings (``madvise(DONTNEED)`` is
        applied where available), letting the OS reclaim those pages on pressure.
        """
        for key, arr in list(self._cache.items()):
            mm = getattr(arr, '_mmap', None) or getattr(getattr(arr, 'base', None),
                                                         '_mmap', None)
            if mm is not None:
                try:
                    mm.madvise(__import__('mmap').MADV_DONTNEED)
                    continue
                except (AttributeError, OSError, ValueError):
                    pass
            # Fallback: drop the handle; it reopens lazily on next access.
            self._cache.pop(key, None)

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Global index -> (chunk_id, native origin o)."""
        # find chunk via the cumulative counts
        import bisect
        k = bisect.bisect_right(self._cum, idx) - 1
        cid, o_lo, L = self._chunks[k]
        local = idx - self._cum[k]
        if self._explicit:
            o = int(self._origins[k][local])
        else:
            o = o_lo + local * L
        return cid, o

    def _norm_input(self, w: np.ndarray) -> np.ndarray:
        return self.normalizer.input_normalizer.transform(w, self.input_cols)

    def _norm_output(self, w: np.ndarray) -> np.ndarray:
        return self.normalizer.output_normalizer.transform(w, self.dynamic_cols)

    def _consec_delta(self, future: np.ndarray, last: np.ndarray,
                      col_names: List[str], rate: int) -> np.ndarray:
        """(future_k - future_{k-1})/scale_rate, future_{-1}=last. -> (K, n)."""
        prev = np.concatenate([last[None, :], future[:-1]], axis=0)
        consec = future - prev
        delta = self.normalizer._delta_for(rate)
        out = np.empty_like(consec, dtype=np.float32)
        for i, col in enumerate(col_names):
            out[:, i] = consec[:, i] / delta.get_scale(col)
        return out

    def _branch_windows(self, chunk_id: int, o: int) -> Dict[str, torch.Tensor]:
        """u_hist/y_hist per encoder branch (normalized)."""
        out: Dict[str, torch.Tensor] = {}
        nin = self.n_input
        for b, (r, H) in self.branch_specs.items():
            a = o // r
            arr = self._arr(chunk_id, r)
            u = np.array(arr[a - H:a, :nin], dtype=np.float32)
            y = np.array(arr[a - H:a, nin:], dtype=np.float32)
            out[f'u_hist__{b}'] = torch.from_numpy(self._norm_input(u)).float()
            out[f'y_hist__{b}'] = torch.from_numpy(self._norm_output(y)).float()
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        self._reads += 1
        if self._reads % self._reclaim_every == 0:
            self._reclaim()
        chunk_id, o = self._locate(idx)
        nin = self.n_input

        if not self.multi_rate:
            # ---- legacy single-rate key contract ----
            (b, (r, H)), = self.branch_specs.items()
            a = o // r
            arr = self._arr(chunk_id, r)
            u = np.array(arr[a - H:a, :nin], dtype=np.float32)
            y_full = np.array(arr[a - H:a, nin:], dtype=np.float32)
            last_dyn = np.array(arr[a - 1, nin:], dtype=np.float32)
            future_dyn = np.array(arr[a:a + self._k_single(), nin:], dtype=np.float32)
            y_delta = self._consec_delta(future_dyn, last_dyn, self.dynamic_cols, r)
            item = {
                'u_hist': torch.from_numpy(self._norm_input(u)).float(),
                'y_hist': torch.from_numpy(self._norm_output(y_full)).float(),
                'y_delta': torch.from_numpy(y_delta).float(),
                'last_dynamic': torch.from_numpy(last_dyn).float(),
                'future_dynamic': torch.from_numpy(future_dyn).float(),
            }
            for h, g in self.group_specs.items():
                ci = g['col_idx']
                item[f'y_delta_{g["suffix"]}'] = torch.from_numpy(
                    y_delta[:, ci]).float()
            if self.include_raw_inputs:
                item['raw_inputs_last'] = torch.from_numpy(
                    np.array(arr[a - 1, :nin], dtype=np.float32)).float()
            return item

        # ---- multi-rate key contract ----
        item: Dict[str, torch.Tensor] = self._branch_windows(chunk_id, o)
        for h, g in self.group_specs.items():
            r, K, ci, suf = g['rate'], g['K'], g['col_idx'], g['suffix']
            a = o // r
            arr = self._arr(chunk_id, r)
            last = np.array(arr[a - 1, nin + ci], dtype=np.float32)        # (n_g,)
            future = np.array(arr[a:a + K, nin + ci], dtype=np.float32)    # (K, n_g)
            cols = [self.dynamic_cols[j] for j in ci]
            item[f'y_delta_{suf}'] = torch.from_numpy(
                self._consec_delta(future, last, cols, r)).float()
            item[f'last_{suf}'] = torch.from_numpy(last).float()
            item[f'future_{suf}'] = torch.from_numpy(future).float()
        if self.physics is not None:
            rp, _ = self.physics
            ap = o // rp
            arr = self._arr(chunk_id, rp)
            item['raw_inputs_last'] = torch.from_numpy(
                np.array(arr[ap - 1, :nin], dtype=np.float32)).float()
            item['last_dynamic_phys'] = torch.from_numpy(
                np.array(arr[ap - 1, nin:], dtype=np.float32)).float()
        return item

    def _k_single(self) -> int:
        """The single shared K (single-rate path)."""
        return next(iter(self.group_specs.values()))['K']


# ─────────────────────────────────────────────────────────────────────────────
# Split assignment + loader builder
# ─────────────────────────────────────────────────────────────────────────────

def _assign_splits(chunk_ids: List[int], config) -> Dict[str, List[int]]:
    """Explicit hold-out if val/test_chunk_ids set, else seeded chunk shuffle."""
    if config.val_chunk_ids is not None or config.test_chunk_ids is not None:
        return _store.assign_chunk_ids_explicit(
            chunk_ids, config.val_chunk_ids, config.test_chunk_ids)
    rng = np.random.default_rng(config.split_seed)
    order = rng.permutation(len(chunk_ids))
    n = len(chunk_ids)
    n_train = int(round(n * config.train_ratio))
    n_val = int(round(n * config.val_ratio))
    n_train = min(n_train, n - 2) if n >= 3 else n_train
    n_val = max(n_val, 1) if (n - n_train) >= 2 else n_val
    pick = lambda ids: sorted(chunk_ids[i] for i in ids)
    return {
        'train': pick(order[:n_train]),
        'val': pick(order[n_train:n_train + n_val]),
        'test': pick(order[n_train + n_val:]),
    }


def _window_span_origins(config, column_info, L: int) -> int:
    """Number of L-spaced origins a single window spans (history + horizon).

    Used as the guard gap between segments so a train window's history/horizon
    never overlaps a val/test window's. A window at origin ``o`` reads back to
    ``o - H*r`` (deepest branch history) and forward to ``o + K*r`` (longest group
    horizon); the total native span / L (rounded up) origins must separate
    segments to remove temporal leakage.
    """
    branch = _branch_specs(config)
    group = _group_specs(config, column_info)
    back = max((H * r for (r, H) in branch.values()), default=0)
    fwd = max((g['rate'] * g['K'] for g in group.values()), default=0)
    if config.physics_rate is not None:
        fwd = max(fwd, int(config.physics_rate) * int(config.physics_K))
    span_native = back + fwd
    return int(np.ceil(span_native / L)) + 1


def segment_origins(
    manifest: Dict[str, Any],
    chunk_ids: List[int],
    config,
    column_info: Dict[str, Any],
    include_raw_inputs: bool = False,
    test_chunk_ids: Optional[List[int]] = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """Carve chunk origin grids into train/val/test windows.

    Two layouts, chosen by ``test_chunk_ids``:

    * ``test_chunk_ids is None`` — every chunk contributes a contiguous head
      segment to train, a middle to val, a tail to test (sizes from
      ``train_ratio``/``val_ratio``). Val/test share the train distribution.
    * ``test_chunk_ids`` given — those whole chunks become the test set (all their
      origins, never seen in train/val); the remaining chunks are carved into
      **train/val only** (the ratio is renormalized so train+val fill each chunk).
      This gives a true held-out-chunk generalization test while keeping val on the
      train distribution.

    Segments are separated by a guard gap of one full window span so no window
    crosses a boundary (no temporal leakage). Returns
    ``{'train': {cid: origins}, 'val': {...}, 'test': {...}}``.
    """
    branch_specs = _branch_specs(config)
    group_specs = _group_specs(config, column_info)
    physics = ((int(config.physics_rate), int(config.physics_K))
               if include_raw_inputs else None)
    lens = {c['chunk_id']: {int(r): m['n_rows'] for r, m in c['rates'].items()}
            for c in manifest['chunks']}

    held = set(test_chunk_ids or [])
    unknown = held - set(chunk_ids)
    if unknown:
        raise ValueError(f"test_chunk_ids reference chunks not in the store: "
                         f"{sorted(unknown)}. Available: {sorted(chunk_ids)}.")

    out = {'train': {}, 'val': {}, 'test': {}}
    tr, vr = float(config.train_ratio), float(config.val_ratio)
    # When test is a separate held-out chunk, train/val fill the rest: renormalize
    # so the train fraction *within* a non-test chunk is tr / (tr + vr).
    tr_frac = tr / (tr + vr) if held else tr

    for cid in chunk_ids:
        o_lo, L, n = origin_grid(lens[cid], branch_specs, group_specs, physics)
        if n <= 0:
            continue
        all_origins = o_lo + np.arange(n, dtype=np.int64) * L
        gap = _window_span_origins(config, column_info, L)

        if cid in held:
            out['test'][cid] = all_origins        # whole chunk -> test
            continue

        if held:
            # train/val only; tail (test) slice not taken from this chunk
            i_tr = int(round(n * tr_frac))
            train = all_origins[:max(0, i_tr - gap)]
            val = all_origins[i_tr:]
            test = np.array([], dtype=np.int64)
        else:
            i_tr = int(round(n * tr))
            i_va = int(round(n * (tr + vr)))
            train = all_origins[:max(0, i_tr - gap)]
            val = all_origins[i_tr:max(i_tr, i_va - gap)]
            test = all_origins[i_va:]

        if train.size:
            out['train'][cid] = train
        if val.size:
            out['val'][cid] = val
        if test.size:
            out['test'][cid] = test
    return out


def _train_row_caps(
    manifest: Dict[str, Any],
    train_origins: Dict[int, np.ndarray],
    config,
    column_info: Dict[str, Any],
) -> Dict[int, Dict[int, int]]:
    """Per-chunk, per-rate row cap = last row any train window touches.

    A train window at native origin ``o`` reads forward to ``o + K*r`` at rate
    ``r`` (row ``a + K`` where ``a = o // r``). The cap is the max such row over the
    train origins, so the normalizer (fit on ``arr[:cap]``) sees train-segment rows
    only — no val/test leakage in the stats.
    """
    group_specs = _group_specs(config, column_info)
    rates = sorted({g['rate'] for g in group_specs.values()})
    Kmax_by_rate = {r: max(g['K'] for g in group_specs.values() if g['rate'] == r)
                    for r in rates}
    if config.physics_rate is not None:
        rp = int(config.physics_rate)
        Kmax_by_rate[rp] = max(Kmax_by_rate.get(rp, 0), int(config.physics_K))
        rates = sorted(set(rates) | {rp})
    caps: Dict[int, Dict[int, int]] = {}
    for cid, origins in train_origins.items():
        if origins.size == 0:
            continue
        o_max = int(origins.max())
        caps[cid] = {r: o_max // r + Kmax_by_rate[r] for r in rates}
    return caps


def fit_store_normalizer(
    store_dir: str, manifest: Dict[str, Any], train_ids: List[int],
    row_caps_by_chunk: Optional[Dict[int, Dict[int, int]]] = None,
) -> FederatedNormalizer:
    """Fit a FederatedNormalizer from the decimated train arrays (one read each).

    ``row_caps_by_chunk[cid][rate]`` optionally limits each chunk's contribution to
    its first ``cap`` rows at that rate. Used by the segment split so normalization
    stats are fit on train-segment rows only (no val/test leakage), since every
    chunk is a train chunk in that mode.
    """
    rates = manifest['rates']
    base = manifest['base_rate']
    nin = manifest['n_input']
    norm = FederatedNormalizer()

    def _slice(cid: int, r: int) -> np.ndarray:
        a = np.asarray(_store.open_chunk_array(store_dir, cid, r), dtype=np.float32)
        if row_caps_by_chunk is not None:
            cap = row_caps_by_chunk.get(cid, {}).get(r)
            if cap is not None:
                a = a[:cap]
        return a

    dyn_by_rate: Dict[int, np.ndarray] = {}
    for r in rates:
        dyn_by_rate[r] = np.concatenate(
            [_slice(cid, r)[:, nin:] for cid in train_ids], axis=0)
    input_base = np.concatenate(
        [_slice(cid, base)[:, :nin] for cid in train_ids], axis=0)
    norm.fit_multirate(input_base, dyn_by_rate,
                       manifest['input_cols'], manifest['dynamic_cols'])
    return norm


def build_store_for_config(
    data_dir: str,
    out_dir: str,
    config,
    column_info: Dict[str, Any],
    overwrite: bool = False,
    chunk_ids: Optional[List[int]] = None,
) -> str:
    """
    Build (or refresh) the memmap store for a federated config.

    Materializes every rate the config needs (``config.store_rates`` = group rates
    ∪ ``physics_rate``) from the chunk parquet files. Returns ``out_dir``. Idempotent
    (skips existing per-rate arrays unless ``overwrite``).

    Parameters
    ----------
    chunk_ids : List[int], optional
        Process only these chunk ids. ``None`` processes all discovered chunks.
    """
    _store.build_chunk_store(
        data_dir=data_dir, out_dir=out_dir,
        input_cols=column_info['input_cols'],
        dynamic_cols=column_info['dynamic_cols'],
        rates=config.store_rates,
        overwrite=overwrite,
        chunk_ids=chunk_ids,
    )
    return out_dir


def create_dataloaders_federated_store(
    store_dir: str,
    config,
    column_info: Dict[str, Any],
    num_workers: int = 4,
    pin_memory: bool = False,
    include_raw_inputs: bool = False,
    normalizer: Optional[FederatedNormalizer] = None,
    chunk_ids: Optional[List[int]] = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, FederatedNormalizer]:
    """
    Build train/val/test dataloaders from a memmap store.

    Split modes (``config.store_split_mode``):

    * ``"chunk"`` (default): whole chunks go to train/val/test by explicit
      ``val/test_chunk_ids`` (else a seeded shuffle); normalizer fit on train
      chunks only.
    * ``"segment"``: every chunk is carved into contiguous train/val/test
      time-segments (by ``train_ratio``/``val_ratio``) with a guard gap, so val/test
      share the train distribution and no window crosses a boundary. Normalizer is
      fit on the train-segment rows only.

    Parameters
    ----------
    chunk_ids : List[int], optional
        Use only these chunk ids from the store. ``None`` uses all chunks in the
        manifest.
    pin_memory : bool
        Default ``False`` for the memmap store path. With many per-key tensors and
        long 1 Hz histories, page-locking every batch is host RAM the OS cannot
        reclaim and was a contributor to the OOM-during-training; the marginal H2D
        speedup is not worth it here. Set ``True`` only if host RAM is comfortable.
    prefetch_factor : int
        Batches each worker prefetches (only used when ``num_workers > 0``). Lower
        it to bound how many batches are buffered in RAM at once.
    """
    manifest = _store.load_store_manifest(store_dir)
    all_chunk_ids = manifest['chunk_ids']
    if chunk_ids is not None:
        allowed = set(chunk_ids)
        all_chunk_ids = [c for c in all_chunk_ids if c in allowed]
        manifest = {**manifest, 'chunk_ids': all_chunk_ids,
                    'chunks': [ch for ch in manifest['chunks']
                               if ch['chunk_id'] in allowed]}
    segment_mode = getattr(config, 'store_split_mode', 'chunk') == 'segment'

    if segment_mode:
        seg = segment_origins(manifest, all_chunk_ids, config, column_info,
                              include_raw_inputs=include_raw_inputs,
                              test_chunk_ids=config.test_chunk_ids)
        # Each split's chunk list = chunks that have any origin in that segment.
        splits = {k: sorted(seg[k].keys()) for k in ('train', 'val', 'test')}
        origins = seg
        if normalizer is None:
            if not splits['train']:
                raise ValueError("No train segments after split assignment.")
            caps = _train_row_caps(manifest, origins['train'], config, column_info)
            normalizer = fit_store_normalizer(
                store_dir, manifest, splits['train'], row_caps_by_chunk=caps)
    else:
        splits = _assign_splits(all_chunk_ids, config)
        origins = {'train': None, 'val': None, 'test': None}
        if normalizer is None:
            if not splits['train']:
                raise ValueError("No train chunks after split assignment.")
            normalizer = fit_store_normalizer(store_dir, manifest, splits['train'])

    def _loader(split: str, shuffle: bool, drop_last: bool) -> DataLoader:
        ds = FederatedStoreDataset(
            store_dir, manifest, splits[split], config, column_info,
            normalizer, include_raw_inputs=include_raw_inputs,
            origins_by_chunk=origins[split])
        # persistent_workers=False so per-worker page caches are released between
        # epochs; prefetch_factor bounds in-flight batches. Both kwargs are only
        # valid when num_workers > 0.
        kw: Dict[str, Any] = {}
        if num_workers > 0:
            kw['prefetch_factor'] = prefetch_factor
            kw['persistent_workers'] = False
        return DataLoader(ds, batch_size=config.batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory,
                          drop_last=drop_last, **kw)

    train = _loader('train', True, True)
    val = _loader('val', False, False)
    test = _loader('test', False, False)
    mode = 'segment' if segment_mode else 'chunk'
    print(f"Federated store [{mode} split]: "
          f"train={splits['train']} val={splits['val']} test={splits['test']} | "
          f"multi_rate={config.multi_rate} rates={manifest['rates']}")
    print(f"  windows: train={len(train.dataset)} val={len(val.dataset)} "
          f"test={len(test.dataset)}")
    return train, val, test, normalizer


# Phase-agnostic alias: the store loaders serve EVERY phase's multi-rate mode
# (the batch contract is per-branch u_hist__{b}/y_hist__{b} + per-group
# y_delta_{suf}), not just the federated phases they were first written for.
create_dataloaders_multirate = create_dataloaders_federated_store
