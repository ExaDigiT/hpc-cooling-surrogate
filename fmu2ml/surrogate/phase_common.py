"""Shared engine for the phase 1-6 multi-rate baseline runs.

WHY THIS IS A SHARED MODULE AND NOT SIX COPIES
    The entire point of these runs is **cross-phase comparability**: every
    phase must train and be scored on the *identical* task so the numbers can
    sit in one table next to MR-MIONet (run19b). Six self-contained copies of
    this pipeline would drift the moment one is edited, and a drifted task
    silently invalidates the comparison. Putting the task spec, the data
    pipeline, the metrics and the report in one place makes identical
    treatment a structural guarantee rather than a convention.

    Each phase still gets its own notebook (``surrogate_models/phase1_baseline_lstm.ipynb``
    ... ``phase5-grouped-deepmmnet.ipynb``) holding exactly what differs: the
    config class, the phase-specific hyperparameters, and the run notes.

WHAT IS HELD IDENTICAL ACROSS PHASES
    Data (``systematic-720``, 128 chunks), the seeded 80/10/10 chunk split,
    every per-head rate / look-back / horizon, the normalizer,
    batch size, and the evaluation + metric code path. Only the architecture
    and its own hyperparameters vary.

MULTI-RATE
    The task spec below is non-uniform (3 s hydraulics, 30 s thermal/power),
    so ``config.multi_rate`` is True for every phase config and the library
    dispatches each architecture to its multi-rate variant automatically
    (``MultiRateLSTM``, ``MultiRateDeepONet``, ``MultiRateHybridDeepONet``,
    ``MultiRateDomainDeepONet``, and the natively multi-rate federated pair).

USAGE
    from fmu2ml.surrogate.phase_common import (
        PhaseRun, task_kwargs, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, scaled_lr)
    cfg = surrogate.Phase1Config(**task_kwargs(), lstm_hidden_size=256, ...,
                                 learning_rate=scaled_lr(1e-3),
                                 weight_decay=WEIGHT_DECAY,
                                 max_epochs=MAX_EPOCHS, patience=PATIENCE)
    PhaseRun(run_id="phase1_lstm_mr", arch="lstm", phase="1",
             config=cfg, notes="...").execute()

Run the notebooks from surrogate_models/. Artifacts land in ``runs/<run_id>/``
relative to the working directory, so a rerun never overwrites the reported
runs kept in ``surrogate_models/<run_id>/``. Data comes from ``PHASE_DATA_DIR``
(default ``<root>/summit/data/systematic-720``, where ``<root>`` is
``../../../`` from the notebooks). Before launching several phases in
parallel, build the store once with
``python -m fmu2ml.surrogate.phase_common --prewarm``.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")                 # headless: no display, never blocks
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# This module lives in <repo>/fmu2ml/surrogate/. Resolve paths from here rather
# than from the caller's cwd: the phase notebooks, `python -m
# fmu2ml.surrogate.phase_common` and any launcher then all agree on where the
# data is. Expected layout (<root> is ../../../ from the notebooks):
#
#     <root>/sc-conf/<repo>/fmu2ml/surrogate/    this file (the library)
#     <root>/sc-conf/<repo>/surrogate_models/    the phase notebooks
#     <root>/summit/data/systematic-720/         the data (chunk_<id>/ folders)
#
# abspath, not resolve(): keep symlinked Lustre paths as the user sees them, so
# "three levels up" means the same thing here as it does in a shell.
REPO_ROOT = Path(os.path.abspath(__file__)).parents[2]
WORKSPACE_ROOT = REPO_ROOT.parents[1]


# ─────────────────────────────────────────────────────────────────────────────
# The shared task: identical for every phase AND matched to run19b (MR-MIONet)
# ─────────────────────────────────────────────────────────────────────────────

# Data location and chunk layout are environment-driven so the phases can be
# pointed at the 720 h dataset without editing six scripts. Defaults match
# run_21b.py exactly (systematic-720, 128 chunks, seeded 80/10/10 ratio split)
# so the phase table and MR-MIONet are scored on the SAME task and the same
# held-out chunks — that comparability is the whole point of these baselines.
# Default: <root>/summit/data/systematic-720 (see the layout above), the data
# the reported runs used. Override with PHASE_DATA_DIR.
DATA_DIR = os.environ.get(
    "PHASE_DATA_DIR",
    str(WORKSPACE_ROOT / "summit" / "data" / "systematic-720"))
STORE_DIR = os.environ.get("PHASE_STORE_DIR", os.path.join(DATA_DIR, "store"))

N_CHUNKS = int(os.environ.get("PHASE_N_CHUNKS", 128))
CHUNK_IDS = list(range(N_CHUNKS))

# Explicit hold-out (legacy 32-chunk layout) vs seeded ratio split. With
# PHASE_SPLIT=ratio (default) val/test_chunk_ids are None, which makes
# _assign_splits fall through to train_ratio/val_ratio/split_seed — identical
# to run_21b.py.
_SPLIT = os.environ.get("PHASE_SPLIT", "ratio")
if _SPLIT == "ratio":
    VAL_CHUNK_IDS = None
    TEST_CHUNK_IDS = None
else:                                    # legacy systematic-new layout
    VAL_CHUNK_IDS = [2, 12]
    TEST_CHUNK_IDS = [9, 18]

TRAIN_RATIO = float(os.environ.get("PHASE_TRAIN_RATIO", 0.8))
VAL_RATIO = float(os.environ.get("PHASE_VAL_RATIO", 0.1))
SPLIT_SEED = int(os.environ.get("PHASE_SPLIT_SEED", 42))

# Loader workers PER PHASE PROCESS. When several phases run concurrently on one
# node this must be cores/n_phases, or the processes oversubscribe the CPUs and
# thrash Lustre. The launcher sets it; 0 (the library default) means no
# prefetch at all and leaves the GPU starved, so never leave it unset.
NUM_WORKERS = int(os.environ.get("PHASE_NUM_WORKERS", 8))

# ─────────────────────────────────────────────────────────────────────────────
# Distributed rank, from SLURM. SurrogateTrainer._setup_distributed() keys off
# exactly these variables (SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS), so a
# phase launched with `srun -n8` gets DDP for free — no torchrun involved.
# WORLD_SIZE == 1 (plain `python run_phaseN.py`) leaves everything single-process.
# ─────────────────────────────────────────────────────────────────────────────
RANK = int(os.environ.get("SLURM_PROCID", 0))
LOCAL_RANK = int(os.environ.get("SLURM_LOCALID", 0))
WORLD_SIZE = int(os.environ.get("SLURM_NTASKS", 1))
IS_MAIN = (RANK == 0)
IS_DIST = WORLD_SIZE > 1

# Bridge SLURM's names to torch's names.
#
# The trainer calls dist.init_process_group(backend="nccl") with no rank or
# world_size argument. That routes to the "env://" rendezvous handler, which
# reads RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT — torchrun's variables,
# which srun does not set. Without this bridge every rank dies with
#   ValueError: environment variable RANK expected, but not set
# even though SLURM_PROCID is right there.
#
# Set before the process group is created, and never clobber values already
# present so a torchrun launch still wins.
if IS_DIST:
    os.environ.setdefault("RANK", str(RANK))
    os.environ.setdefault("WORLD_SIZE", str(WORLD_SIZE))
    os.environ.setdefault("LOCAL_RANK", str(LOCAL_RANK))
    # Frontier compute nodes have no IPv6: a hostname resolves to
    # localhost.localdomain and the TCPStore aborts with
    # "errno 97 - Address family not supported by protocol". Single-node
    # rendezvous over loopback avoids it; multi-node must export MASTER_ADDR
    # as a literal IPv4 in the launcher.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

# ─────────────────────────────────────────────────────────────────────────────
# MIOpen kernel cache (ROCm builds only). On AMD GPUs the LSTMs run through
# MIOpen, which compiles kernels on first use and caches them, together with
# its tuning database, under $HOME (~/.cache/miopen, ~/.config/miopen). On
# Frontier /ccs/home is read-only from the compute nodes, and a cache shared
# by several ranks on a network file system is fragile; either way the first
# LSTM forward dies with
#   RuntimeError: miopenStatusInternalError
# Default both to the node-local temp directory ($TMPDIR, else /tmp), one
# directory per job and rank. Locations the launcher already exported win.
# MIOpen reads these lazily, at first use, so setting them here (before any
# training) is early enough.
# If MIOpen still fails, PHASE_DISABLE_MIOPEN=1 runs the LSTMs with PyTorch's
# own GPU kernels instead (see PhaseRun._execute_inner): slower, but
# independent of MIOpen.
# ─────────────────────────────────────────────────────────────────────────────
import tempfile  # noqa: E402
import torch as _torch  # noqa: E402  (already imported by fmu2ml.surrogate)

if getattr(_torch.version, "hip", None):
    _miopen_default = os.path.join(
        tempfile.gettempdir(), f"miopen-{os.environ.get('USER', 'user')}-"
                               f"{os.environ.get('SLURM_JOB_ID', 'nojob')}-{RANK}")
    for _var in ("MIOPEN_USER_DB_PATH", "MIOPEN_CUSTOM_CACHE_DIR"):
        if not os.environ.get(_var):          # never touch a location set by hand
            os.environ[_var] = _miopen_default
            os.makedirs(_miopen_default, exist_ok=True)

SUBSAMPLE = 30          # default rate (s/step); G_T / G_W inherit it
HISTORY = 40
PREDICT_K = 1           # overridden per head below

# ─────────────────────────────────────────────────────────────────────────────
# Training budget — held identical across phases, and invariant to WORLD_SIZE
#
# `config.batch_size` is PER RANK. Setting it to a constant under `srun -n8`
# silently multiplies the global batch by 8 and divides the optimizer-step count
# by 8: the phase 1-6 runs of 2026-07-28 took 30-31 steps/epoch (global batch
# 2048) while run21 took 245 (global batch 256, world=1) — a 20-180x gap in
# gradient steps, in the direction that flatters the model being promoted. That
# is exactly the quantity the note below says is scarce, so it must not depend on
# how many ranks the launcher happens to use.
#
# Pin the GLOBAL batch instead and derive the per-rank batch from it. 256 global
# is the validated point: ~245 optimizer steps/epoch on the 720 h data, matching
# run21/run19b. Larger batches trade steps away, and step count is what the
# gated/residual heads are short of (run21's 61-step collapse).
GLOBAL_BATCH = int(os.environ.get("PHASE_GLOBAL_BATCH", 256))
if GLOBAL_BATCH % WORLD_SIZE:
    raise ValueError(
        f"PHASE_GLOBAL_BATCH={GLOBAL_BATCH} is not divisible by WORLD_SIZE="
        f"{WORLD_SIZE}; ranks would carry unequal batches and the step count "
        f"would stop being comparable across phases.")
PER_RANK_BATCH = GLOBAL_BATCH // WORLD_SIZE
if PER_RANK_BATCH < 1:
    raise ValueError(
        f"PHASE_GLOBAL_BATCH={GLOBAL_BATCH} over {WORLD_SIZE} ranks leaves "
        f"<1 sample per rank; lower -n or raise PHASE_GLOBAL_BATCH.")

# Square-root LR scaling against the reference global batch, the same rule
# run_21b.py uses (`LR_SCALE = (GLOBAL_BATCH / REF_BATCH) ** 0.5`). At the
# default GLOBAL_BATCH=256 this is exactly 1.0, so each phase keeps its OWN
# tuned base LR: the budget is equalized, the per-phase learning rate is not.
REF_BATCH = 256
LR_SCALE = (GLOBAL_BATCH / REF_BATCH) ** 0.5


def scaled_lr(base_lr: float) -> float:
    """Base LR adjusted for the global batch. Identity at GLOBAL_BATCH=256."""
    return base_lr * LR_SCALE


# Epoch budget and early-stopping patience, identical for every phase. Phases
# 5/6 drive their curricula from phase1_epochs/phase2_epochs(/phase3_epochs)
# instead of MAX_EPOCHS, but share PATIENCE.
MAX_EPOCHS = int(os.environ.get("PHASE_MAX_EPOCHS", 100))
PATIENCE = int(os.environ.get("PHASE_PATIENCE", 20))

# Shared L2 regularisation. Phases 1-4 used 1e-5 and phases 5/6 used 1e-4, a
# 10x difference that confounded every cross-family comparison in the table:
# P5/P6 are also ~10x smaller (5.9M vs 55.6M), so "smaller model, heavier decay"
# was two changes at once. One value for all six; override per phase only with
# a stated reason.
WEIGHT_DECAY = float(os.environ.get("PHASE_WEIGHT_DECAY", 1e-5))

# Training-time perturbation of the output history, off by default. Training is
# teacher-forced (y_hist is ground truth) but deployment is closed-loop (y_hist
# is the model's own output), so the model never learns to tolerate a history it
# should not fully trust. Enable with PHASE_HISTORY_NOISE=0.01 (normalised
# units) to trade a little teacher-forced accuracy for rollout robustness.
# Sweep it -- the right value is a property of the data, not a constant.
HISTORY_NOISE_STD = float(os.environ.get("PHASE_HISTORY_NOISE", 0.0))
HISTORY_NOISE_WARMUP = int(os.environ.get("PHASE_HISTORY_NOISE_WARMUP", 5))

ALL_HEAD_NAMES = ["G_T", "G_V", "G_p", "G_Vs", "G_ps", "G_W"]
GROUP_COLORS = {"G_T": "steelblue", "G_V": "coral", "G_p": "green",
                "G_Vs": "mediumpurple", "G_ps": "goldenrod", "G_W": "teal"}
GROUP_UNITS = {"G_T": "°C", "G_V": "GPM", "G_Vs": "GPM",
               "G_p": "psig", "G_ps": "psig", "G_W": "kW"}
SUF = {"G_T": "T", "G_V": "V", "G_p": "p",
       "G_Vs": "Vs", "G_ps": "ps", "G_W": "W"}

INPUT_PATTERNS = {
    "Q_flow": "simulator_1_datacenter_1_computeBlock_{}_cabinet_1_sources_Q_flow_total",
    "T_Air": "simulator_1_datacenter_1_computeBlock_{}_cabinet_1_sources_T_Air",
    "T_ext": "simulator_1_centralEnergyPlant_1_coolingTowerLoop_1_sources_T_ext",
}
_S = "simulator[1].datacenter[1].computeBlock[{}].cdu[1].summary."
OUTPUT_PATTERNS = {
    "T_prim_s": _S + "T_prim_s_C",   "T_prim_r": _S + "T_prim_r_C",
    "T_sec_s": _S + "T_sec_s_C",     "T_sec_r": _S + "T_sec_r_C",
    "V_flow_prim": _S + "V_flow_prim_GPM",
    "V_flow_sec": _S + "V_flow_sec_GPM",
    "p_prim_s": _S + "p_prim_s_psig", "p_prim_r": _S + "p_prim_r_psig",
    "p_sec_s": _S + "p_sec_s_psig",   "p_sec_r": _S + "p_sec_r_psig",
    "W_flow": _S + "W_flow_CDUP_kW",
}


def first_chunk_file() -> str:
    """Path of the first chunk parquet under DATA_DIR; its schema drives column_info.

    Fails with the searched path and what IS there, instead of the bare
    IndexError an empty discover_chunk_files() list would give.
    """
    from fmu2ml.surrogate.data import store as _store

    found = _store.discover_chunk_files(DATA_DIR)
    if found:
        return found[0][1]
    data_root = WORKSPACE_ROOT / "summit" / "data"
    candidates = (sorted(p.name for p in data_root.iterdir()
                         if p.is_dir() and any(p.glob("chunk_*")))
                  if data_root.is_dir() else [])
    raise FileNotFoundError(
        f"No chunk_<id>/{_store.CHUNK_GLOB} files under DATA_DIR={DATA_DIR} "
        f"({'exists' if os.path.isdir(DATA_DIR) else 'does not exist'}).\n"
        f"  workspace root (../../../ from the notebooks): {WORKSPACE_ROOT}\n"
        f"  datasets found in {data_root}: {candidates or 'none'}\n"
        "Set PHASE_DATA_DIR to the dataset directory (the one holding the "
        "chunk_<id>/ folders) before importing phase_common.")


def ensure_store(cfg, column_info, timeout_s: int = 7200) -> None:
    """Materialize the on-disk store EXACTLY ONCE, even with N phases running.

    Why this exists: every phase used to call ``build_store_for_config``
    unconditionally. Launch five phases in parallel against one STORE_DIR and
    they all write the same memmap files at the same time — silent corruption,
    or a torn manifest that later readers mis-parse.

    Protocol (no external deps, works on Lustre):
      * if the manifest is already there, return immediately — the common case
        once the launcher has pre-warmed it;
      * otherwise try to create a lock file with O_CREAT|O_EXCL. Exactly one
        process wins and builds; it writes the manifest, then removes the lock;
      * losers poll for the manifest rather than building. A stale lock older
        than ``timeout_s`` is reclaimed so a killed builder cannot wedge the
        node forever.

    Call ``python -m fmu2ml.surrogate.phase_common --prewarm`` to do the build
    serially up front, which is what run_phases_parallel.sh does.
    """
    import fmu2ml.surrogate as surrogate

    from fmu2ml.surrogate.data.store import MANIFEST_NAME
    manifest = os.path.join(STORE_DIR, MANIFEST_NAME)   # store_manifest.json
    if os.path.exists(manifest):
        print(f"[store] present, reusing: {STORE_DIR}")
        return

    os.makedirs(STORE_DIR, exist_ok=True)
    lock = os.path.join(STORE_DIR, ".build.lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
        os.close(fd)
        owner = True
    except FileExistsError:
        owner = False

    if owner:
        print(f"[store] building (this process owns the lock): {STORE_DIR}")
        try:
            surrogate.build_store_for_config(DATA_DIR, STORE_DIR, cfg,
                                             column_info, chunk_ids=CHUNK_IDS)
        finally:
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
        print("[store] build complete")
        return

    print("[store] another process is building; waiting...")
    waited = 0.0
    stale_s = float(os.environ.get("PHASE_STALE_LOCK_S", 900))
    while not os.path.exists(manifest):
        # STALE LOCK RECLAMATION. If the builder died (it exited before writing
        # the manifest, e.g. a phase that failed on GPU allocation), the lock
        # file survives and every other process would wait here forever. If the
        # lock stops being refreshed for stale_s AND no manifest appeared,
        # assume the builder is gone, take the lock over, and build.
        try:
            age = time.time() - os.path.getmtime(lock)
        except FileNotFoundError:
            age = None                      # builder finished or cleaned up
        if age is not None and age > stale_s:
            print(f"[store] lock is {age:.0f}s stale (> {stale_s:.0f}s) and no "
                  f"manifest — assuming the builder died; taking over.")
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
            return ensure_store(cfg, column_info, timeout_s)   # retry as owner
        time.sleep(5.0)
        waited += 5.0
        if waited > timeout_s:
            raise TimeoutError(
                f"waited {timeout_s}s for {manifest}. If no builder is alive, "
                f"delete {lock} and rerun (or pre-warm with "
                f"`python -m fmu2ml.surrogate.phase_common --prewarm`).")
        if waited % 60 < 5:
            print(f"[store] still waiting ({waited:.0f}s)...")
    print(f"[store] ready after {waited:.0f}s")


NORMALIZER_PATH = os.path.join(STORE_DIR, "normalizer_shared.json")


def ensure_normalizer(cfg, column_info, timeout_s: int = 7200):
    """Fit the normalizer ONCE and share it across phases.

    THIS IS THE EXPENSIVE STEP, and the reason five parallel phases appeared to
    hang. create_dataloaders_federated_store() calls fit_store_normalizer()
    whenever `normalizer=None`, and that function does:

        np.concatenate([chunk[:, nin:] for cid in train_ids])

    i.e. it materializes EVERY train chunk into one array. On the 720 h / 128
    chunk data (102 train chunks) that is ~7.3 GB at rate 3 s, ~0.7 GB at 30 s
    and ~1.3 GB of inputs, and np.concatenate needs roughly double that while
    copying. Each phase did it independently: five processes x ~10-20 GB, all
    streaming the whole dataset off Lustre at once. Not a deadlock — just an
    enormous amount of duplicated I/O and memory.

    Fitting once and passing the result to every phase removes 5x the work AND
    guarantees the phases share bit-identical normalization, which the
    cross-phase comparison depends on anyway.
    """
    from fmu2ml.surrogate.data.normalizer import FederatedNormalizer
    from fmu2ml.surrogate.data import federated_store as _fs
    from fmu2ml.surrogate.data import store as _store

    if os.path.exists(NORMALIZER_PATH):
        print(f"[norm] loading shared normalizer: {NORMALIZER_PATH}")
        return FederatedNormalizer.load(NORMALIZER_PATH)

    lock = NORMALIZER_PATH + ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {time.time()}\n".encode())
        os.close(fd)
        owner = True
    except FileExistsError:
        owner = False

    if owner:
        print("[norm] fitting shared normalizer (one full pass over the train "
              "chunks; this is the slow step)...")
        t0 = time.time()
        try:
            manifest = _store.load_store_manifest(STORE_DIR)
            allowed = set(CHUNK_IDS)
            manifest = {**manifest,
                        "chunk_ids": [c for c in manifest["chunk_ids"]
                                      if c in allowed],
                        "chunks": [c for c in manifest["chunks"]
                                   if c["chunk_id"] in allowed]}
            splits = _fs._assign_splits(manifest["chunk_ids"], cfg)
            norm = _fs.fit_store_normalizer(STORE_DIR, manifest,
                                            splits["train"])
            norm.save(NORMALIZER_PATH)
        finally:
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
        print(f"[norm] fitted and saved in {time.time() - t0:.0f}s -> "
              f"{NORMALIZER_PATH}")
        return norm

    print("[norm] another process is fitting; waiting...")
    waited = 0.0
    stale_s = float(os.environ.get("PHASE_STALE_LOCK_S", 1800))
    while not os.path.exists(NORMALIZER_PATH):
        try:
            age = time.time() - os.path.getmtime(lock)
        except FileNotFoundError:
            age = None
        if age is not None and age > stale_s:
            print(f"[norm] lock {age:.0f}s stale — taking over")
            try:
                os.remove(lock)
            except FileNotFoundError:
                pass
            return ensure_normalizer(cfg, column_info, timeout_s)
        time.sleep(5.0); waited += 5.0
        if waited > timeout_s:
            raise TimeoutError(f"waited {timeout_s}s for {NORMALIZER_PATH}; "
                               f"delete {lock} and retry")
        if waited % 60 < 5:
            print(f"[norm] still waiting ({waited:.0f}s)...")
    print(f"[norm] ready after {waited:.0f}s")
    return FederatedNormalizer.load(NORMALIZER_PATH)


def task_kwargs(**overrides: Any) -> Dict[str, Any]:
    """The multi-rate task spec, identical for every phase.

    Rates are matched to run19b/run21: hydraulics at 3 s (PSD-justified — 30 s
    aliases them, and run18 proved 1 s is *worse* because rate 1 has no
    anti-alias block-mean), thermal and power at 30 s.

    Look-backs and horizons are matched to run21 as well, so ALL SEVEN models
    (phases 1-6 and MR-MIONet) predict the same windows:

        group        rate    look-back          horizon
        G_T          30 s    40 x 30 s = 1200 s  20 x 30 s = 600 s
        G_W          30 s    40 x 30 s = 1200 s  20 x 30 s = 600 s
        G_V, G_p      3 s   100 x  3 s =  300 s  20 x  3 s =  60 s
        G_Vs, G_ps    3 s   100 x  3 s =  300 s  20 x  3 s =  60 s

    Until 2026-07-29 this default read K = {G_T 10, G_V 10, G_p 10, G_Vs 100,
    G_ps 100, G_W 10} with a 50-step hydraulic look-back, while run_phase5.py /
    run_phase6.py / run_21.py overrode it to the uniform K = 20 above. Phases
    1-4 therefore scored on 300 s secondary horizons and phases 5-7 on 60 s,
    which is the entire 0.807 -> 0.925 mean-R2 "jump" between P4 and P5 — a task
    difference read as an architecture result. It also split the evaluation sets
    (63,954/8,151/8,151 windows vs 62,934/8,021/8,021), since the longer 600 s
    G_T/G_W horizon needs more room at each chunk tail.

    ``**overrides`` lets a phase adjust a field it genuinely needs; anything
    overridden here breaks comparability, so do it deliberately.
    """
    spec = dict(
        system_name="summit",
        num_cdus=257,
        cdu_ids=list(range(1, 258)),

        history_steps=HISTORY,
        prediction_steps=PREDICT_K,
        subsample_factor=SUBSAMPLE,

        # per-group rate (s/step); omitted -> subsample_factor (30 s)
        head_subsample={"G_V": 3, "G_p": 3, "G_Vs": 3, "G_ps": 3},
        # look-backs: G_T/G_W 40*30 = 1200 s ; hydraulics 100*3 = 300 s
        head_history_steps={"G_T": 40, "G_V": 100, "G_p": 100, "G_Vs": 100,
                            "G_ps": 100, "G_W": 40},
        # horizons: uniform K = 20 -> G_T/G_W 600 s, hydraulics 60 s.
        # Identical for every phase; see the docstring for why this changed.
        head_prediction_steps={"G_T": 20, "G_V": 20, "G_p": 20,
                               "G_Vs": 20, "G_ps": 20, "G_W": 20},
        encoder_groups={"G_T": "slow", "G_W": "slow",
                        "G_V": "fast", "G_p": "fast",
                        "G_Vs": "fast", "G_ps": "fast"},

        val_chunk_ids=VAL_CHUNK_IDS,
        test_chunk_ids=TEST_CHUNK_IDS,
        store_split_mode="chunk",
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,

        # PER-RANK batch, derived from the pinned global batch so the optimizer
        # step count is the same whether this runs on 1 rank or 8. See the
        # GLOBAL_BATCH block near the top of this module.
        batch_size=PER_RANK_BATCH,
        num_workers=NUM_WORKERS,
        gradient_clip=1.0,

        # ── per-head loss balance ────────────────────────────────────────────
        # Shared by every phase, for the same reason the task spec is: the
        # objective has to be identical or the comparison is meaningless.
        #
        # Measured on the 2026-07-30 runs, the UNIFORM-weight validation loss
        # decomposed as G_Vs 52.2%, G_ps 46.6%, and everything else 1.2%
        # combined (G_T 0.39, G_V 0.62, G_p 0.02, G_W 0.17). G_Vs is a measured
        # channel noise floor, so ~half the gradient was being spent on a
        # quantity no model can fit, and the four heads with real headroom were
        # sharing a hundredth of the budget. Every architecture hit its final
        # loss within ONE epoch and then moved only in the 6th decimal for the
        # remaining 20-36 — which is why all seven models scored identically.
        #
        # The 2026-08-02 runs then showed that G_Vs is not the only dead head.
        # Measured per-lead skill, every architecture, every lead:
        #
        #   G_V   persistence R2 = 1.0000 at short leads; skill went from ~0 to
        #         -0.17 (P3/P4) once normalisation gave it 20% of the budget --
        #         FLAT across all leads, i.e. an injected constant bias, not
        #         compounding error. There is nothing there to win.
        #   G_ps  skill <= +0.0014 at EVERY lead for ALL SIX models, under both
        #         the uniform and the normalised objective. Persistence is
        #         strong (R2 0.9986 -> 0.8896 over the horizon) and the residual
        #         after it is unpredictable.
        #
        # So only G_T, G_W and G_p carry exploitable structure (1,799 of the
        # 2,827 outputs). The other three (1,028 outputs) are persistence-locked
        # and are now declared as such: weight 0 here, and the models emit exact
        # zero deltas for them (see SurrogateConfig.persistence_heads), which
        # reconstructs to exact persistence.
        #
        # Weight 0 alone is NOT enough and was a real bug: an excluded head
        # still runs and still emits its random initialisation. Phase 1's
        # untrained G_Vs head hit R2 = -1.46 teacher-forced and -2274 in rollout
        # on 2026-08-02, while phases 2-4 happened to initialise near zero and
        # looked fine. That is why the zeroing is structural, in the model.
        head_loss_weights={"G_T": 1.0, "G_V": 0.0, "G_p": 1.0,
                           "G_Vs": 0.0, "G_ps": 0.0, "G_W": 1.0},
        # Divide each head by its persistence-baseline loss, so the objective
        # becomes "fraction of persistence error remaining" — the same quantity
        # the reported skill score measures.
        head_loss_normalize=True,

        # Off unless PHASE_HISTORY_NOISE is set; see the constant above.
        history_noise_std=HISTORY_NOISE_STD,
        history_noise_warmup_epochs=HISTORY_NOISE_WARMUP,

        input_patterns=INPUT_PATTERNS,
        output_patterns=OUTPUT_PATTERNS,
    )
    spec.update(overrides)
    return spec


# ─────────────────────────────────────────────────────────────────────────────
# Script harness — headless figures, streamed log, markdown report
# ─────────────────────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """Write to several streams at once, flushing each time.

    Used to capture EVERYTHING printed during a run — including the library's
    own training output, which a module-level print shadow would miss — into
    both the console and a line-flushed run_log.txt.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            try:
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


class PhaseRun:
    """One phase's end-to-end run: data -> train -> evaluate -> report.

    Parameters
    ----------
    run_id : str
        Unique id; artifacts land in ``runs/<run_id>/``. Never reuse one.
    arch : str
        Registry name: lstm | deeponet | hybrid_deeponet | domain_deeponet |
        federated | federated_pi.
    phase : str
        Phase label "1".."6", passed to the evaluator for its dispatch.
    config : Any
        A fully-built phase config (use ``task_kwargs()`` for the shared spec).
    notes : str
        Free text recorded in the report; say *why* this run exists.
    arch_kwargs : dict, optional
        Extra kwargs for the architecture factory.
    """

    def __init__(self, run_id: str, arch: str, phase: str, config: Any,
                 notes: str = "", arch_kwargs: Optional[Dict] = None):
        self.run_id = run_id
        self.arch = arch
        self.phase = phase
        self.config = config
        self.notes = notes
        self.arch_kwargs = arch_kwargs or {}
        self.run_dir = Path("runs") / run_id
        self.figures: List[tuple] = []
        self._log_file = None
        self._buffer = io.StringIO()
        self._stdout = None

    # ── harness ─────────────────────────────────────────────────────────────
    def _start_log(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if not IS_MAIN:
            self._log_file = None      # non-main ranks never touch the log file
            return
        self._log_file = open(self.run_dir / "run_log.txt", "w",
                              encoding="utf-8", buffering=1)
        self._stdout = sys.stdout
        sys.stdout = _Tee(self._stdout, self._log_file, self._buffer)

    def _stop_log(self):
        if self._stdout is not None:
            sys.stdout = self._stdout
        if self._log_file is not None:
            self._log_file.flush()
            self._log_file.close()
            self._log_file = None

    def save_fig(self, fig, filename: str, caption: str):
        fig.savefig(self.run_dir / filename, dpi=120, bbox_inches="tight")
        self.figures.append((caption, filename))
        plt.close(fig)

    # ── pipeline ────────────────────────────────────────────────────────────
    def execute(self):
        """Run the whole pipeline. Returns the metrics DataFrame."""
        self._start_log()
        try:
            return self._execute_inner()
        finally:
            self._stop_log()

    def _execute_inner(self):
        import torch
        import pyarrow.parquet as pq
        import fmu2ml.surrogate as surrogate
        from fmu2ml.surrogate.data import store as _store

        t_start = time.time()
        np.random.seed(42)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # One process per GPU: the launcher pins each phase with
        # HIP_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES, so "cuda" is that phase's
        # own device and no two phases share a GCD.
        # One GPU per rank. The trainer wraps the model with
        # DistributedDataParallel(device_ids=[SLURM_LOCALID]), so every rank
        # must place its model on cuda:LOCAL_RANK — not the default cuda:0, or
        # all eight ranks would pile onto GPU 0.
        _vis = {k: os.environ.get(k) for k in
                ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES",
                 "CUDA_VISIBLE_DEVICES")}
        _n_dev = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # ROCR_VISIBLE_DEVICES and HIP_/CUDA_VISIBLE_DEVICES are NOT aliases on
        # ROCm — they compose. ROCr filters the physical GCDs first; HIP then
        # filters whatever ROCr left. Slurm's gres plugin sets all three to the
        # same physical index, so on rank 4 the pair means "expose only GCD 4"
        # (it becomes index 0) and then "select index 4 of that 1-device list"
        # -> zero devices. Only rank 0 survives. Diagnose it by name, because
        # the symptom (cuda_available=False) points nowhere near the cause.
        _rocr, _hip = _vis["ROCR_VISIBLE_DEVICES"], _vis["HIP_VISIBLE_DEVICES"]
        _double = (_rocr and (_hip or _vis["CUDA_VISIBLE_DEVICES"])
                   and _rocr.strip() != "" )
        if _n_dev == 0 and _double:
            raise RuntimeError(
                "No GPU visible, and the visible-device env is double-filtered.\n"
                f"  visible-device env: {_vis}\n"
                f"  torch.version.hip : {getattr(torch.version, 'hip', None)}\n"
                "ROCR_VISIBLE_DEVICES and HIP_/CUDA_VISIBLE_DEVICES COMPOSE on "
                "ROCm, they do not alias: ROCr filters first, then HIP filters "
                "the result. Setting both to the same index N>0 always yields "
                "zero devices.\n"
                "Fix: set exactly ONE of them. In an srun launcher, clear the "
                "HIP/CUDA layer and restate the full list at the ROCr layer:\n"
                "  unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES\n"
                "  export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\n"
                "Set PHASE_ALLOW_CPU=1 to override (trains on CPU, ~100x slower).")

        if torch.cuda.is_available():
            # A rank whose local index exceeds what it can see would otherwise
            # fail deep inside DDP with an opaque device-ordinal error.
            if LOCAL_RANK >= _n_dev:
                raise RuntimeError(
                    f"rank {RANK}: local_rank {LOCAL_RANK} needs cuda:{LOCAL_RANK} "
                    f"but only {_n_dev} device(s) are visible.\n"
                    f"  visible-device env: {_vis}\n"
                    "Every rank must see ALL GCDs, because the trainer wraps the "
                    "model as DDP(device_ids=[SLURM_LOCALID]). Launch with "
                    "--gpus-per-node=8 --gpu-bind=none (NOT --gpus-per-task=1) "
                    "and clear any per-task HIP_/CUDA_VISIBLE_DEVICES.")
            torch.cuda.set_device(LOCAL_RANK)
            device = torch.device(f"cuda:{LOCAL_RANK}")
        else:
            device = torch.device("cpu")
        if IS_DIST:
            print(f"[ddp] rank {RANK}/{WORLD_SIZE} local_rank {LOCAL_RANK} "
                  f"device {device}  visible={_n_dev}")
        if device.type == "cpu" and os.environ.get("PHASE_ALLOW_CPU") != "1":
            # exadigit_env ships a CUDA-built torch; on MI250X that silently
            # trains on CPU ~100x slower. Refuse rather than waste the run.
            raise RuntimeError(
                "CUDA/ROCm not available — refusing to train on CPU.\n"
                f"  visible-device env: {_vis}\n"
                f"  torch.version.hip : {getattr(torch.version, 'hip', None)}\n"
                "Most common cause when launching several phases at once: the "
                "phase was pinned to a GPU INDEX THAT DOES NOT EXIST in this "
                "allocation (e.g. ROCR_VISIBLE_DEVICES=2 with only 2 GCDs "
                "allocated), so torch sees zero devices. Check "
                "`torch.cuda.device_count()` and request at least as many GPUs "
                "as phases, or run fewer phases.\n"
                "Other cause: a CUDA-built torch (hip=None) — use exadigit_rocm.\n"
                "Set PHASE_ALLOW_CPU=1 to override.")

        # Opt-in escape hatch for a broken MIOpen (see the MIOpen block at the
        # top of this module): the LSTMs then run on PyTorch's own GPU kernels.
        if os.environ.get("PHASE_DISABLE_MIOPEN") == "1":
            torch.backends.cudnn.enabled = False

        cfg = self.config
        print("=" * 74)
        print(f"PHASE {self.phase} — {self.arch}   [{self.run_id}]")
        print("=" * 74)
        print(f"device: {device}")
        if getattr(torch.version, "hip", None):
            print("MIOpen: " + (
                "disabled (PHASE_DISABLE_MIOPEN=1), PyTorch LSTM kernels"
                if not torch.backends.cudnn.enabled else
                f"cache {os.environ.get('MIOPEN_USER_DB_PATH')}"))
        print(f"multi_rate: {cfg.multi_rate}   store rates (s): {cfg.store_rates}")
        print(f"{'group':6s} {'rate(s)':>8s} {'history':>8s} {'K':>4s} "
              f"{'window(s)':>10s} {'horizon(s)':>11s}  branch")
        for g in ALL_HEAD_NAMES:
            r = cfg.head_subsample[g]; h = cfg.head_history_steps[g]
            k = cfg.head_prediction_steps[g]; b = cfg.encoder_groups[g]
            print(f"{g:6s} {r:>8d} {h:>8d} {k:>4d} {h*r:>10d} {k*r:>11d}  {b}")
        if not cfg.multi_rate:
            raise RuntimeError(
                "config.multi_rate is False — the phase would train on the "
                "legacy single-rate path and NOT be comparable. Check that the "
                "task spec from task_kwargs() survived your overrides.")

        # ── data ────────────────────────────────────────────────────────────
        print(f"data dir: {DATA_DIR}")
        first_chunk = first_chunk_file()
        schema_df = pd.DataFrame(columns=pq.ParquetFile(first_chunk).schema.names)
        column_info = surrogate.build_federated_column_info(schema_df, cfg)
        print(f"\ninputs: {len(column_info['input_cols'])}   "
              f"dynamic outputs: {len(column_info['dynamic_cols'])}")

        # Build-once-under-lock: safe when several phases start together.
        ensure_store(cfg, column_info)
        # Fit-once / share: without this every phase re-fits the normalizer,
        # materializing the whole train set in RAM (see ensure_normalizer).
        shared_norm = ensure_normalizer(cfg, column_info)
        # create_dataloaders_multirate is exported from the data subpackage.
        from fmu2ml.surrogate.data import create_dataloaders_multirate
        _nw = getattr(cfg, "num_workers", NUM_WORKERS)
        print(f"loader workers: {_nw}   batch: {cfg.batch_size}")
        train_loader, val_loader, test_loader, normalizer = (
            create_dataloaders_multirate(
                STORE_DIR, cfg, column_info,
                num_workers=_nw,
                normalizer=shared_norm,
                chunk_ids=CHUNK_IDS))

        # ── DDP sharding ────────────────────────────────────────────────
        # create_dataloaders_federated_store() has no `sampler` argument, so
        # without this every rank would iterate the FULL train set and DDP
        # would all-reduce gradients over identical batches: 8x the compute for
        # zero speedup. Rebuild the train loader over the same dataset with a
        # DistributedSampler so each rank owns a disjoint 1/WORLD_SIZE shard.
        # Val/test stay unsharded and identical on every rank, which is what
        # keeps the early-stopping decision consistent (all ranks compute the
        # same val loss and therefore stop on the same epoch — no deadlock).
        if IS_DIST:
            from torch.utils.data.distributed import DistributedSampler

            class _AutoEpochSampler(DistributedSampler):
                """DistributedSampler that re-shuffles itself each epoch.

                The library trainer owns the epoch loop, so nothing external can
                call set_epoch(). Without it the sampler would emit the same
                permutation every epoch. Bumping the epoch as each iterator is
                created gives per-epoch reshuffling with no trainer changes.
                """

                def __iter__(self):
                    it = super().__iter__()
                    self.set_epoch(self.epoch + 1)
                    return it

            _orig = train_loader
            _ds = _orig.dataset

            # An IterableDataset cannot be index-sharded by a sampler; it would
            # silently replay the whole stream on every rank.
            if isinstance(_ds, torch.utils.data.IterableDataset):
                raise RuntimeError(
                    "train loader wraps an IterableDataset — a DistributedSampler "
                    "cannot shard it. Shard inside the dataset by rank instead.")

            _sampler = _AutoEpochSampler(_ds, num_replicas=WORLD_SIZE,
                                         rank=RANK, shuffle=True,
                                         drop_last=True)
            _kw = ({"prefetch_factor": 2, "persistent_workers": False}
                   if _nw > 0 else {})
            # Carry the ORIGINAL loader's collate_fn across. The federated store
            # supplies a custom collate that assembles the multi-rate batch dict
            # ('x', 'x_temporal', per-head targets, ...). Rebuilding without it
            # falls back to default_collate, whose output has none of those keys,
            # and training dies at `batch["x"]` with a bare KeyError. Everything
            # here is copied rather than re-specified for the same reason: any
            # loader argument not carried over is silently reverted to a default.
            train_loader = torch.utils.data.DataLoader(
                _ds,
                batch_size=(_orig.batch_size or cfg.batch_size),
                sampler=_sampler,
                num_workers=_nw,
                collate_fn=_orig.collate_fn,
                pin_memory=_orig.pin_memory,
                worker_init_fn=_orig.worker_init_fn,
                timeout=_orig.timeout,
                drop_last=True,
                **_kw)
            if IS_MAIN:
                print(f"[ddp] train sharded: {len(_ds)} windows / "
                      f"{WORLD_SIZE} ranks = ~{len(_ds)//WORLD_SIZE} each; "
                      f"{len(train_loader)} steps/rank/epoch")

        def _chunks_of(loader):
            return sorted({c[0] for c in loader.dataset._chunks})
        data_manifest = {
            "train_chunks": _chunks_of(train_loader),
            "val_chunks": _chunks_of(val_loader),
            "test_chunks": _chunks_of(test_loader),
            "windows": {s: len(l.dataset) for s, l in
                        [("train", train_loader), ("val", val_loader),
                         ("test", test_loader)]},
        }
        print("DATA MANIFEST:", data_manifest)
        if IS_MAIN:
            with open(self.run_dir / "data_manifest.json", "w") as f:
                json.dump(data_manifest, f, indent=2)

        # ── model ───────────────────────────────────────────────────────────
        model = surrogate.get_architecture(self.arch)(
            cfg, column_info=column_info, **self.arch_kwargs)
        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nmodel: {type(model).__name__}   parameters: {n_params:,}")
        assert getattr(model, "multi_rate", False), (
            f"{type(model).__name__} is not a multi-rate model — the registry "
            "did not dispatch to the multi-rate variant.")

        if IS_MAIN:
            with open(self.run_dir / "config.json", "w") as f:
                json.dump({k: v for k, v in vars(cfg).items()
                           if isinstance(v, (int, float, str, bool, list, dict,
                                             type(None)))},
                          f, indent=2, default=str)
            with open(self.run_dir / "notes.txt", "w") as f:
                f.write(self.notes + "\n")

        # ── train ───────────────────────────────────────────────────────────
        print("\n" + "-" * 74)
        print("TRAINING")
        print("-" * 74)
        trainer = surrogate.SurrogateTrainer(model, cfg)
        t0 = time.time()
        history = trainer.fit(train_loader, val_loader)
        elapsed = time.time() - t0
        print(f"\ntraining done in {elapsed/60:.1f} min   "
              f"best val {history.best_val_loss:.6f} @ epoch {history.best_epoch}")

        # ── DDP teardown ────────────────────────────────────────────────────
        # Every rank holds identical weights after the last all-reduce, so the
        # evaluation/plot/report tail below only needs to run once. Non-main
        # ranks sync, drop the process group and return; rank 0 continues
        # single-process, which means none of that code needs rank guards.
        if IS_DIST:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
                dist.destroy_process_group()
            if not IS_MAIN:
                print(f"[ddp] rank {RANK} done (eval runs on rank 0)")
                return None

        # unwrap DDP so state_dict keys have no 'module.' prefix
        if hasattr(model, "module"):
            model = model.module

        torch.save({"model_state_dict": model.state_dict(),
                    "phase": self.phase, "arch": self.arch},
                   self.run_dir / "model.pt")
        normalizer.save(self.run_dir / "normalizer.json")
        hist_dict = {"train": history.train_losses, "val": history.val_losses,
                     "lr": history.learning_rates,
                     "best_epoch": history.best_epoch,
                     "best_val_loss": history.best_val_loss,
                     "epochs_completed": history.epochs_completed}
        with open(self.run_dir / "history.json", "w") as f:
            json.dump(hist_dict, f, indent=2, default=float)

        # ── evaluate ────────────────────────────────────────────────────────
        print("\n" + "-" * 74)
        print("EVALUATION (test split)")
        print("-" * 74)
        predictions = surrogate.collect_predictions(
            model, test_loader, normalizer, column_info, cfg,
            phase=self.phase, device=device)
        metrics_df = surrogate.compute_metrics(predictions, column_info, cfg,
                                               phase=self.phase)
        metrics_df.to_csv(self.run_dir / "metrics.csv", index=False)

        summary = self._print_summary(metrics_df, cfg)
        per_step = self._per_step_metrics(predictions)
        np.save(self.run_dir / "per_step.npy", per_step, allow_pickle=True)

        # ── figures ─────────────────────────────────────────────────────────
        self._plot_training(hist_dict)
        self._plot_skill_vs_lead(per_step)
        self._plot_error_accum(predictions, cfg)
        self._plot_group_summary(metrics_df)

        # ── report ──────────────────────────────────────────────────────────
        summary.update({"run_id": self.run_id, "phase": self.phase,
                        "arch": self.arch, "model": type(model).__name__,
                        "n_params": int(n_params),
                        "train_minutes": round(elapsed / 60, 2),
                        "best_epoch": int(history.best_epoch),
                        "epochs_completed": int(history.epochs_completed)})
        with open(self.run_dir / "phase_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)

        self._write_report(metrics_df, cfg, summary, data_manifest)
        print(f"\ntotal wall clock: {(time.time()-t_start)/60:.1f} min")
        print(f"artifacts: {self.run_dir}")
        return metrics_df

    # ── reporting helpers ───────────────────────────────────────────────────
    @staticmethod
    def _r2col(df):
        return "R²" if "R²" in df.columns else "R2"

    def _print_summary(self, metrics_df, cfg) -> Dict[str, Any]:
        r2 = self._r2col(metrics_df)
        pcol = "Persistence_R²" if "Persistence_R²" in metrics_df.columns \
            else "Persistence_R2"
        print("=" * 70)
        print(f"RESULTS SUMMARY — {self.run_id}")
        print("=" * 70)
        print(f"\n--- All {len(metrics_df)} Outputs ---")
        print(f"  Mean/Median R²:    {metrics_df[r2].mean():.4f} / "
              f"{metrics_df[r2].median():.4f}")
        print(f"  Min R²:            {metrics_df[r2].min():.4f}")
        print(f"  Beats Persistence: {metrics_df['Beats_Persistence'].sum()}/"
              f"{len(metrics_df)} ({metrics_df['Beats_Persistence'].mean():.1%})")
        print(f"  Mean Skill Score:  {metrics_df['Skill_Score'].mean():+.4f}")

        print("\n--- Per Decoder Head Group ---")
        groups = {}
        for g in ALL_HEAD_NAMES:
            grp = metrics_df[metrics_df["Group"] == g]
            if len(grp) == 0:
                continue
            rate = cfg.head_subsample[g]
            K = cfg.head_prediction_steps[g]
            margin = (grp[r2] - grp[pcol]).mean() if pcol in grp else float("nan")
            groups[g] = {"n": int(len(grp)), "horizon_s": int(rate * K),
                         "mean_r2": float(grp[r2].mean()),
                         "median_r2": float(grp[r2].median()),
                         "beats": int(grp["Beats_Persistence"].sum()),
                         "beats_frac": float(grp["Beats_Persistence"].mean()),
                         "skill": float(grp["Skill_Score"].mean())}
            print(f"\n  {g} ({len(grp)} outputs, horizon {rate*K}s):")
            print(f"    Mean/Median R²: {grp[r2].mean():.4f} / "
                  f"{grp[r2].median():.4f}"
                  f"   Beats: {grp['Beats_Persistence'].sum()}/{len(grp)} "
                  f"({grp['Beats_Persistence'].mean():.1%})"
                  f"   Skill: {grp['Skill_Score'].mean():+.4f}"
                  f"   R²-margin: {margin:+.2e}")

        print("\n--- Per Output Type ---")
        agg = {r2: ["mean", "min"], "RMSE": "mean", "MAE": "mean",
               "Beats_Persistence": "mean"}
        if "Variance_Ratio" in metrics_df.columns:
            agg["Variance_Ratio"] = "mean"
        print(metrics_df.groupby("Output_Type").agg(agg).round(4).to_string())

        return {"n_outputs": int(len(metrics_df)),
                "mean_r2": float(metrics_df[r2].mean()),
                "median_r2": float(metrics_df[r2].median()),
                "min_r2": float(metrics_df[r2].min()),
                "beats_frac": float(metrics_df["Beats_Persistence"].mean()),
                "mean_skill": float(metrics_df["Skill_Score"].mean()),
                "groups": groups}

    @staticmethod
    def _per_step_metrics(predictions) -> Dict[str, Dict[str, np.ndarray]]:
        """Skill and beats-persistence resolved per lead step, per head."""
        eps = 1e-10
        per_group = predictions.get("per_group", {})
        out = {}
        for h in ALL_HEAD_NAMES:
            g = per_group.get(SUF[h])
            if g is None:
                continue
            pred = g["pred_absolute"]; targ = g["target_absolute"]
            last = g["last"]; rate = g["rate"]
            ss_tot = ((targ - targ.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
            ss_res = ((targ - pred) ** 2).sum(axis=0)
            ss_per = ((targ - last[:, None, :]) ** 2).sum(axis=0)
            r2 = 1 - ss_res / (ss_tot + eps)
            r2p = 1 - ss_per / (ss_tot + eps)
            out[h] = {"lead_s": (np.arange(pred.shape[1]) + 1) * rate,
                      "skill": ((r2 - r2p) / (1 - r2p + eps)).mean(axis=1),
                      "beat": (r2 > r2p).mean(axis=1) * 100}
        return out

    def _plot_training(self, hist):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        plotted = False
        for key, vals in list(hist["train"].items())[:8]:
            if vals:
                ax.plot(np.arange(1, len(vals) + 1), vals, lw=1.6,
                        label=key, alpha=0.85)
                plotted = True
        for key, vals in list(hist["val"].items())[:8]:
            if vals:
                ax.plot(np.arange(1, len(vals) + 1), vals, lw=1.6, ls="--",
                        label=key, alpha=0.85)
                plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set(yscale="log", xlabel="epoch", ylabel="loss",
               title=f"Training curves — {self.run_id}")
        ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7, ncols=2)
        self.save_fig(fig, "training_curves.png",
                      "Training and validation loss per epoch.")

    def _plot_skill_vs_lead(self, per_step):
        if not per_step:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        for h, d in per_step.items():
            ax1.plot(d["lead_s"], d["skill"], marker="o", ms=3, lw=2,
                     color=GROUP_COLORS[h], label=h)
            ax2.plot(d["lead_s"], d["beat"], marker="o", ms=3, lw=2,
                     color=GROUP_COLORS[h], label=h)
        ax1.axhline(0, color="0.4", lw=1, ls=":")
        ax1.set(xscale="log", xlabel="lead time (s)",
                ylabel="mean skill vs persistence",
                title="Skill vs lead time (0 = persistence-equivalent)")
        ax2.set(xscale="log", xlabel="lead time (s)",
                ylabel="% outputs beating persistence", ylim=(-3, 103),
                title="Beats-persistence vs lead time")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.25); ax.legend(ncols=2, fontsize=8)
        self.save_fig(fig, "skill_vs_lead.png",
                      "Skill and beats-persistence fraction versus lead time.")

    def _plot_error_accum(self, predictions, cfg):
        per_group = predictions.get("per_group", {})
        heads = [h for h in ALL_HEAD_NAMES if SUF[h] in per_group]
        if not heads:
            return
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        for ax, h in zip(axes.ravel(), heads):
            g = per_group[SUF[h]]
            pred, targ, last = (g["pred_absolute"], g["target_absolute"],
                                g["last"])
            lead = (np.arange(pred.shape[1]) + 1) * g["rate"]
            ax.plot(lead, np.abs(targ - pred).mean(axis=(0, 2)), marker="o",
                    ms=3, lw=2, color=GROUP_COLORS[h], label="model")
            ax.plot(lead, np.abs(targ - last[:, None, :]).mean(axis=(0, 2)),
                    lw=1.5, ls="--", color="0.45", label="persistence")
            ax.set_title(f"{h}  (rate {g['rate']}s, "
                         f"branch {cfg.encoder_groups[h]})", fontsize=11)
            ax.set_xlabel("lead time (s)")
            ax.set_ylabel(f"MAE ({GROUP_UNITS[h]})")
            ax.grid(alpha=0.25); ax.legend(fontsize=8)
        for ax in axes.ravel()[len(heads):]:
            ax.axis("off")
        fig.suptitle("Error accumulation across the horizon "
                     "(model vs persistence)", y=1.02)
        self.save_fig(fig, "error_accum.png",
                      "Mean absolute error across the horizon, model versus "
                      "persistence, per head.")

    def _plot_group_summary(self, metrics_df):
        r2 = self._r2col(metrics_df)
        groups = [g for g in ALL_HEAD_NAMES
                  if (metrics_df["Group"] == g).any()]
        if not groups:
            return
        skill = [metrics_df[metrics_df["Group"] == g]["Skill_Score"].mean()
                 for g in groups]
        beats = [metrics_df[metrics_df["Group"] == g]["Beats_Persistence"].mean()
                 * 100 for g in groups]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.bar(groups, skill, color=[GROUP_COLORS[g] for g in groups])
        a1.axhline(0, color="0.3", lw=1)
        a1.set(ylabel="mean skill vs persistence", title="Skill by head group")
        a2.bar(groups, beats, color=[GROUP_COLORS[g] for g in groups])
        a2.set(ylabel="% outputs beating persistence", ylim=(0, 105),
               title="Beats-persistence by head group")
        for a in (a1, a2):
            a.grid(alpha=0.25, axis="y")
        self.save_fig(fig, "group_summary.png",
                      "Per-head-group skill and beats-persistence.")

    def _write_report(self, metrics_df, cfg, summary, data_manifest):
        r2 = self._r2col(metrics_df)
        L = []
        A = L.append
        A(f"# Phase {self.phase} ({self.arch}) — `{self.run_id}`")
        A("")
        A(f"*Generated {datetime.datetime.now():%Y-%m-%d %H:%M:%S} · "
          f"{summary['model']} · {summary['n_params']:,} parameters · "
          f"{summary['train_minutes']:.1f} min*")
        A("")
        if self.notes:
            A("## Run notes"); A(""); A(self.notes); A("")

        A("## Headline")
        A("")
        A("| metric | value |")
        A("| --- | --- |")
        A(f"| Outputs | {summary['n_outputs']} |")
        A(f"| Mean R² | {summary['mean_r2']:.4f} |")
        A(f"| Median R² | {summary['median_r2']:.4f} |")
        A(f"| Min R² | {summary['min_r2']:.4f} |")
        A(f"| Beats persistence | {summary['beats_frac']:.1%} |")
        A(f"| **Mean skill score** | **{summary['mean_skill']:+.4f}** |")
        A("")
        A("> Skill vs persistence is the headline metric, not mean R². "
          "Several channels sit at their noise floor, where R² is low for "
          "*any* predictor including persistence.")
        A("")

        A("## Task (identical across phases 1-6 and run19b)")
        A("")
        A("| group | rate (s) | history | K | window (s) | horizon (s) | branch |")
        A("| --- | --- | --- | --- | --- | --- | --- |")
        for g in ALL_HEAD_NAMES:
            rr = cfg.head_subsample[g]; hh = cfg.head_history_steps[g]
            kk = cfg.head_prediction_steps[g]
            A(f"| {g} | {rr} | {hh} | {kk} | {hh*rr} | {kk*rr} | "
              f"{cfg.encoder_groups[g]} |")
        A("")
        A("| split | chunks | windows |")
        A("| --- | --- | --- |")
        for s in ("train", "val", "test"):
            A(f"| {s} | {data_manifest[f'{s}_chunks']} | "
              f"{data_manifest['windows'][s]:,} |")
        A("")

        A("## Per head group")
        A("")
        A("| group | outputs | horizon (s) | mean R² | median R² | beats | "
          "mean skill |")
        A("| --- | --- | --- | --- | --- | --- | --- |")
        for g, d in summary["groups"].items():
            A(f"| {g} | {d['n']} | {d['horizon_s']} | {d['mean_r2']:.4f} | "
              f"{d['median_r2']:.4f} | {d['beats']}/{d['n']} "
              f"({d['beats_frac']:.0%}) | {d['skill']:+.4f} |")
        A("")

        A("## Per output type")
        A("")
        agg = {r2: ["mean", "min"], "RMSE": "mean", "MAE": "mean",
               "Beats_Persistence": "mean"}
        ts = metrics_df.groupby("Output_Type").agg(agg).round(4)
        ts.columns = [" ".join(c).strip() for c in ts.columns]
        A(_df_to_markdown(ts.reset_index()))
        A("")

        A("## 20 worst outputs by R²")
        A("")
        cols = [c for c in ["Output", "Group", r2, "Skill_Score", "RMSE"]
                if c in metrics_df.columns]
        A(_df_to_markdown(metrics_df.nsmallest(20, r2)[cols]))
        A("")

        A("## Figures")
        A("")
        for caption, filename in self.figures:
            A(f"### {caption}"); A(""); A(f"![{caption}]({filename})"); A("")

        A("## Full console log")
        A(""); A("```text"); A(self._buffer.getvalue().rstrip("\n")); A("```")
        A("")
        A("## Artifacts")
        A("")
        for name in sorted(p.name for p in self.run_dir.iterdir() if p.is_file()):
            A(f"- `{name}`")
        A("")
        (self.run_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
        print(f"report: {self.run_dir / 'report.md'}")


def _df_to_markdown(df) -> str:
    """GitHub-flavored markdown table, with a no-dependency fallback."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        cols = list(df.columns)
        head = "| " + " | ".join(str(c) for c in cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = ["| " + " | ".join(
            f"{v:.4f}" if isinstance(v, float) else str(v)
            for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
        return "\n".join([head, sep, *rows])


# ─────────────────────────────────────────────────────────────────────────────
# `python -m fmu2ml.surrogate.phase_common --prewarm` — build the store once,
# serially, before the parallel phases start. Doing it here means the N
# training processes all find the manifest already present and skip straight
# to their loaders.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import pyarrow.parquet as pq
    import fmu2ml.surrogate as surrogate
    from fmu2ml.surrogate.data import store as _store

    ap = argparse.ArgumentParser(description="phase_common utilities")
    ap.add_argument("--prewarm", action="store_true",
                    help="build the multi-rate store for the shared task spec")
    args = ap.parse_args()

    if args.prewarm:
        print(f"[prewarm] DATA_DIR  = {DATA_DIR}")
        print(f"[prewarm] STORE_DIR = {STORE_DIR}")
        print(f"[prewarm] chunks    = {len(CHUNK_IDS)} "
              f"(split={_SPLIT}, ratios {TRAIN_RATIO}/{VAL_RATIO}, "
              f"seed {SPLIT_SEED})")
        cfg = surrogate.FederatedConfig(**task_kwargs())
        first_chunk = first_chunk_file()
        schema_df = pd.DataFrame(
            columns=pq.ParquetFile(first_chunk).schema.names)
        column_info = surrogate.build_federated_column_info(schema_df, cfg)
        ensure_store(cfg, column_info)
        ensure_normalizer(cfg, column_info)
        print("[prewarm] done — phases can now start in parallel")
    else:
        ap.print_help()
