#!/usr/bin/env python3
"""Phase 5 — Grouped DeepMMNet, multi-rate, on `systematic-720`.

Usage, from any directory (the run is written to
surrogate_models/runs/<RUN_ID>/, next to the notebooks' runs):

    python surrogate_models/scripts/run_phase5.py
    srun -n8 --gpus-per-node=8 --gpu-bind=none \
        python surrogate_models/scripts/run_phase5.py       # 8-GPU DDP

Data, split and training budget come from the PHASE_* environment variables
(use absolute paths; see fmu2ml/surrogate/phase_common.py). Before starting
several phases at once, build the shared store once, from the repo root:
``python -m fmu2ml.surrogate.phase_common --prewarm``.

Shared task spec / data / metrics / report: see `fmu2ml/surrogate/phase_common.py`.

ARCHITECTURE (`GroupedDeepMMNet` — natively multi-rate)
    The library's grouped model IS the multi-rate backbone MR-MIONet was
    built from: per-branch u/y LSTM encoders, one shared Fourier trunk, and
    per-group decoder heads (standard or skip). Phase 5 is therefore the
    direct, in-library ancestor of run19b — the honest "what does the base
    grouped model score before the run-series' custom additions
    (future-forcing branch, attention fusion, GatedDeltaHead, beta-NLL,
    per-head checkpointing)."

TRAINING
    Two-phase protocol (round-robin per-head warmup, then joint fine-tuning
    with differential LRs) driven by `phase1_epochs` / `phase2_epochs`, NOT
    `max_epochs`.
"""

import os
import sys

# Use THIS repo's fmu2ml (<repo>/surrogate_models/scripts/../..), whatever the
# working directory or PYTHONPATH say, and write the run to
# <repo>/surrogate_models/runs/<RUN_ID>/, where the notebooks write theirs.
SURROGATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(SURROGATE_DIR)
os.chdir(SURROGATE_DIR)
sys.path.insert(0, REPO_ROOT)

import fmu2ml.surrogate as surrogate  # noqa: E402
assert surrogate.__file__.startswith(os.path.join(REPO_ROOT, "fmu2ml", "")), (
    f"fmu2ml came from {surrogate.__file__}, expected {REPO_ROOT}/fmu2ml")
from fmu2ml.surrogate.phase_common import (  # noqa: E402
    PhaseRun, task_kwargs, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, scaled_lr)

RUN_ID = "phase5_federated_mr"

RUN_NOTES = (
    "Phase 5 grouped DeepMMNet trained MULTI-RATE on the 720 h "
    "`systematic-720` data, identical task to run19b and the other phases.\n\n"
    "Architecture: per-branch u/y LSTM encoders + one shared Fourier trunk + "
    "per-group decoder heads (standard/skip per head_types). This is the "
    "in-library ancestor of the MR-MIONet run series: run19b = this backbone "
    "PLUS a future-forcing branch, attention fusion, one learnable "
    "GatedDeltaHead per group, beta-NLL, and per-head best checkpointing. "
    "Phase 5's gap to run19b is exactly the value of those additions.\n\n"
    "Two-phase training: round-robin per-head warmup (phase1_epochs) then "
    "joint fine-tuning with differential LRs (phase2_epochs).\n\n"
    "Read by mean SKILL vs persistence, not mean R²."
)

config = surrogate.Phase5Config(
    **task_kwargs(),

    #  phase-5 architecture (the federated backbone) ────────────────────────
    u_branch_lstm_hidden=128,
    u_branch_lstm_layers=2,
    u_branch_lstm_dropout=0.3,
    y_branch_lstm_hidden=128,
    y_branch_lstm_layers=2,
    y_branch_lstm_dropout=0.3,
    trunk_n_fourier=8,
    trunk_hidden_sizes=[128, 128],
    basis_dim=256,
    head_hidden_sizes={"G_T": [256, 128], "G_V": [256, 128], "G_p": [256, 128],
                       "G_Vs": [128, 64], "G_ps": [128, 64], "G_W": [128, 64]},

    #  training (two-phase; NOT max_epochs) ─────────────────────────────────
    learning_rate=scaled_lr(1e-3),
    # weight decay and patience are shared by every phase (see
    # phase_common); these are the values the reported runs used.
    weight_decay=WEIGHT_DECAY,
    phase1_epochs=50,
    phase2_epochs=100,
    patience=PATIENCE,
    loss_type="huber",
    huber_delta=0.5,
)

if __name__ == "__main__":
    PhaseRun(run_id=RUN_ID, arch="federated", phase="5",
             config=config, notes=RUN_NOTES).execute()
