#!/usr/bin/env python3
"""Phase 3 — Hybrid DeepONet, multi-rate, on the 720 h `systematic-720` data.

Usage, from any directory (the run is written to
surrogate_models/runs/<RUN_ID>/, next to the notebooks' runs):

    python surrogate_models/scripts/run_phase3.py
    srun -n8 --gpus-per-node=8 --gpu-bind=none \
        python surrogate_models/scripts/run_phase3.py       # 8-GPU DDP

Data, split and training budget come from the PHASE_* environment variables
(use absolute paths; see fmu2ml/surrogate/phase_common.py). Before starting
several phases at once, build the shared store once, from the repo root:
``python -m fmu2ml.surrogate.phase_common --prewarm``.

Shared task spec / data / metrics / report: see `fmu2ml/surrogate/phase_common.py`.

ARCHITECTURE (multi-rate variant, `MultiRateHybridDeepONet`)
    Phase 2's operator plus an alpha-blend gate: each head learns to mix its
    operator (basis-coefficient) prediction with a skip/algebraic path. On the
    all-CDU store task the CDU-embedding / explicit algebraic pathway of the
    original single-rate hybrid has no analogue (documented in the library);
    the multi-rate variant keeps the alpha blend, which is the part that
    transfers.
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

RUN_ID = "phase3_hybrid_deeponet_mr"

RUN_NOTES = (
    "Phase 3 hybrid DeepONet trained MULTI-RATE on the 720 h "
    "`systematic-720` data, identical task to run19b and the other phases.\n\n"
    "Architecture: phase-2 operator + a learned alpha-blend gate per head "
    "(operator vs skip path). NOTE the original single-rate hybrid's "
    "CDU-embedding / algebraic pathway has no analogue on the all-CDU store "
    "task, so the multi-rate variant keeps only the alpha blend — the "
    "component that transfers to this task. Comparing to phase 2 isolates "
    "what the blend gate adds.\n\n"
    "Read by mean SKILL vs persistence, not mean R²."
)

config = surrogate.Phase3Config(
    **task_kwargs(),

    #  phase-3 architecture (inherits DeepONetConfig fields) ────────────────
    branch_lstm_hidden=256,
    branch_lstm_layers=2,
    branch_lstm_dropout=0.1,
    branch_mlp_hidden=[256, 128],
    trunk_n_fourier=16,
    trunk_hidden_sizes=[128, 128],
    basis_dim=128,
    include_output_history=True,
    use_input_whitening=False,
    use_skip_connection=True,

    #  training ────────────────────────────────────────────────────────────
    learning_rate=scaled_lr(1e-3),
    # weight decay, epoch budget and patience are shared by every phase
    # (see phase_common); these are the values the reported runs used.
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
)

if __name__ == "__main__":
    PhaseRun(run_id=RUN_ID, arch="hybrid_deeponet", phase="3",
             config=config, notes=RUN_NOTES).execute()
