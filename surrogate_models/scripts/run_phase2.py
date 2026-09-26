#!/usr/bin/env python3
"""Phase 2 — Basic DeepONet, multi-rate, on the 720 h `systematic-720` data.

Usage, from any directory (the run is written to
surrogate_models/runs/<RUN_ID>/, next to the notebooks' runs):

    python surrogate_models/scripts/run_phase2.py
    srun -n8 --gpus-per-node=8 --gpu-bind=none \
        python surrogate_models/scripts/run_phase2.py       # 8-GPU DDP

Data, split and training budget come from the PHASE_* environment variables
(use absolute paths; see fmu2ml/surrogate/phase_common.py). Before starting
several phases at once, build the shared store once, from the repo root:
``python -m fmu2ml.surrogate.phase_common --prewarm``.

Shared task spec / data / metrics / report: see `fmu2ml/surrogate/phase_common.py`.

ARCHITECTURE (multi-rate variant, `MultiRateDeepONet`)
    The first operator-learning phase: per-branch LSTM branch nets produce
    basis coefficients, a shared Fourier-feature trunk over physical lead time
    produces the basis, and each head is the coefficient x basis inner product
    (the "plain" operator decoration). This is what phase 1's direct MLP is
    being compared against — does the operator basis + trunk buy anything?
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

RUN_ID = "phase2_deeponet_mr"

RUN_NOTES = (
    "Phase 2 basic DeepONet trained MULTI-RATE on the 720 h "
    "`systematic-720` data, on the identical task as run19b and the other "
    "phases (3 s hydraulics / 30 s thermal-power, seeded 80/10/10 chunk "
    "split (seed 42)).\n\n"
    "Architecture: per-branch LSTM branch net -> basis coefficients; shared "
    "Fourier trunk over physical lead time -> basis; per-head plain operator "
    "(coeff x basis). This is the first phase with an operator basis and a "
    "trunk; comparing it to phase 1 isolates what those two components add.\n\n"
    "Read by mean SKILL vs persistence, not mean R² (G_Vs is a measured "
    "channel noise floor for all phases)."
)

config = surrogate.Phase2Config(
    **task_kwargs(),

    #  phase-2 architecture ────────────────────────────────────────────────
    branch_lstm_hidden=256,
    branch_lstm_layers=2,
    branch_lstm_dropout=0.1,
    branch_mlp_hidden=[256, 128],
    trunk_n_fourier=16,
    trunk_hidden_sizes=[128, 128],
    basis_dim=128,
    include_output_history=True,
    # PCA input whitening is a phase-2/3 feature; off here because the
    # multi-rate store feeds each branch its own concat(u, y) window and the
    # whitening was fit for the single-rate flat-input path. Left explicit so
    # the choice is visible rather than a silent default.
    use_input_whitening=False,

    #  training ────────────────────────────────────────────────────────────
    learning_rate=scaled_lr(1e-3),
    # weight decay, epoch budget and patience are shared by every phase
    # (see phase_common); these are the values the reported runs used.
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
)

if __name__ == "__main__":
    PhaseRun(run_id=RUN_ID, arch="deeponet", phase="2",
             config=config, notes=RUN_NOTES).execute()
