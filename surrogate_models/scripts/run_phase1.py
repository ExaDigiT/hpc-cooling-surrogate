#!/usr/bin/env python3
"""Phase 1 — Baseline LSTM, multi-rate, on the 720 h `systematic-720` data.

Usage, from any directory (the run is written to
surrogate_models/runs/<RUN_ID>/, next to the notebooks' runs):

    python surrogate_models/scripts/run_phase1.py
    srun -n8 --gpus-per-node=8 --gpu-bind=none \
        python surrogate_models/scripts/run_phase1.py       # 8-GPU DDP

Data, split and training budget come from the PHASE_* environment variables
(use absolute paths; see fmu2ml/surrogate/phase_common.py). Before starting
several phases at once, build the shared store once, from the repo root:
``python -m fmu2ml.surrogate.phase_common --prewarm``.

The shared task spec, data pipeline, metrics and report live in
`fmu2ml/surrogate/phase_common.py` — see its module docstring for why that
is shared rather than copied. This script holds only what is specific to phase 1.

ARCHITECTURE (multi-rate variant, `MultiRateLSTM`)
    One LSTM + temporal-attention encoder per branch over concat(u_hist,
    y_hist) at that branch's own rate, the branch contexts concatenated, then
    a direct per-head MLP decoder. No operator basis and no trunk — that is
    exactly what separates phase 1 from phases 2+, and keeping it is the point
    of running it as a baseline.

WHAT THIS RUN IS FOR
    The floor of the comparison table. Everything phases 2-6 add (operator
    basis, Fourier trunk, domain experts, federated heads, physics) has to
    earn its place against this.
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

RUN_ID = "phase1_lstm_mr"

RUN_NOTES = (
    "Phase 1 baseline LSTM trained MULTI-RATE on the 720 h "
    "`systematic-720` data, on exactly the task run19b (MR-MIONet) was "
    "scored on: hydraulics at 3 s with 300 s look-back, thermal/power at "
    "30 s with 1200 s look-back, uniform K=20 horizons (600 s "
    "thermal/power, 60 s hydraulic), seeded 80/10/10 chunk split.\n\n"
    "Why it exists: until the library gained multi-rate support for phases "
    "1-6, the baselines were trained on a DIFFERENT (single-rate) task than "
    "MR-MIONet, so every previous cross-phase comparison was apples to "
    "oranges. This run and its siblings put all seven models on one task so "
    "the comparison table is honest.\n\n"
    "Architecture: per-branch LSTM + temporal attention -> concatenated "
    "context -> direct per-head MLP. Deliberately NO operator basis and NO "
    "Fourier trunk; phase 1 is the floor those features must beat.\n\n"
    "Read the result by mean SKILL vs persistence, not mean R². G_Vs "
    "(V_flow_sec) is a measured channel noise floor — the signal-floor "
    "diagnostic showed its deltas are temporally white — so a low R² there "
    "is expected for every phase and is not a phase-1 failure."
)

config = surrogate.Phase1Config(
    **task_kwargs(),

    #  phase-1 architecture ────────────────────────────────────────────────
    lstm_hidden_size=256,
    lstm_num_layers=2,
    lstm_dropout=0.1,
    attention_heads=4,
    attention_dropout=0.1,
    decoder_hidden_sizes=[512, 256],
    decoder_dropout=0.1,

    #  training ────────────────────────────────────────────────────────────
    # 1e-3 is this phase's own default and is appropriate for a from-scratch
    # LSTM; MR-MIONet's 3e-5 is tuned for its much deeper operator stack and
    # would badly underfit here. Each phase trains at ITS OWN best-known
    # settings — the task is what must be identical, not the hyperparameters.
    learning_rate=scaled_lr(1e-3),
    # weight decay, epoch budget and patience are shared by every phase
    # (see phase_common); these are the values the reported runs used.
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
)

if __name__ == "__main__":
    PhaseRun(run_id=RUN_ID, arch="lstm", phase="1",
             config=config, notes=RUN_NOTES).execute()
