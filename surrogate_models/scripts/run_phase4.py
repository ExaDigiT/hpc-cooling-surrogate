#!/usr/bin/env python3
"""Phase 4 — Domain-specific DeepONet, multi-rate, on `systematic-720`.

Usage, from any directory (the run is written to
surrogate_models/runs/<RUN_ID>/, next to the notebooks' runs):

    python surrogate_models/scripts/run_phase4.py
    srun -n8 --gpus-per-node=8 --gpu-bind=none \
        python surrogate_models/scripts/run_phase4.py       # 8-GPU DDP

Data, split and training budget come from the PHASE_* environment variables
(use absolute paths; see fmu2ml/surrogate/phase_common.py). Before starting
several phases at once, build the shared store once, from the repo root:
``python -m fmu2ml.surrogate.phase_common --prewarm``.

Shared task spec / data / metrics / report: see `fmu2ml/surrogate/phase_common.py`.

ARCHITECTURE (multi-rate variant, `MultiRateDomainDeepONet`)
    Every head group is treated as its own domain expert with a scale + bias +
    sigmoid skip gate. In multi-rate mode the six canonical groups ARE the
    domains, so this is the natural home of the domain idea: a specialized
    operator head per physical quantity (temperature, primary/secondary flow,
    primary/secondary pressure, power) sharing the branch encoder bank.
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

RUN_ID = "phase4_domain_deeponet_mr"

RUN_NOTES = (
    "Phase 4 domain-specific DeepONet trained MULTI-RATE on the 720 h "
    "`systematic-720` data, identical task to run19b and the other phases.\n\n"
    "Architecture: one domain-expert operator head per canonical group with a "
    "scale + bias + sigmoid skip gate, over the shared branch encoder bank. "
    "In multi-rate mode the six groups (T, V, p, Vs, ps, W) map exactly onto "
    "the phase-4 'domains', so this is the most natural multi-rate form of the "
    "domain-expert idea. Comparing to phase 2/3 isolates what per-domain "
    "specialization buys over a shared operator.\n\n"
    "Read by mean SKILL vs persistence, not mean R²."
)

config = surrogate.Phase4Config(
    **task_kwargs(),

    #  phase-4 architecture ────────────────────────────────────────────────
    # `domain` is the single-rate expert selector; unused in multi-rate mode
    # (every group is its own domain head) but kept at its default so the
    # config validates.
    branch_lstm_hidden=256,
    branch_lstm_layers=2,
    branch_lstm_dropout=0.1,
    trunk_n_fourier=16,
    trunk_hidden_sizes=[128, 128],
    basis_dim=128,
    include_output_history=True,

    #  training ────────────────────────────────────────────────────────────
    learning_rate=scaled_lr(1e-3),
    # weight decay, epoch budget and patience are shared by every phase
    # (see phase_common); these are the values the reported runs used.
    weight_decay=WEIGHT_DECAY,
    max_epochs=MAX_EPOCHS,
    patience=PATIENCE,
)

if __name__ == "__main__":
    PhaseRun(run_id=RUN_ID, arch="domain_deeponet", phase="4",
             config=config, notes=RUN_NOTES).execute()
