"""Phase 5 (grouped DeepMMNet) on Frontier, reusing run_phase5.py's config.

Usage, from any directory:

    PHASE_SYSTEM=frontier PHASE_DATA_DIR=/abs/path/systematic-200 PHASE_N_CHUNKS=24 \\
    FRONTIER_TRAIN_ALL_HEADS=1 FRONTIER_RUN_ID=frontier_phase5_mr_allheads_200h \\
        python surrogate_models/scripts/run_frontier_phase5.py

The run is written to surrogate_models/runs/<FRONTIER_RUN_ID>/. Build the store
first with ``python -m fmu2ml.surrogate.phase_common --prewarm`` (same env).

FRONTIER_TRAIN_ALL_HEADS=1 trains all six decoder groups. Without it the
Summit task spec applies, where G_V, G_Vs and G_ps have loss weight 0 and emit
exact persistence. On Frontier a ridge model beats persistence on all three, so
they are learnable there; see surrogate_models/frontier/README.md.
"""
import dataclasses
import os

import run_phase5 as p5  # chdirs to surrogate_models/, builds config from PHASE_* env
from fmu2ml.surrogate.phase_common import DATA_DIR, NUM_CDUS, SYSTEM, PhaseRun

assert SYSTEM == "frontier" and NUM_CDUS == 25, (
    f"expected PHASE_SYSTEM=frontier (25 CDUs), got {SYSTEM} ({NUM_CDUS} CDUs)")

RUN_ID = os.environ.get("FRONTIER_RUN_ID", "frontier_phase5_mr")
ALL_HEADS = os.environ.get("FRONTIER_TRAIN_ALL_HEADS") == "1"

config = p5.config
if ALL_HEADS:
    config = dataclasses.replace(
        config, head_loss_weights={h: 1.0 for h in config.head_loss_weights})
    assert config.persistence_heads == [], config.persistence_heads

if __name__ == "__main__":
    print("head_loss_weights:", config.head_loss_weights)
    PhaseRun(run_id=RUN_ID, arch="federated", phase="5", config=config,
             notes=(f"Frontier (Simulator_olcf5_base, 25 CDUs): Phase 5 grouped "
                    f"DeepMMNet, run_phase5.py config, "
                    f"{'all six heads trained' if ALL_HEADS else 'Summit head weights'}, "
                    f"data {os.path.basename(DATA_DIR)}.")).execute()
