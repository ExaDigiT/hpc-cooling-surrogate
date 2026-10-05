"""Phase 5 (grouped DeepMMNet) for any system, reusing run_phase5.py's config.

Usage, from any directory:

    PHASE_SYSTEM=marconi100 PHASE_DATA_DIR=/abs/path/systematic-200h PHASE_N_CHUNKS=24 \\
    SURROGATE_RUN_ID=marconi100_phase5_mr \\
    SURROGATE_HEAD_WEIGHTS="G_T=1,G_V=1,G_p=1,G_Vs=1,G_ps=1,G_W=1" \\
        python surrogate_models/scripts/run_system_phase5.py

PHASE_SYSTEM selects the system (CDU count and input columns from
config/<system>/). SURROGATE_HEAD_WEIGHTS sets the per-group loss weights; a
weight of 0 makes that group persistence-locked (exact persistence output).
Set it from task_spec_analysis.py: train every group the ridge model can
predict. Unset, the Summit spec applies (G_V, G_Vs, G_ps at 0). The run is
written to surrogate_models/runs/<SURROGATE_RUN_ID>/.
"""
import dataclasses
import os

import run_phase5 as p5  # chdirs to surrogate_models/, builds config from PHASE_* env
from fmu2ml.surrogate.phase_common import DATA_DIR, NUM_CDUS, SYSTEM, PhaseRun

RUN_ID = os.environ.get("SURROGATE_RUN_ID", f"{SYSTEM}_phase5_mr")

config = p5.config
spec = os.environ.get("SURROGATE_HEAD_WEIGHTS")
if spec:
    weights = dict(config.head_loss_weights)
    for item in spec.split(","):
        head, value = item.split("=")
        if head.strip() not in weights:
            raise ValueError(f"unknown head {head!r}; expected one of {sorted(weights)}")
        weights[head.strip()] = float(value)
    config = dataclasses.replace(config, head_loss_weights=weights)

if __name__ == "__main__":
    print(f"system {SYSTEM} ({NUM_CDUS} CDUs), run {RUN_ID}")
    print("head_loss_weights:", config.head_loss_weights,
          "| persistence-locked:", config.persistence_heads or "none")
    PhaseRun(run_id=RUN_ID, arch="federated", phase="5", config=config,
             notes=(f"{SYSTEM} ({NUM_CDUS} CDUs): Phase 5 grouped DeepMMNet, run_phase5.py "
                    f"config, head weights {config.head_loss_weights}, data "
                    f"{os.path.basename(DATA_DIR)}.")).execute()
