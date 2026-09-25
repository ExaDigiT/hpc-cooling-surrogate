# HPC Cooling Surrogate

**Deep Learning Surrogate Models for FMU-Based Cooling System Simulation in HPC Digital Twins**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---


## Overview

This repository provides deep learning surrogates that replace the FMU-based cooling-system simulator of the [ExaDigiT](https://exadigit.readthedocs.io/). Five architectures are developed, from a baseline LSTM through DeepONet variants to the final **grouped multi-decoder DeepM&Mnet**, and all five are trained and scored on an identical multi-rate task for Summit (257 CDUs, 2,827 outputs).

The grouped multi-decoder DeepM&Mnet (5.87M parameters):
- Mean R² = 0.926 (median 0.999) across all 2,827 outputs.
- Beats persistence on 1,541 of the 1,799 outputs that carry learnable structure (85.7%); the remaining 1,028 are persistence-locked.
- 19,514× faster than the FMU at the 60 s evaluation window: 4.1 ms per forward for all 257 CDUs against 79.3 s of FMU wall clock (16,093× at the 30 s window). A 10,000-member uncertainty ensemble drops from 220 hours to 41 seconds.

The five architectures agree on aggregate R² to within 0.0002 and separate only at long horizons, where the grouped model keeps a closed-loop skill of 0.492 at a 20–30 min lead against 0.083 for the LSTM. Skill against persistence, not R², is the headline metric: several channels sit at a measured noise floor.

| Model | Parameters | Mean R² | Closed-loop skill (20–30 min) | ms / forward | Speedup (60 s window) |
|---|---|---|---|---|---|
| P1: LSTM | 20.79M | 0.926 | 0.083 | 4.289 | 18,499× |
| P2: DeepONet | 55.59M | 0.926 | 0.268 | 4.410 | 17,992× |
| P3: Hybrid DeepONet | 55.60M | 0.926 | 0.375 | 4.631 | 17,133× |
| P4: Domain-specific DeepONet | 55.60M | 0.926 | 0.231 | 4.716 | 16,824× |
| P5: Grouped multi-decoder DeepM&Mnet | **5.87M** | 0.926 | **0.492** | **4.066** | **19,514×** |

---

## Artifacts corresponding to the paper

- Code and reported runs: this repository, release tag `sc26-camera-ready`.
  - `surrogate_models/phase1_lstm_mr/` … `phase5_federated_mr/`: the five reported runs (configuration, data split, training history, per-output metrics, run log and report). The per-output-type results of the paper come from their `metrics.csv` and `phase_summary.json`.
  - `surrogate_models/horizon_eval/`: closed-loop rollout, validity horizons and inference timing (produced by `horizon_rollout_speedup.ipynb`).
- Data and model weights: Zenodo, [doi:10.5281/zenodo.19595930](https://doi.org/10.5281/zenodo.19595930): simulations of Summit, Marconi100 and Lassen for the data analysis, and 11-hour and 25-hour systematic Summit simulations for sample model training.
- 720-hour systematic dataset (128 chunks at 1 s, used for the reported runs): not on Zenodo; it will be made available in the future through this repository.
---

## Repository layout

| Path | Contents |
|---|---|
| `fmu2ml/` | Python package: system configuration, FMU input generation and simulation, data analysis, and the `surrogate` sub-package (architectures, data store, training, evaluation). |
| `analysis_notebooks/` | Data analysis per system (`summit/`, `lassen/`, `marconi/`). |
| `surrogate_models/` | Phase notebooks (`phase1_baseline_lstm.ipynb` … `phase5-grouped-deepmmnet.ipynb`), training scripts (`scripts/run_phase1.py` … `run_phase5.py`), the reported runs and the horizon/speedup evaluation. |
| `config/`, `models/`, `raps/` | ExaDigiT system configurations, FMU assets and the RAPS interface. |

---

## Training a phase

Each phase is a short notebook or script that builds its configuration on the shared task and calls `PhaseRun(...).execute()`. Runs are written to `surrogate_models/runs/<RUN_ID>/`, so the reported runs are never overwritten.

```bash
# one GPU
python surrogate_models/scripts/run_phase1.py

# one node, 8 GPUs with DDP (as for the reported runs on OLCF Frontier)
export PHASE_NUM_WORKERS=6
srun -N1 -n8 -c7 --gpus-per-node=8 --gpu-bind=none \
     python surrogate_models/scripts/run_phase1.py
```

Data location, split and training budget are set through `PHASE_*` environment variables (for example `PHASE_DATA_DIR`); the defaults reproduce the paper's task. Before running several phases at once, build the shared data store once, from the repository root:

```bash
python -m fmu2ml.surrogate.phase_common --prewarm
```
