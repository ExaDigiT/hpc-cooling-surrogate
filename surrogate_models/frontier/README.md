# Frontier runs (Simulator_olcf5_base)

Phase 5 grouped DeepM&Mnet on the Frontier cooling FMU: 25 CDUs, inputs are
per-CDU `Q_flow_total` (per cabinet, 3 cabinets per CDU) plus the plant-wide
wet-bulb `Towb`; 275 outputs (11 per CDU). Each directory is a full run report
(`report.md`, `phase_summary.json`, `metrics.csv`, figures, logs) without the
model weights.

All runs use `run_phase5.py`'s architecture and the Summit-derived task spec
(rates 3 s / 30 s, look-backs 300 s / 1200 s, horizons 60 s / 600 s), trained on
one RTX A6000.

## Results

Skill against persistence (1 - MSE_model / MSE_persistence) on the test chunks,
averaged over lead times. "frozen" means loss weight 0 and exact persistence
output, as in the Summit spec.

| Run | Data | G_T | G_V | G_p | G_Vs | G_ps | G_W | Overall | Beats persistence |
|---|---|---|---|---|---|---|---|---|---|
| `frontier_phase5_mr` | 25 h | -0.023 | frozen | +0.008 | frozen | frozen | -0.017 | -0.008 | 27.6% |
| `frontier_phase5_mr_allheads_25h` | 25 h | -0.122 | +0.013 | +0.010 | -0.007 | +0.002 | -0.050 | -0.046 | 41.5% |
| `frontier_phase5_mr_200h` | 201 h | +0.212 | frozen | +0.933 | frozen | frozen | +0.027 | +0.249 | 61.1% |
| `frontier_phase5_mr_allheads_200h` | 201 h | +0.177 | +0.267 | +0.937 | +0.063 | +0.030 | +0.022 | **+0.272** | **86.2%** |

- 25 h (8 chunks, 1,968 train windows) overfits: train loss falls while
  validation loss stays flat or rises. 201 h (24 chunks, 17,974 train windows;
  test chunks 1, 8, 13) removes most of it.
- Training all six groups is best overall, mainly through G_V, at a small cost
  on G_T (+0.212 to +0.177); the groups compete for the shared encoder.
- Temperature R2 is low (about 0.05) for every model and for persistence: the
  systematic data has large load steps whose effect on return temperatures
  appears within 20 s, and inputs after the prediction origin are not visible.

## Why train G_V, G_Vs and G_ps on Frontier

On Summit those groups are persistence-locked. On Frontier, a ridge model on
(last state, own Q_flow history, Towb history) beats persistence on every
output type (test chunk 0 of the 201 h data):

| Output | Group | Skill @ 60 s | Skill @ 600 s |
|---|---|---|---|
| T_prim_s / T_prim_r / T_sec_s / T_sec_r | G_T | +1.00 / +0.72 / +0.78 / +0.67 | +0.96 / +0.35 / +0.35 / +0.36 |
| V_flow_prim | G_V | +0.97 | +0.37 |
| V_flow_sec | G_Vs | +0.86 | +0.37 |
| p_prim_s / p_prim_r | G_p | +0.99 / +0.99 | +0.56 / +0.79 |
| p_sec_s / p_sec_r | G_ps | +0.65 / +0.66 | +0.35 / +0.35 |
| W_flow | G_W | +0.67 | +0.36 |

The ridge scores one lead on one chunk while the surrogate averages all leads
over three chunks, so the comparison is indicative. It shows headroom on G_T,
G_Vs, G_ps and G_W.

Other measurements: all groups are band-limited at a 30 s rate (mean variance
above Nyquist at most 0.13%); return temperatures, the secondary loop and pump
power follow Q_flow within 10-20 s; Towb reaches the primary side after 11-17 min.

## Reproducing

From the repository root, with absolute paths:

```bash
# inputs (201 h, 24 chunks), then FMU outputs (1 s steps, warm-up up to 4 h)
python -m fmu2ml.data.generators.systematic_chunk_generator \
    --system-name frontier --duration-hours 200 --chunks 24 --output $DATA --save-raw
python -m fmu2ml.data.generators.parallel_output_generator \
    --data-dir $DATA --system-name frontier --parallel 24 \
    --stabilization-hours 4 --memory-limit 4096

# store, then training
export PHASE_SYSTEM=frontier PHASE_DATA_DIR=$DATA PHASE_N_CHUNKS=24
python -m fmu2ml.surrogate.phase_common --prewarm
FRONTIER_TRAIN_ALL_HEADS=1 FRONTIER_RUN_ID=frontier_phase5_mr_allheads_200h \
    python surrogate_models/scripts/run_frontier_phase5.py
```

The 201 h output generation took about 3 h with 24 workers; each training run
took 12-16 min.
