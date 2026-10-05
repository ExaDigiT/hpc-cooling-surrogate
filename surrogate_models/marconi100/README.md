# Marconi100 surrogate

First demonstration model for the Marconi100 FMU (`marconi100.fmu`, 49 CDUs),
trained with the steps in `docs/new-system-surrogate.md`. It runs in RAPS but
is not accurate yet.

## Data

200 h in 24 chunks (`systematic-200h-v2`), 1 s steps, 4 h stabilization:

```bash
python -m fmu2ml.data.generators.systematic_chunk_generator --system-name marconi100 \
    --duration-hours 200 --chunks 24 --output $DATA --save-raw \
    --steady-state-duration 1800 --t-ext-hold-minutes 60 120 --exact-t-ext-holds \
    --t-ext-range-k 268 298
```

With the generator's default ambient range of 265-315 K, the FMU produced
non-physical states at high ambient: the primary return was not warmer than the
supply in 2.7% of rows (47% above 310 K), and the secondary supply was not
cooler than the primary supply in 0.5%. Capping the range at 268-298 K cut
these to 0.03% and 0%. `validate_dataset.py` still reports FAIL on the
remaining 10k rows.

## Run

`marconi100_phase5_mr_v2`: all six output groups trained (the ridge baseline
beat persistence on every group), about 16 minutes on one RTX A6000.

| Group | Beats persistence | Mean skill |
|---|---|---|
| G_T | 50.5% | +0.051 |
| G_V | 40.8% | -0.015 |
| G_p | 0% | -0.102 |
| G_Vs | 44.9% | -0.001 |
| G_ps | 73.5% | +0.001 |
| G_W | 100% | +0.290 |
| overall | 48.6% | +0.025 |

For comparison the Frontier model has overall skill +0.272 and 86% of outputs
beating persistence.

## In RAPS

Exported with `export_for_raps.py` and run with `--cooling-model surrogate`
(4 h benchmark workload, seed 1) against the FMU: `T_prim_s` is about +17 C,
`T_sec_s` about -9 C and primary flow about 5x too high. The cause is not
established. A leading suspect is the `T_Air` input: RAPS sends the wet-bulb
temperature (290 K) as `T_Air` for every CDU, while the training data has
`T_Air` between about 290 and 315 K, so 48 of 50 inputs were clipped at start-up.
The next step is to regenerate with `T_Air` matching what RAPS sends and retrain.
