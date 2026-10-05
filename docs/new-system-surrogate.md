# Building a cooling surrogate for a new system

How the Frontier surrogate (`Simulator_olcf5_base.fmu`, 25 CDUs) was built and
put into RAPS, written so the same steps work for another system's FMU. Paths
assume this repository at `~/exa/hpc-cooling-surrogate` and RAPS at `~/exa/raps`;
commands run from the repository root unless noted. Frontier numbers are given
as reference points; measure your own system rather than copying them.

The steps: configure the system, smoke-test the FMU, generate data, validate it,
analyze it to set the task spec, train, export, and check the result in RAPS
against the FMU.

## 0. Environment

The `raps` venv (`~/exa/raps/.venv`) has torch and fmpy but lacks `psutil`,
`scikit-learn`, `dask` and `statsmodels`. Install `fmu2ml/requirements.txt` into
a venv for this repo, or into a side directory on `PYTHONPATH`:

```bash
~/exa/raps/.venv/bin/python -m pip install --target ~/pydeps psutil scikit-learn "dask[distributed]" statsmodels
export PYTHONPATH=$PWD:~/pydeps
```

Run from the repository root: the bundled `raps/` package resolves `config/`
against the working directory.

## 1. Configure the system

Create `config/<system>/` with `system.json`, `power.json`, `scheduler.json` and
`cooling.json` (copy `config/frontier/`). What matters for the surrogate:

| Key | File | Notes |
|---|---|---|
| `NUM_CDUS`, `RACKS_PER_CDU` | system.json | must match the FMU's `computeBlock_<i>` count |
| `MIN_POWER`, `MAX_POWER` | system.json | per-CDU heat in kW at idle and peak; sets the input range of the generated data |
| `MIN_NORMAL_LOAD`, `MAX_NORMAL_LOAD` | system.json | fraction of the range for "normal" scenarios (0.4 / 0.6) |
| `FMU_PATH` | cooling.json | e.g. `models/<system>/<name>.fmu`; check whether the FMU may be published before committing it (see below) |
| `TEMPERATURE_KEYS` | cooling.json | every FMU temperature input: per-CDU `..._T_Air` (POWER9 FMUs only) and the plant-wide ambient input |

The POWER9 FMUs (Summit, Lassen, Marconi100) are public in this repository. The
Frontier FMU is not: it is gitignored at `models/frontier/`, so place
`Simulator_olcf5_base.fmu` there yourself (it ships with RAPS's FMU models,
locally `~/exa/raps/models/fmu-models/Frontier/`; ask the ExaDigiT team for
access). Treat any new system's FMU the same way until its release is cleared.

List the FMU's inputs first; they decide `TEMPERATURE_KEYS`:

```python
from fmpy import read_model_description
md = read_model_description("models/<system>/<name>.fmu")
print([v.name for v in md.modelVariables if v.causality == "input"])
```

Input layouts differ: Summit, Lassen and Marconi100 take per-CDU `Q_flow_total`
and `T_Air` plus `..._sources_T_ext`; Frontier takes per-CDU `Q_flow_total` and
`..._sources_Towb`, with no `T_Air`. `fmu2ml/config/fmu_io.py` (`FMUInputSpec`)
derives the layout from the config, so nothing else needs editing for a new
layout of the same kind.

`Q_flow_total` is per cabinet: RAPS sends `cdu_power * COOLING_EFFICIENCY /
RACKS_PER_CDU`, and the generators divide by `RACKS_PER_CDU` the same way.
Compute `MIN_POWER` / `MAX_POWER` with the RAPS power model (all nodes idle, then
all at full CPU and GPU utilization; median of `PowerManager.compute_rack_power()`
per CDU). That method reproduces Summit's configured 13.8226 / 49.3232 kW;
Frontier is 287.6 / 1135.8 kW.

Set `MIN_POWER` slightly below the idle value RAPS will actually send
(idle x `COOLING_EFFICIENCY`): the surrogate clips inputs below its training
range, and RAPS idle at efficiency 0.945 sat just under Frontier's training
minimum.

## 2. Smoke-test the FMU

Run a few hundred steps through `fmu2ml.simulation.fmu_simulator.FMUSimulator`
and time it. Frontier costs about 0.06 s per 1 s step (Summit about 0.5 s), which
sets how much data you can afford. Expect a start-up transient: outputs need
2-4 h of simulated time with constant inputs to settle.

## 3. Generate data

```bash
DATA=/abs/path/<system>/systematic-200
python -m fmu2ml.data.generators.systematic_chunk_generator \
    --system-name <system> --duration-hours 200 --chunks 24 --output $DATA --save-raw
python -m fmu2ml.data.generators.parallel_output_generator \
    --data-dir $DATA --system-name <system> --parallel 24 \
    --stabilization-hours 4 --memory-limit 4096
```

- One chunk per worker: each chunk re-stabilizes once (up to `--stabilization-hours`),
  so fewer, longer chunks waste less. 24 x 8.4 h took about 3 h on 32 cores.
- `--memory-limit` is per worker in MB. The default (32768) cuts the worker
  count to available RAM / 32 GB; Frontier workers used under 1 GB.
- How much: 25 h (1,968 training windows) overfit badly (train loss fell,
  validation did not); 201 h (17,974 windows) worked. Plan for 200 h or more.
- Run long generations detached (`setsid nohup ...`), or inside tmux: a job
  started from an interactive session can be stopped when the session ends.
  `--resume` skips chunks that already have output.

Known gap: the original systematic scenarios never hold the inputs constant
(ambient moves about 0.5 K per 30 s, loads carry noise, and steady-state
segments last 600 s). RAPS holds ambient constant and loads steady for hours,
and the Frontier surrogate drifts after 15-30 min there (it learned that trends
continue). The generator now takes `--steady-state-duration 1800
--t-ext-hold-minutes 60 120 --exact-t-ext-holds` for long exactly-constant
ambient holds; whether this removes the drift in RAPS has not been shown yet.

Limit the ambient range to what the system sees (`--t-ext-range-k MIN MAX`,
default 265-315 K). The Marconi100 FMU produced non-physical states above about
300 K (the primary return no warmer than the supply in 10% of rows at 301-304 K
and 47% above 310 K); 268-298 K removed nearly all of them. Also check that the
inputs RAPS sends are inside the training range: RAPS sends the wet-bulb
temperature as `T_Air` on the POWER9 systems, which the generated `T_Air` did
not cover for Marconi100 (see `surrogate_models/marconi100/README.md`).

## 4. Validate

```bash
python surrogate_models/scripts/validate_dataset.py $DATA
```

Checks row counts, 1 s steps, missing outputs, NaN/inf, that the inputs the FMU
recorded equal the generated inputs, value ranges, and the temperature-ordering
and heat-exchanger physics checks. Exits 1 on failure. Always look at the time
step: a generator bug once produced 0.1 s steps with no error (fixed, see the
gotchas in `CLAUDE.md`).

## 5. Analyze: set the task spec from this system's data

```bash
python surrogate_models/scripts/task_spec_analysis.py $DATA --test-chunk 0
```

| Measurement | Decides | Frontier result |
|---|---|---|
| Variance above Nyquist per rate | sampling rate per group: slowest rate keeping it under ~1% | all groups band-limited at 30 s; 3 s hydraulics kept |
| Response lag to Q_flow and ambient | look-back must cover the longest lag | Q_flow 10-20 s (primary flow 210 s); ambient 11-17 min, inside 1200 s |
| Ridge vs persistence skill at 60 s / 600 s | which groups to train vs freeze at persistence | every output learnable, including G_V, G_Vs, G_ps |

The Summit spec in `fmu2ml/surrogate/phase_common.py` (`task_kwargs`) freezes
G_V, G_Vs and G_ps (`head_loss_weights` 0, exact persistence output). That held
on Summit and not on Frontier, so decide per system from step 3 of the analysis:
freeze a group only if the ridge model cannot beat persistence on it.

## 6. Train

```bash
export PHASE_SYSTEM=<system> PHASE_DATA_DIR=$DATA PHASE_N_CHUNKS=24
python -m fmu2ml.surrogate.phase_common --prewarm          # store + shared normalizer, once
FRONTIER_TRAIN_ALL_HEADS=1 FRONTIER_RUN_ID=<system>_phase5_mr_allheads \
    python surrogate_models/scripts/run_frontier_phase5.py
```

`PHASE_SYSTEM` sets the CDU count and input columns; rates, look-backs, horizons
and head weights still come from the Summit spec in `task_kwargs()` unless
overridden. `run_frontier_phase5.py` asserts Frontier's 25 CDUs; copy it for a
new system and set the head weights from step 5. A Phase 5 run on 200 h took
12-16 min on one RTX A6000.

Read `surrogate_models/runs/<run_id>/report.md`. The headline metric is skill
against persistence (1 - MSE_model / MSE_persistence), not R2: many channels are
near-persistent, and temperature R2 stays low when large load steps fall inside
the horizon. Compare per group with the ridge skills from step 5; a model below
the ridge baseline has headroom. Frontier, 201 h, all heads: overall skill
+0.272, 86.2% of outputs beat persistence (`surrogate_models/frontier/README.md`).

To report a run, copy it to `surrogate_models/<system>/<run_id>/` without
`model.pt` and `normalizer.json`; `surrogate_models/runs/` is gitignored.

## 7. Export to RAPS

RAPS cannot import `fmu2ml` (it bundles a top-level `raps` package that would
shadow the real one), so export a self-contained bundle:

```bash
python surrogate_models/scripts/export_for_raps.py \
    --run surrogate_models/runs/<run_id> --store $DATA/store \
    --out ~/exa/raps/models/surrogates/<bundle_name>
```

This writes `model.ts` (TorchScript, checked equal to the eager model) and
`bundle.json` (columns, rates, normalization, input ranges, start-up
initializer). Bundles are not committed in either repository.

In RAPS (branch `cooling-surrogate`), point the system's cooling config at it
and run with `--cooling-model surrogate`:

```yaml
cooling:
  surrogate_path: "../models/surrogates/<bundle_name>"
```

A system without its own FMU can borrow another's as a scaled design with
`fmu_num_cdus` and `heat_scale` (model peak / system peak heat per CDU); see
`config/lux.yaml` in RAPS.

## 8. Check against the FMU in RAPS

```bash
cd ~/exa/raps
for m in fmu surrogate; do
  python main.py run --system <system> -w benchmark -c --cooling-model $m \
      -t 1h --time-delta 15 --noui --seed 1 -o /tmp/cmp_$m < /dev/null
done
python scripts/compare_cooling_models.py /tmp/cmp_fmu /tmp/cmp_surrogate --plot /tmp/cmp.png
```

- Use `--time-delta 15` for the FMU: at the default 1 s tick RAPS advances the
  FMU by `power_update_freq` (15 s) per tick, so FMU time runs 15x fast.
- Pick a workload whose load sits inside the training range; the default random
  workload can leave a small system near idle, below it.
- `-o` stops at a prompt if the directory exists; use a fresh one.
- Frontier result: within about 1-2 C on supply temperatures for the first
  15-30 min, then linear drift (step 3's known gap).
