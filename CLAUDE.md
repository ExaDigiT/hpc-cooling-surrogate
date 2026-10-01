# hpc-cooling-surrogate

Deep-learning surrogates for ExaDigiT's FMU cooling models (SC26 fmu2ml paper:
Summit, five architectures). The `frontier` branch adds Frontier
(`Simulator_olcf5_base.fmu`) and the export path into RAPS.

To build a surrogate for a new system, follow `docs/new-system-surrogate.md`
(configure, generate, validate, analyze, train, export, check in RAPS).

## Layout

- `fmu2ml/`: package. `config/fmu_io.py` (FMU input layout per system),
  `data/generators/` (systematic inputs, parallel FMU runs), `surrogate/`
  (architectures, store, training; `phase_common.py` holds the shared task spec).
- `raps/`: a bundled copy of an old RAPS, used for FMU simulation. Not the real RAPS.
- `config/<system>/`: system, power, scheduler, cooling JSON. `models/`: FMUs. The
  POWER9 FMUs are tracked; the Frontier FMU is not (gitignored, not cleared for
  public release): copy it to `models/frontier/Simulator_olcf5_base.fmu`.
- `surrogate_models/scripts/`: `run_phase{1..5}.py`, `run_frontier_phase5.py`,
  `validate_dataset.py`, `task_spec_analysis.py`, `export_for_raps.py`.
- `surrogate_models/<run_id>/`, `surrogate_models/frontier/<run_id>/`: reported
  runs without weights. `surrogate_models/runs/`: new runs (gitignored, has `model.pt`).
- Data lives outside the repo: `<root>/<system>/data/systematic-*` by default
  (`PHASE_DATA_DIR` overrides); Frontier data is in `~/exa/data/frontier/`.

## Environment

- Run from the repository root: the bundled `raps/` resolves `config/` from the
  working directory.
- `~/exa/raps/.venv` has torch and fmpy but not `psutil`, `scikit-learn`, `dask`,
  `statsmodels`; install `fmu2ml/requirements.txt` somewhere on `PYTHONPATH`.
- Training settings come from `PHASE_*` env vars: `PHASE_SYSTEM` (default
  summit), `PHASE_DATA_DIR`, `PHASE_N_CHUNKS`, `PHASE_GLOBAL_BATCH`, ...
  Build the store once with `python -m fmu2ml.surrogate.phase_common --prewarm`.

## Conventions

- Headline metric is skill against persistence, not R2.
- Summit is the reference task: changes to shared code must keep Summit output
  identical (check `task_kwargs()` and generator output before/after).
- Never commit model weights (`model.pt`), normalizers or exported bundles.
- Rates, look-backs, horizons and head weights in `task_kwargs()` are
  Summit-derived; set them per system from `task_spec_analysis.py`.

## Gotchas

- `parallel_output_generator.py` once passed `step_size` and
  `stabilization_threshold` swapped (fixed in e2dbf9b): datasets made with its
  CLI before that ran at 0.1 s steps. Check the `time` column of any dataset.
- RAPS cannot import `fmu2ml` (the bundled `raps/` shadows the real package);
  RAPS loads surrogates as TorchScript bundles from `export_for_raps.py`.
- Systematic data never holds inputs constant, so the Frontier surrogate drifts
  after 15-30 min of steady RAPS operation. New data should include long
  steady holds.
- The generated input range must cover the RAPS idle load (idle heat x cooling
  efficiency / racks per CDU), or RAPS inputs get clipped.
- A job started from an interactive session can be stopped when the session
  ends; run multi-hour generation detached (`setsid nohup`) or in tmux.
