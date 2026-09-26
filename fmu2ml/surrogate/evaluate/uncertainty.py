"""
Uncertainty quantification for surrogate models.

Two complementary methods share a single results/metrics surface so they can be
compared on the same calibration diagnostics and plots:

1. Monte-Carlo dropout (this module): keep dropout layers stochastic at
   inference time, run ``n_passes`` forward passes, and summarise the predictive
   distribution as a per-output mean and standard deviation. Works on any of the
   six phases with no architecture change and no retraining — every architecture
   already carries live ``nn.Dropout`` modules.

2. PI3NN (``architecture/pi3nn.py`` + helpers here): distribution-free,
   OOD-aware prediction intervals. PI3NN produces asymmetric ``[lower, upper]``
   bounds rather than a Gaussian ``sigma``; the calibration metrics below accept
   either representation.

Design notes
------------
* Accumulation uses Welford's online algorithm over the *absolute-space*
  predictions, so memory stays at two arrays of shape ``(N, K, n_out)``
  regardless of ``n_passes`` (stacking 100 passes for Summit would be ~5 GB).
* Uncertainty is summarised in physical units (after ``inverse_delta``), so a
  reported ``sigma`` is directly interpretable (°C, GPM, psig, kW).
* The mean prediction is fed through the *existing* ``MetricsComputer``, so every
  current R²/persistence/skill metric is preserved and computed on the ensemble
  mean; uncertainty columns are appended alongside.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluator import EvalResults, SurrogateEvaluator

# Module types whose only train/eval difference is dropout. Flipping just these
# to train() keeps LayerNorm/etc. deterministic (there is no BatchNorm anywhere
# in the surrogate stack, so this is equivalent to model.train() here, but stays
# correct if BatchNorm is added later).
_DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
)
_RNN_TYPES = (nn.LSTM, nn.GRU, nn.RNN)


# ───────────────────────────────────────────────────────────────────────
# Dropout state management
# ───────────────────────────────────────────────────────────────────────

def iter_dropout_modules(model: nn.Module):
    """Yield every dropout-bearing module in ``model``."""
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            yield m
        elif isinstance(m, _RNN_TYPES) and getattr(m, 'dropout', 0):
            yield m


def enable_mc_dropout(model: nn.Module) -> int:
    """
    Put the model in eval mode but re-activate dropout for MC sampling.

    Returns the number of stochastic modules that were enabled. A return value
    of 0 means the model has no usable dropout (MC dropout will be degenerate —
    every pass identical).
    """
    model.eval()  # freeze LayerNorm running behaviour, disable grad-only paths
    n = 0
    for m in iter_dropout_modules(model):
        m.train()
        n += 1
    return n


def set_dropout_p(model: nn.Module, p: float) -> Dict[nn.Module, float]:
    """
    Override the dropout probability on every ``nn.Dropout*`` module.

    Returns a mapping of module → original ``p`` so the change can be undone with
    :func:`restore_dropout_p`. RNN internal dropout is left untouched (it is a
    construction-time argument and not safe to mutate in place).
    """
    originals: Dict[nn.Module, float] = {}
    for m in model.modules():
        if isinstance(m, _DROPOUT_TYPES):
            originals[m] = m.p
            m.p = p
    return originals


def restore_dropout_p(originals: Dict[nn.Module, float]) -> None:
    """Undo :func:`set_dropout_p`."""
    for m, p in originals.items():
        m.p = p


class mc_dropout_mode:
    """
    Context manager: enable MC dropout (optionally with a ``p`` override) and
    restore the model to eval()/original-``p`` on exit.

    Example
    -------
    >>> with mc_dropout_mode(model, p=0.2):
    ...     preds = model(x)
    """

    def __init__(self, model: nn.Module, p: Optional[float] = None):
        self.model = model
        self.p = p
        self._originals: Dict[nn.Module, float] = {}

    def __enter__(self):
        if self.p is not None:
            self._originals = set_dropout_p(self.model, self.p)
        self.n_enabled = enable_mc_dropout(self.model)
        return self.model

    def __exit__(self, *exc):
        if self._originals:
            restore_dropout_p(self._originals)
        self.model.eval()
        return False


# ───────────────────────────────────────────────────────────────────────
# Result container
# ───────────────────────────────────────────────────────────────────────

@dataclass
class UQResults(EvalResults):
    """
    Evaluation results augmented with uncertainty quantification.

    Adds a per-output ``sigma`` (MC dropout) and/or interval bounds (PI3NN) to
    the standard ``EvalResults``. ``metrics_df`` carries the extra uncertainty
    columns so all existing reporting/plotting keeps working.
    """

    method: str = "mc_dropout"          # "mc_dropout" | "pi3nn"
    n_passes: int = 0                   # MC dropout only
    dropout_p: Optional[float] = None   # inference-time p override, if any
    n_stochastic_modules: int = 0
    recal_scale: float = 1.0            # global sigma multiplier for nominal coverage
    nominal_coverage: float = 0.95
    uncertainty_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_sigma(self) -> float:
        col = 'Mean_Sigma'
        return float(self.metrics_df[col].mean()) if col in self.metrics_df else float('nan')

    @property
    def picp(self) -> float:
        col = 'PICP_95'
        return float(self.metrics_df[col].mean()) if col in self.metrics_df else float('nan')


# ───────────────────────────────────────────────────────────────────────
# MC sampling
# ───────────────────────────────────────────────────────────────────────

def _is_prediction_key(key: str) -> bool:
    """Keys in a predictions_dict that vary across stochastic passes."""
    return ('pred' in key) or (key == 'alpha')


def collect_mc_predictions(
    run_pass: Callable[[], Dict[str, np.ndarray]],
    n_passes: int,
    base_seed: Optional[int] = 0,
    progress: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Run ``run_pass`` ``n_passes`` times with dropout active and accumulate the
    predictive mean and standard deviation of the absolute-space predictions.

    ``run_pass`` must return a phase predictions_dict (the same contract as
    ``PredictionCollector.collect_phaseX``). Statistics are accumulated online
    (Welford) over every prediction key; the physical-space ``sigma`` is taken
    from ``pred_absolute``.

    Returns a predictions_dict whose prediction keys hold the *mean* over passes,
    plus ``pred_std`` (per-output absolute-space std) and ``pred_mean`` (alias of
    the averaged ``pred_absolute``).
    """
    if n_passes < 2:
        raise ValueError("MC dropout needs n_passes >= 2 to estimate a std.")

    iterator = range(n_passes)
    if progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc=f"MC dropout ({n_passes} passes)")
        except Exception:
            pass

    mean: Dict[str, np.ndarray] = {}      # running mean per prediction key
    m2_abs: Optional[np.ndarray] = None   # running sum of squared deviations (abs)
    base: Optional[Dict[str, np.ndarray]] = None
    pred_keys: List[str] = []
    count = 0

    for t in iterator:
        if base_seed is not None:
            torch.manual_seed(base_seed + t)
        d = run_pass()
        count += 1

        if base is None:
            base = d
            pred_keys = [k for k in d if _is_prediction_key(k)]
            for k in pred_keys:
                mean[k] = d[k].astype(np.float64).copy()
            m2_abs = np.zeros_like(d['pred_absolute'], dtype=np.float64)
            continue

        # Welford update
        for k in pred_keys:
            x = d[k].astype(np.float64)
            delta = x - mean[k]
            mean[k] += delta / count
            if k == 'pred_absolute':
                m2_abs += delta * (x - mean[k])

    var_abs = m2_abs / count if count > 0 else m2_abs
    std_abs = np.sqrt(np.clip(var_abs, 0.0, None))

    # Assemble: start from a base pass (gives targets/last_* unchanged), then
    # overwrite prediction keys with their across-pass means.
    out = dict(base)
    for k in pred_keys:
        out[k] = mean[k].astype(np.float32)
    out['pred_std'] = std_abs.astype(np.float32)
    out['pred_mean'] = out['pred_absolute']
    return out


# ───────────────────────────────────────────────────────────────────────
# Calibration metrics
# ───────────────────────────────────────────────────────────────────────

def _nominal_z(coverage: float) -> float:
    """Two-sided Gaussian z for a target central coverage (0.95 → 1.96)."""
    return float(NormalDist().inv_cdf(0.5 + coverage / 2.0))


def gaussian_interval_metrics(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    nominal: float = 0.95,
    recal_scale: float = 1.0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Per-output calibration metrics for a Gaussian predictive distribution.

    All inputs are 2-D ``(N, K)`` slices for one output column (or already
    flattened). ``recal_scale`` multiplies ``sigma`` before coverage/width are
    measured (used to report post-recalibration numbers).
    """
    sigma = (pred_std.flatten() * recal_scale) + eps
    err = target.flatten() - pred_mean.flatten()
    z = err / sigma
    absz = np.abs(z)
    znom = _nominal_z(nominal)

    return {
        'Mean_Sigma': float(np.mean(pred_std)),
        'Coverage_1sigma': float(np.mean(absz <= 1.0)),
        'Coverage_2sigma': float(np.mean(absz <= 2.0)),
        'PICP_95': float(np.mean(absz <= znom)),
        'MPIW_95': float(np.mean(2.0 * znom * sigma)),
        'NLL': float(np.mean(0.5 * np.log(2.0 * np.pi * sigma ** 2) + 0.5 * z ** 2)),
        'Calib_Err_95': float(abs(np.mean(absz <= znom) - nominal)),
    }


def fit_recalibration_scale(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    nominal: float = 0.95,
    eps: float = 1e-12,
) -> float:
    """
    Single global multiplier ``s`` such that ``s * sigma`` attains ``nominal``
    central coverage across all outputs.

    ``s = quantile(|z|, nominal) / z_nominal``. Values > 1 mean the raw MC
    dropout intervals are over-confident (typical at low dropout rates).
    """
    sigma = pred_std.flatten() + eps
    absz = np.abs((target.flatten() - pred_mean.flatten()) / sigma)
    emp_q = float(np.quantile(absz, nominal))
    return emp_q / _nominal_z(nominal)


def compute_uncertainty_columns(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    nominal: float = 0.95,
    recal_scale: float = 1.0,
) -> pd.DataFrame:
    """
    Build a per-output DataFrame of calibration metrics, row-aligned to the
    ``MetricsComputer`` output order (output column ``i`` ↔ row ``i``).

    Shapes: ``pred_mean``/``pred_std``/``target`` are ``(N, K, n_out)``.
    """
    n_out = pred_mean.shape[2]
    rows = []
    for i in range(n_out):
        rows.append(gaussian_interval_metrics(
            pred_mean[:, :, i], pred_std[:, :, i], target[:, :, i],
            nominal=nominal, recal_scale=recal_scale,
        ))
    return pd.DataFrame(rows)


def interval_metrics(
    pred_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    nominal: float = 0.95,
) -> Dict[str, float]:
    """
    Per-output calibration metrics for an explicit ``[lower, upper]`` interval
    (PI3NN). Reports an *effective* sigma ``(upper - lower) / (2·z_nominal)`` so
    interval methods plug into the same plots/columns as the Gaussian path.
    """
    lo, hi, t = lower.flatten(), upper.flatten(), target.flatten()
    inside = (t >= lo) & (t <= hi)
    width = hi - lo
    znom = _nominal_z(nominal)
    return {
        'Mean_Sigma': float(np.mean(width) / (2.0 * znom)),
        'PICP_95': float(np.mean(inside)),
        'MPIW_95': float(np.mean(width)),
        'Calib_Err_95': float(abs(np.mean(inside) - nominal)),
        'Mean_Width': float(np.mean(width)),
    }


def compute_interval_columns(
    pred_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target: np.ndarray,
    nominal: float = 0.95,
) -> pd.DataFrame:
    """Per-output interval metrics, row-aligned to ``MetricsComputer`` order."""
    n_out = pred_mean.shape[2]
    rows = [
        interval_metrics(pred_mean[:, :, i], lower[:, :, i], upper[:, :, i],
                         target[:, :, i], nominal=nominal)
        for i in range(n_out)
    ]
    return pd.DataFrame(rows)


# ───────────────────────────────────────────────────────────────────────
# Orchestration
# ───────────────────────────────────────────────────────────────────────

def evaluate_mc_dropout(
    model: torch.nn.Module,
    test_loader,
    normalizer: Any,
    column_info: Dict,
    config: Any,
    phase: Optional[str] = None,
    n_passes: int = 50,
    dropout_p: Optional[float] = None,
    nominal_coverage: float = 0.95,
    recalibrate: bool = True,
    calibration_loader=None,
    base_seed: Optional[int] = 0,
    device: Optional[torch.device] = None,
    evaluator: Optional[SurrogateEvaluator] = None,
) -> UQResults:
    """
    Monte-Carlo dropout evaluation for any surrogate phase.

    Parameters
    ----------
    model, test_loader, normalizer, column_info, config, phase
        Same as :meth:`SurrogateEvaluator.evaluate`. ``phase`` is auto-detected
        if ``None``.
    n_passes
        Number of stochastic forward passes (typically 30–100).
    dropout_p
        Optional inference-time dropout probability. ``None`` uses the rate the
        model was trained with (standard MC dropout). A larger value yields wider
        (but more biased) intervals — useful when the trained rate is small.
    nominal_coverage
        Target central coverage for PICP/MPIW (default 95%).
    recalibrate
        If True, fit a single global ``sigma`` multiplier so coverage hits
        ``nominal_coverage``. Fitted on ``calibration_loader`` if given, otherwise
        on the test set (reported, with a note that this is in-sample).
    calibration_loader
        Optional held-out loader for fitting the recalibration scale.

    Returns
    -------
    UQResults
        ``metrics_df`` carries the standard metrics (computed on the MC mean)
        plus per-output uncertainty columns; ``predictions_dict`` includes
        ``pred_std``.
    """
    evaluator = evaluator or SurrogateEvaluator(device=device)
    device = evaluator.device
    model = model.to(device)
    phase = phase or evaluator._detect_phase(model, column_info)

    collector = evaluator.collector
    originals: Dict[nn.Module, float] = {}

    def _run_pass_on(loader) -> Dict[str, np.ndarray]:
        return evaluator._collect_predictions(
            model, loader, normalizer, column_info, config, phase
        )

    n_enabled = sum(1 for _ in iter_dropout_modules(model))

    try:
        # Route the collectors through MC-dropout inference mode.
        collector.mc_dropout = True
        collector.mc_dropout_p = dropout_p
        if dropout_p is not None:
            originals = set_dropout_p(model, dropout_p)

        import time
        t0 = time.time()
        mc = collect_mc_predictions(
            lambda: _run_pass_on(test_loader),
            n_passes=n_passes, base_seed=base_seed,
        )
        inference_time = time.time() - t0

        # Fit recalibration scale (held-out if available, else in-sample).
        recal_scale = 1.0
        recal_source = "none"
        if recalibrate:
            if calibration_loader is not None:
                cal = collect_mc_predictions(
                    lambda: _run_pass_on(calibration_loader),
                    n_passes=n_passes, base_seed=base_seed,
                )
                recal_scale = fit_recalibration_scale(
                    cal['pred_mean'], cal['pred_std'], cal['target_absolute'],
                    nominal=nominal_coverage,
                )
                recal_source = "held-out"
            else:
                recal_scale = fit_recalibration_scale(
                    mc['pred_mean'], mc['pred_std'], mc['target_absolute'],
                    nominal=nominal_coverage,
                )
                recal_source = "in-sample (test)"
    finally:
        if originals:
            restore_dropout_p(originals)
        collector.mc_dropout = False
        collector.mc_dropout_p = None
        model.eval()

    # Standard metrics on the MC mean.
    metrics_df = evaluator.metrics_computer.compute(mc, column_info, config, phase)

    # Append per-output uncertainty columns (raw + recalibrated coverage).
    unc_df = compute_uncertainty_columns(
        mc['pred_mean'], mc['pred_std'], mc['target_absolute'],
        nominal=nominal_coverage, recal_scale=1.0,
    )
    recal_df = compute_uncertainty_columns(
        mc['pred_mean'], mc['pred_std'], mc['target_absolute'],
        nominal=nominal_coverage, recal_scale=recal_scale,
    )
    for col in unc_df.columns:
        metrics_df[col] = unc_df[col].values
    metrics_df['PICP_95_recal'] = recal_df['PICP_95'].values
    metrics_df['MPIW_95_recal'] = recal_df['MPIW_95'].values

    n_samples = mc['pred_absolute'].shape[0]
    uncertainty_summary = {
        'method': 'mc_dropout',
        'n_passes': n_passes,
        'dropout_p_override': dropout_p,
        'n_stochastic_modules': n_enabled,
        'mean_sigma': float(metrics_df['Mean_Sigma'].mean()),
        'picp_95_raw': float(metrics_df['PICP_95'].mean()),
        'picp_95_recal': float(metrics_df['PICP_95_recal'].mean()),
        'mpiw_95_raw': float(metrics_df['MPIW_95'].mean()),
        'mean_nll': float(metrics_df['NLL'].mean()),
        'recal_scale': recal_scale,
        'recal_source': recal_source,
        'nominal_coverage': nominal_coverage,
    }

    return UQResults(
        metrics_df=metrics_df,
        predictions_dict=mc,
        inference_time=inference_time,
        n_samples=n_samples,
        overall_summary=evaluator._compute_overall_summary(metrics_df),
        group_summaries=evaluator._compute_group_summaries(metrics_df, phase),
        r2_distribution=evaluator._compute_r2_distribution(metrics_df),
        phase=phase,
        model_name=model.__class__.__name__,
        method='mc_dropout',
        n_passes=n_passes,
        dropout_p=dropout_p,
        n_stochastic_modules=n_enabled,
        recal_scale=recal_scale,
        nominal_coverage=nominal_coverage,
        uncertainty_summary=uncertainty_summary,
    )


def print_uncertainty_summary(results: UQResults, file=None) -> None:
    """Print a compact calibration report for a :class:`UQResults`."""
    def _p(*a, **k):
        print(*a, **k, file=file)

    s = results.uncertainty_summary
    _p("\n" + "=" * 70)
    _p(f"UNCERTAINTY SUMMARY — method={results.method}  phase={results.phase}")
    _p("=" * 70)
    if results.method == 'mc_dropout':
        _p(f"  MC passes:            {s.get('n_passes')}")
        _p(f"  Stochastic modules:   {s.get('n_stochastic_modules')}")
        if s.get('dropout_p_override') is not None:
            _p(f"  Dropout p override:   {s.get('dropout_p_override')}")
    _p(f"  Mean sigma:           {s.get('mean_sigma', float('nan')):.5g}")
    _p(f"  Mean NLL:             {s.get('mean_nll', float('nan')):.4f}")
    nominal = s.get('nominal_coverage', 0.95)
    _p(f"\n  Target coverage:      {nominal:.0%}")
    _p(f"  PICP (raw):           {s.get('picp_95_raw', float('nan')):.3f}")
    _p(f"  PICP (recalibrated):  {s.get('picp_95_recal', float('nan')):.3f}")
    _p(f"  MPIW (raw):           {s.get('mpiw_95_raw', float('nan')):.5g}")
    _p(f"  Recal scale:          {s.get('recal_scale', 1.0):.3f}  "
       f"(fit on {s.get('recal_source', 'n/a')})")
    _p("=" * 70)


__all__ = [
    'UQResults',
    'enable_mc_dropout',
    'iter_dropout_modules',
    'set_dropout_p',
    'restore_dropout_p',
    'mc_dropout_mode',
    'collect_mc_predictions',
    'gaussian_interval_metrics',
    'fit_recalibration_scale',
    'compute_uncertainty_columns',
    'interval_metrics',
    'compute_interval_columns',
    'evaluate_mc_dropout',
    'print_uncertainty_summary',
]
