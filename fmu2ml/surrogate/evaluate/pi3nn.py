"""
PI3NN training, calibration, and evaluation for the federated surrogate.

Pipeline (see ``architecture/pi3nn.py`` for the model):

1. ``train_pi3nn_bounds`` — freeze the mean model, fit the two non-negative
   deviation heads on the positive / negative residuals of the frozen model.
2. ``calibrate_pi3nn`` — per-output coefficients ``c_u, c_l`` so the interval
   ``[f - c_l·l, f + c_u·u]`` attains the target central coverage. For this
   monotone, single-variable problem the PI3NN root-finding has a closed form: a
   per-output empirical quantile of the residual-to-deviation ratio (fully
   vectorised across the ~2,827 outputs).
3. ``evaluate_pi3nn`` — assemble intervals on the test set and report standard
   metrics (on the mean) plus interval calibration, into the shared
   :class:`UQResults` surface so MC dropout and PI3NN are directly comparable.

Calibrate on a held-out split (``calibration_loader``); using the test set is
supported but flagged as in-sample.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .evaluator import SurrogateEvaluator
from .uncertainty import UQResults, compute_interval_columns, _nominal_z
from ..architecture.pi3nn import FederatedPI3NN


# ───────────────────────────────────────────────────────────────────────
# Training
# ───────────────────────────────────────────────────────────────────────

def train_pi3nn_bounds(
    pi3nn: FederatedPI3NN,
    train_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Fit the PI3NN deviation heads on the frozen mean model's residuals.

    Upper head learns positive residuals ``max(y - f, 0)``; lower head learns the
    magnitude of negative residuals ``max(f - y, 0)``. The mean model is frozen,
    so this is cheap relative to training the surrogate.
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pi3nn = pi3nn.to(device)
    pi3nn.freeze_mean()
    opt = torch.optim.Adam(pi3nn.bound_parameters(), lr=lr, weight_decay=weight_decay)
    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}
    eps = 1e-6

    def _step(batch, train: bool):
        u = batch['u_hist'].to(device, non_blocking=True)
        y = batch['y_hist'].to(device, non_blocking=True)
        target = batch['y_delta'].to(device, non_blocking=True)
        out = pi3nn(u, y)
        r = target - out['mean']                  # residual (norm. delta space)
        pos = (r > 0).float()
        neg = (r < 0).float()
        loss_up = ((out['upper_dev'] - r) ** 2 * pos).sum() / (pos.sum() + eps)
        loss_lo = ((out['lower_dev'] + r) ** 2 * neg).sum() / (neg.sum() + eps)
        loss = loss_up + loss_lo
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach())

    for ep in range(epochs):
        pi3nn.upper.train(); pi3nn.lower.train()
        losses = [_step(b, train=True) for b in train_loader]
        tr = float(np.mean(losses)) if losses else float('nan')
        history['train_loss'].append(tr)

        vl = float('nan')
        if val_loader is not None:
            pi3nn.upper.eval(); pi3nn.lower.eval()
            with torch.no_grad():
                vlosses = [_step(b, train=False) for b in val_loader]
            vl = float(np.mean(vlosses)) if vlosses else float('nan')
            history['val_loss'].append(vl)

        if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
            msg = f"[PI3NN] epoch {ep+1}/{epochs}  train={tr:.5g}"
            if val_loader is not None:
                msg += f"  val={vl:.5g}"
            print(msg)

    return history


# ───────────────────────────────────────────────────────────────────────
# Collection + calibration
# ───────────────────────────────────────────────────────────────────────

def _collect_pi3nn(
    pi3nn: FederatedPI3NN,
    loader: DataLoader,
    normalizer: Any,
    column_info: Dict,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run the PI3NN model over a loader and return absolute-space mean, per-unit
    (c=1) upward/downward deviations, and the target.
    """
    pi3nn = pi3nn.to(device)
    pi3nn.mean_model.eval(); pi3nn.upper.eval(); pi3nn.lower.eval()

    means, ups, los, lasts, futures, tgt_norm = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            u = batch['u_hist'].to(device, non_blocking=True)
            y = batch['y_hist'].to(device, non_blocking=True)
            out = pi3nn(u, y)
            means.append(out['mean'].cpu().numpy())
            ups.append(out['upper_dev'].cpu().numpy())
            los.append(out['lower_dev'].cpu().numpy())
            lasts.append(batch['last_dynamic'].numpy())
            futures.append(batch['future_dynamic'].numpy())
            tgt_norm.append(batch['y_delta'].numpy())

    mean_norm = np.concatenate(means)
    up_dev = np.concatenate(ups)
    lo_dev = np.concatenate(los)
    last_dyn = np.concatenate(lasts)
    future_dyn = np.concatenate(futures)
    target_norm = np.concatenate(tgt_norm)

    cols = column_info['dynamic_cols']
    mean_abs = normalizer.inverse_delta(mean_norm, last_dyn, cols)
    upper_full = normalizer.inverse_delta(mean_norm + up_dev, last_dyn, cols)
    lower_full = normalizer.inverse_delta(mean_norm - lo_dev, last_dyn, cols)

    up_dev_abs = np.clip(upper_full - mean_abs, 0.0, None)
    lo_dev_abs = np.clip(mean_abs - lower_full, 0.0, None)

    return {
        'pred_mean': mean_abs.astype(np.float32),
        'pred_absolute': mean_abs.astype(np.float32),
        'up_dev_abs': up_dev_abs.astype(np.float32),
        'lo_dev_abs': lo_dev_abs.astype(np.float32),
        'target_absolute': future_dyn.astype(np.float32),
        'pred_normalized': mean_norm.astype(np.float32),
        'target_normalized': target_norm.astype(np.float32),
        'last_dynamic': last_dyn.astype(np.float32),
    }


def calibrate_pi3nn(
    collected: Dict[str, np.ndarray],
    nominal: float = 0.95,
    eps: float = 1e-6,
    c_max: float = 1e4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-output calibration coefficients ``(c_u, c_l)``.

    Closed-form root-finding: for the upper bound, the fraction of points above
    ``f + c·u`` equals ``P(ratio_u > c)`` where ``ratio_u = (y - f) / u``. Setting
    that tail to ``alpha/2`` gives ``c_u = quantile_{1-alpha/2}(ratio_u)`` —
    computed per output, vectorised. The lower side is symmetric with
    ``ratio_l = (f - y) / l``.
    """
    mean_abs = collected['pred_mean']
    target = collected['target_absolute']
    up = collected['up_dev_abs'] + eps
    lo = collected['lo_dev_abs'] + eps

    tail = (1.0 - nominal) / 2.0
    n_out = mean_abs.shape[2]

    resid = (target - mean_abs).reshape(-1, n_out)
    ratio_u = (resid / up.reshape(-1, n_out))
    ratio_l = (-resid / lo.reshape(-1, n_out))

    c_u = np.clip(np.quantile(ratio_u, 1.0 - tail, axis=0), 0.0, c_max)
    c_l = np.clip(np.quantile(ratio_l, 1.0 - tail, axis=0), 0.0, c_max)
    return c_u.astype(np.float32), c_l.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────
# Evaluation
# ───────────────────────────────────────────────────────────────────────

def evaluate_pi3nn(
    pi3nn: FederatedPI3NN,
    test_loader: DataLoader,
    normalizer: Any,
    column_info: Dict,
    config: Any,
    calibration_loader: Optional[DataLoader] = None,
    nominal_coverage: float = 0.95,
    phase: str = "5",
    device: Optional[torch.device] = None,
    evaluator: Optional[SurrogateEvaluator] = None,
) -> UQResults:
    """
    Evaluate PI3NN prediction intervals on the test set.

    Returns a :class:`UQResults` (``method='pi3nn'``) whose ``metrics_df`` holds
    the standard metrics (on the mean) plus interval calibration columns
    (``PICP_95``, ``MPIW_95``, ``Mean_Sigma`` as effective σ), and whose
    ``predictions_dict`` carries ``pred_lower``/``pred_upper``/``pred_std``.
    """
    evaluator = evaluator or SurrogateEvaluator(device=device)
    device = evaluator.device

    t0 = time.time()
    test = _collect_pi3nn(pi3nn, test_loader, normalizer, column_info, device)
    inference_time = time.time() - t0

    if calibration_loader is not None:
        cal = _collect_pi3nn(pi3nn, calibration_loader, normalizer, column_info, device)
        cal_source = "held-out"
    else:
        cal = test
        cal_source = "in-sample (test)"
    c_u, c_l = calibrate_pi3nn(cal, nominal=nominal_coverage)

    mean_abs = test['pred_mean']
    cu = c_u[None, None, :]
    cl = c_l[None, None, :]
    upper = mean_abs + cu * test['up_dev_abs']
    lower = mean_abs - cl * test['lo_dev_abs']

    znom = _nominal_z(nominal_coverage)
    eff_sigma = (upper - lower) / (2.0 * znom)

    pred_dict = dict(test)
    pred_dict['pred_lower'] = lower.astype(np.float32)
    pred_dict['pred_upper'] = upper.astype(np.float32)
    pred_dict['pred_std'] = eff_sigma.astype(np.float32)

    # Standard metrics on the mean prediction.
    metrics_df = evaluator.metrics_computer.compute(pred_dict, column_info, config, phase)

    # Interval calibration columns (row-aligned to output order).
    icol = compute_interval_columns(
        mean_abs, lower, upper, test['target_absolute'], nominal=nominal_coverage,
    )
    for col in icol.columns:
        metrics_df[col] = icol[col].values

    n_samples = mean_abs.shape[0]
    uncertainty_summary = {
        'method': 'pi3nn',
        'mean_sigma': float(metrics_df['Mean_Sigma'].mean()),
        'picp_95_raw': float(metrics_df['PICP_95'].mean()),
        'picp_95_recal': float(metrics_df['PICP_95'].mean()),  # PI3NN is calibrated
        'mpiw_95_raw': float(metrics_df['MPIW_95'].mean()),
        'mean_nll': float('nan'),
        'recal_scale': 1.0,
        'recal_source': cal_source,
        'nominal_coverage': nominal_coverage,
        'c_u_mean': float(np.mean(c_u)),
        'c_l_mean': float(np.mean(c_l)),
    }

    return UQResults(
        metrics_df=metrics_df,
        predictions_dict=pred_dict,
        inference_time=inference_time,
        n_samples=n_samples,
        overall_summary=evaluator._compute_overall_summary(metrics_df),
        group_summaries=evaluator._compute_group_summaries(metrics_df, phase),
        r2_distribution=evaluator._compute_r2_distribution(metrics_df),
        phase=phase,
        model_name=pi3nn.mean_model.__class__.__name__ + "+PI3NN",
        method='pi3nn',
        nominal_coverage=nominal_coverage,
        recal_scale=1.0,
        uncertainty_summary=uncertainty_summary,
    )


__all__ = [
    'train_pi3nn_bounds',
    'calibrate_pi3nn',
    'evaluate_pi3nn',
]
