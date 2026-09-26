"""
Surrogate model evaluation with phase-aware prediction collection and metrics computation.

Supports all 6 phases:
- Phase 1: Baseline LSTM
- Phase 2: Basic DeepONet
- Phase 3: Hybrid DeepONet
- Phase 4: Domain-specific DeepONet
- Phase 5: Federated DeepMMNet
- Phase 6: Physics-Informed Federated DeepMMNet
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Type aliases
ArrayLike = Union[np.ndarray, torch.Tensor]


# ───────────────────────────────────────────────────────────────────────
# Result Containers
# ───────────────────────────────────────────────────────────────────────

@dataclass
class MetricsSummary:
    """Summary statistics for a group of metrics."""
    mean: float
    median: float
    min: float
    max: float
    std: float
    count: int
    
    @classmethod
    def from_series(cls, series: pd.Series) -> 'MetricsSummary':
        """Create summary from pandas Series."""
        return cls(
            mean=float(series.mean()),
            median=float(series.median()),
            min=float(series.min()),
            max=float(series.max()),
            std=float(series.std()),
            count=len(series),
        )
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mean': self.mean,
            'median': self.median,
            'min': self.min,
            'max': self.max,
            'std': self.std,
            'count': self.count,
        }


@dataclass
class EvalResults:
    """Container for evaluation results."""
    
    # Core results
    metrics_df: pd.DataFrame
    predictions_dict: Dict[str, np.ndarray]
    
    # Timing
    inference_time: float
    n_samples: int
    
    # Summary statistics
    overall_summary: Dict[str, Any] = field(default_factory=dict)
    group_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    r2_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Metadata
    phase: str = "unknown"
    model_name: str = "unknown"
    
    @property
    def ms_per_sample(self) -> float:
        """Inference time in milliseconds per sample."""
        return (self.inference_time / self.n_samples) * 1000 if self.n_samples > 0 else 0.0
    
    @property
    def mean_r2(self) -> float:
        """Mean R² across all outputs."""
        r2_col = 'R²' if 'R²' in self.metrics_df.columns else 'R2'
        return float(self.metrics_df[r2_col].mean())
    
    @property
    def beats_persistence_rate(self) -> float:
        """Fraction of outputs that beat persistence baseline."""
        return float(self.metrics_df['Beats_Persistence'].mean())
    
    def get_metrics_for_output(self, output_name: str) -> Optional[pd.Series]:
        """Get metrics for a specific output column."""
        mask = self.metrics_df['Output'] == output_name
        if mask.any():
            return self.metrics_df[mask].iloc[0]
        return None
    
    def get_metrics_for_group(self, group_name: str) -> pd.DataFrame:
        """Get metrics for a specific group (e.g., 'G_T', 'temperature')."""
        group_col = 'Group' if 'Group' in self.metrics_df.columns else 'Domain'
        if group_col in self.metrics_df.columns:
            return self.metrics_df[self.metrics_df[group_col] == group_name]
        return pd.DataFrame()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'phase': self.phase,
            'model_name': self.model_name,
            'n_samples': self.n_samples,
            'inference_time': self.inference_time,
            'ms_per_sample': self.ms_per_sample,
            'mean_r2': self.mean_r2,
            'beats_persistence_rate': self.beats_persistence_rate,
            'overall_summary': self.overall_summary,
            'group_summaries': self.group_summaries,
            'r2_distribution': self.r2_distribution,
        }


# ───────────────────────────────────────────────────────────────────────
# Prediction Collectors (Phase-specific)
# ───────────────────────────────────────────────────────────────────────

class PredictionCollector:
    """
    Collects predictions from model and converts to absolute values.
    
    Handles phase-specific data formats and inverse transformations.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # MC-dropout routing: when True, inference keeps dropout layers active
        # (see fmu2ml.surrogate.evaluate.uncertainty). Default False ⇒ plain eval().
        self.mc_dropout = False
        self.mc_dropout_p: Optional[float] = None

    def _set_inference_mode(self, model: torch.nn.Module) -> None:
        """Put the model in eval mode, or MC-dropout mode if enabled."""
        if self.mc_dropout:
            from .uncertainty import enable_mc_dropout
            enable_mc_dropout(model)
        else:
            model.eval()

    def collect_phase1(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Collect predictions for Phase 1 (Baseline LSTM).
        
        Returns predictions and targets in absolute space.
        """
        self._set_inference_mode(model)
        all_preds, all_targets, all_last_out, all_future = [], [], [], []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions", leave=False,
                              disable=self.mc_dropout):
                x = batch['x'].to(self.device, non_blocking=True)
                pred = model(x)  # (B, K, n_outputs)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch['y'].numpy())
                all_last_out.append(batch['last_output'].numpy())
                all_future.append(batch['future_output'].numpy())
        
        pred_norm = np.concatenate(all_preds)
        target_norm = np.concatenate(all_targets)
        last_out = np.concatenate(all_last_out)
        future_out = np.concatenate(all_future)
        
        # Convert normalized deltas → absolute values
        output_cols = column_info['output_cols']
        pred_absolute = normalizer.inverse_delta(pred_norm, last_out, output_cols)
        
        return {
            'pred_absolute': pred_absolute,
            'target_absolute': future_out,
            'pred_normalized': pred_norm,
            'target_normalized': target_norm,
            'last_output': last_out,
        }
    
    def collect_phase2_3(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        normalizers: Dict[str, Any],
        column_info: Dict,
        config: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Collect predictions for Phase 2/3 (Basic/Hybrid DeepONet).
        
        Handles separate temporal and algebraic pathways.
        """
        self._set_inference_mode(model)
        temporal_cols = column_info['temporal_cols']
        algebraic_cols = column_info['algebraic_cols']
        output_cols = column_info['output_cols']
        
        all_pred_temporal, all_target_temporal = [], []
        all_pred_algebraic, all_target_algebraic = [], []
        all_last_temporal, all_last_full, all_future_full = [], [], []
        all_alpha = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions", leave=False,
                              disable=self.mc_dropout):
                x_temporal = batch['x_temporal'].to(self.device, non_blocking=True)
                x_algebraic = batch['x_algebraic'].to(self.device, non_blocking=True)

                # Phase 3 (HybridDeepONet) requires four extra batch fields.
                # HybridDataset always emits last_temporal_normalized for both phases,
                # so dispatch on the model class instead of the batch key.
                is_hybrid = 'hybrid' in model.__class__.__name__.lower()
                if is_hybrid and 'last_temporal_normalized' in batch:
                    output = model(
                        x_temporal,
                        x_algebraic,
                        batch['last_temporal_normalized'].to(self.device, non_blocking=True),
                        batch['temporal_mean'].to(self.device, non_blocking=True),
                        batch['temporal_cdu_indices'].to(self.device, non_blocking=True),
                        batch['algebraic_cdu_indices'].to(self.device, non_blocking=True),
                    )
                else:
                    output = model(x_temporal, x_algebraic)

                # Handle different output formats (Phase 3 returns 3-tuple with alpha)
                if isinstance(output, tuple):
                    if len(output) == 3:
                        pred_temporal, pred_algebraic, alpha = output
                    else:
                        pred_temporal, pred_algebraic = output
                        alpha = None
                elif isinstance(output, dict):
                    pred_temporal = output['temporal']
                    pred_algebraic = output['algebraic']
                    alpha = output.get('alpha')
                else:
                    pred_temporal, pred_algebraic = output, None
                    alpha = None
                
                all_pred_temporal.append(pred_temporal.cpu().numpy())
                all_target_temporal.append(batch['y_temporal'].numpy())
                
                if pred_algebraic is not None:
                    all_pred_algebraic.append(pred_algebraic.cpu().numpy())
                    all_target_algebraic.append(batch['y_algebraic'].numpy())
                
                all_last_temporal.append(batch['last_temporal_output'].numpy())
                all_last_full.append(batch['last_output_full'].numpy())
                all_future_full.append(batch['future_output_full'].numpy())
                
                if alpha is not None:
                    all_alpha.append(alpha.cpu().numpy())
        
        pred_temporal_norm = np.concatenate(all_pred_temporal)
        target_temporal_norm = np.concatenate(all_target_temporal)
        last_temporal = np.concatenate(all_last_temporal)
        last_full = np.concatenate(all_last_full)
        future_full = np.concatenate(all_future_full)
        
        # Convert temporal deltas to absolute
        delta_norm = normalizers.get('delta') or normalizers
        temporal_pred_abs = self._inverse_temporal_delta(
            pred_temporal_norm, last_temporal, temporal_cols, delta_norm
        )
        temporal_target_abs = self._inverse_temporal_delta(
            target_temporal_norm, last_temporal, temporal_cols, delta_norm
        )
        
        # Convert algebraic to absolute
        K = pred_temporal_norm.shape[1]
        if all_pred_algebraic:
            pred_algebraic_norm = np.concatenate(all_pred_algebraic)
            target_algebraic_norm = np.concatenate(all_target_algebraic)
            out_norm = normalizers.get('output', normalizers)
            algebraic_pred_abs = self._inverse_algebraic(
                pred_algebraic_norm, algebraic_cols, out_norm
            )
            algebraic_target_abs = self._inverse_algebraic(
                target_algebraic_norm, algebraic_cols, out_norm
            )
        else:
            pred_algebraic_norm = None
            target_algebraic_norm = None
            algebraic_pred_abs = None
            algebraic_target_abs = None
        
        # Reconstruct combined predictions in original output order
        N = pred_temporal_norm.shape[0]
        n_out = len(output_cols)
        pred_combined = np.zeros((N, K, n_out), dtype=np.float32)
        target_combined = np.zeros((N, K, n_out), dtype=np.float32)
        
        temporal_indices = [output_cols.index(c) for c in temporal_cols]
        for i, idx in enumerate(temporal_indices):
            pred_combined[:, :, idx] = temporal_pred_abs[:, :, i]
            target_combined[:, :, idx] = temporal_target_abs[:, :, i]
        
        if algebraic_pred_abs is not None:
            algebraic_indices = [output_cols.index(c) for c in algebraic_cols]
            for i, idx in enumerate(algebraic_indices):
                pred_combined[:, :, idx] = algebraic_pred_abs[:, :, i]
                target_combined[:, :, idx] = algebraic_target_abs[:, :, i]
        
        result = {
            'pred_absolute': pred_combined,
            'target_absolute': target_combined,
            'last_output': last_full,
            'pred_temporal_norm': pred_temporal_norm,
            'target_temporal_norm': target_temporal_norm,
            'temporal_pred_abs': temporal_pred_abs,
            'temporal_target_abs': temporal_target_abs,
        }
        
        if algebraic_pred_abs is not None:
            result.update({
                'pred_algebraic_norm': pred_algebraic_norm,
                'target_algebraic_norm': target_algebraic_norm,
                'algebraic_pred_abs': algebraic_pred_abs,
                'algebraic_target_abs': algebraic_target_abs,
            })
        
        if all_alpha:
            result['alpha'] = np.concatenate(all_alpha)
        
        return result
    
    def collect_phase4(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
        config: Any,
    ) -> Dict[str, np.ndarray]:
        """
        Collect predictions for Phase 4 (Domain-specific DeepONet).
        
        Single domain model evaluation.
        """
        self._set_inference_mode(model)
        all_preds, all_targets, all_last_out, all_future = [], [], [], []
        all_alpha = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions", leave=False,
                              disable=self.mc_dropout):
                x = batch['x'].to(self.device, non_blocking=True)
                
                output = model(x)
                
                # Handle different output formats
                if isinstance(output, dict):
                    pred = output['predictions']
                    alpha = output.get('alpha')
                else:
                    pred = output
                    alpha = None
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(batch['y'].numpy())
                all_last_out.append(batch['last_output'].numpy())
                all_future.append(batch['future_output'].numpy())
                
                if alpha is not None:
                    all_alpha.append(alpha.cpu().numpy())
        
        pred_norm = np.concatenate(all_preds)
        target_norm = np.concatenate(all_targets)
        last_out = np.concatenate(all_last_out)
        future_out = np.concatenate(all_future)
        
        # Convert normalized deltas → absolute values
        output_cols = column_info['output_cols']
        if hasattr(normalizer, 'inverse_transform'):
            pred_absolute = normalizer.inverse_transform(pred_norm, last_out, output_cols)
        else:
            pred_absolute = normalizer.inverse_delta(pred_norm, last_out, output_cols)
        
        result = {
            'pred_absolute': pred_absolute,
            'target_absolute': future_out,
            'pred_normalized': pred_norm,
            'target_normalized': target_norm,
            'last_output': last_out,
        }
        
        if all_alpha:
            result['alpha'] = np.concatenate(all_alpha)
        
        return result
    
    def collect_phase5_6(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
    ) -> Dict[str, np.ndarray]:
        """
        Collect predictions for Phase 5/6 (Federated DeepMMNet).
        
        Handles multi-head outputs with dynamic columns.
        """
        self._set_inference_mode(model)
        if getattr(model, "multi_rate", False):
            return self._collect_phase5_6_multirate(model, loader, normalizer, column_info)
        all_preds, all_targets, all_last_dyn, all_future_dyn = [], [], [], []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions", leave=False,
                              disable=self.mc_dropout):
                u_hist = batch['u_hist'].to(self.device, non_blocking=True)
                y_hist = batch['y_hist'].to(self.device, non_blocking=True)
                
                output = model(u_hist, y_hist)
                
                # Handle output format
                if isinstance(output, dict):
                    predictions = output['predictions']  # (B, K, n_dynamic)
                else:
                    predictions = output
                
                all_preds.append(predictions.cpu().numpy())
                all_targets.append(batch['y_delta'].numpy())
                all_last_dyn.append(batch['last_dynamic'].numpy())
                all_future_dyn.append(batch['future_dynamic'].numpy())
        
        pred_norm = np.concatenate(all_preds)
        target_norm = np.concatenate(all_targets)
        last_dyn = np.concatenate(all_last_dyn)
        future_dyn = np.concatenate(all_future_dyn)
        
        # Convert normalized deltas → absolute values
        dynamic_cols = column_info['dynamic_cols']
        pred_absolute = normalizer.inverse_delta(pred_norm, last_dyn, dynamic_cols)
        
        return {
            'pred_absolute': pred_absolute,
            'target_absolute': future_dyn,
            'pred_normalized': pred_norm,
            'target_normalized': target_norm,
            'last_dynamic': last_dyn,
        }

    # head -> (batch-key suffix, column_info index key, pred key)
    _MR_HEAD_META = {
        'G_T':  ('T',  'temp_indices',         'pred_T'),
        'G_V':  ('V',  'flow_indices',         'pred_V'),
        'G_p':  ('p',  'pressure_indices',     'pred_p'),
        'G_Vs': ('Vs', 'flow_sec_indices',     'pred_Vs'),
        'G_ps': ('ps', 'pressure_sec_indices', 'pred_ps'),
        'G_W':  ('W',  'power_indices',         'pred_W'),
    }

    def _collect_phase5_6_multirate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
    ) -> Dict[str, Any]:
        """
        Collect predictions for the multi-rate federated model, **per group**.

        Each decoder group lives on its own ``(rate, K)`` grid, so there is no
        single combined ``(N, K, n_dynamic)`` tensor. Returns
        ``{'multi_rate': True, 'per_group': {suffix: {pred_absolute, target_absolute,
        pred_normalized, target_normalized, last, rate, cols}}}`` with absolute
        values reconstructed using each group's own delta rate.
        """
        cfg = model.config
        dynamic_cols = column_info['dynamic_cols']
        active_heads = set(getattr(cfg, 'head_outputs', self._MR_HEAD_META))
        groups = {}
        for h, (suf, idx_key, pred_key) in self._MR_HEAD_META.items():
            if h not in active_heads:
                continue
            ci = column_info[idx_key]
            groups[h] = {
                'suf': suf, 'pred_key': pred_key, 'col_idx': ci,
                'cols': [dynamic_cols[i] for i in ci],
                'rate': int(cfg.head_subsample[h]),
                'preds': [], 'targets': [], 'lasts': [],
            }
        with torch.no_grad():
            for batch in tqdm(loader, desc="Collecting predictions", leave=False,
                              disable=self.mc_dropout):
                branch_inputs = {
                    b: (batch[f'u_hist__{b}'].to(self.device, non_blocking=True),
                        batch[f'y_hist__{b}'].to(self.device, non_blocking=True))
                    for b in model.branches
                }
                output = model.forward_multirate(branch_inputs)
                for h, g in groups.items():
                    g['preds'].append(output[g['pred_key']].cpu().numpy())
                    g['targets'].append(batch[f'future_{g["suf"]}'].numpy())
                    g['lasts'].append(batch[f'last_{g["suf"]}'].numpy())

        per_group: Dict[str, Any] = {}
        for h, g in groups.items():
            pred_norm = np.concatenate(g['preds'])        # (N, K_g, n_g) consec deltas
            future = np.concatenate(g['targets'])          # (N, K_g, n_g) absolute
            last = np.concatenate(g['lasts'])              # (N, n_g) absolute
            pred_abs = normalizer.inverse_delta(pred_norm, last, g['cols'], rate=g['rate'])
            per_group[g['suf']] = {
                'pred_absolute': pred_abs,
                'target_absolute': future,
                'pred_normalized': pred_norm,
                'last': last,
                'rate': g['rate'],
                'cols': g['cols'],
            }
        return {'multi_rate': True, 'per_group': per_group}

    # Public, phase-agnostic name: the collector works for ANY model exposing
    # the multi-rate contract (forward_multirate / branches / config), not just
    # the federated phases it was first written for.
    collect_multirate = _collect_phase5_6_multirate

    def _inverse_temporal_delta(
        self,
        pred_norm: np.ndarray,
        last_values: np.ndarray,
        cols: List[str],
        normalizer: Any,
    ) -> np.ndarray:
        """Convert normalized temporal deltas to absolute values."""
        result = np.zeros_like(pred_norm)
        for i, col in enumerate(cols):
            if hasattr(normalizer, 'get_scale'):
                scale = normalizer.get_scale(col)
            elif hasattr(normalizer, 'stats') and col in normalizer.stats:
                scale = normalizer.stats[col].get('scale', 1.0)
            else:
                scale = 1.0
            
            # Cumulative sum of deltas
            result[:, :, i] = last_values[:, i:i+1] + \
                np.cumsum(pred_norm[:, :, i] * scale, axis=1)
        
        return result
    
    def _inverse_algebraic(
        self,
        pred_norm: np.ndarray,
        cols: List[str],
        normalizer: Any,
    ) -> np.ndarray:
        """Convert normalized algebraic predictions to absolute values."""
        result = np.zeros_like(pred_norm)
        
        for step in range(pred_norm.shape[1]):
            for i, col in enumerate(cols):
                if hasattr(normalizer, 'stats') and col in normalizer.stats:
                    s = normalizer.stats[col]
                    mean = s.get('mean', 0.0)
                    std = s.get('std', 1.0)
                else:
                    mean, std = 0.0, 1.0
                
                result[:, step, i] = pred_norm[:, step, i] * std + mean
        
        return result


# ───────────────────────────────────────────────────────────────────────
# Metrics Computer
# ───────────────────────────────────────────────────────────────────────

class MetricsComputer:
    """
    Computes comprehensive per-output metrics.
    
    Metrics include:
    - Primary: MAE, RMSE, R², Variance Ratio, Correlation
    - Delta: Delta_Std_Ratio, Cumulative_Drift
    - Baseline: Persistence_R², Beats_Persistence, Skill_Score
    - Per-step: R²_step{k+1} for each prediction step
    """
    
    EPS = 1e-10  # Small constant to avoid division by zero
    
    def compute(
        self,
        predictions_dict: Dict[str, np.ndarray],
        column_info: Dict,
        config: Any,
        phase: str = "1",
    ) -> pd.DataFrame:
        """
        Compute metrics based on phase.
        
        Args:
            predictions_dict: Dict containing predictions and targets
            column_info: Column metadata
            config: Configuration object
            phase: Phase identifier ("1", "2", "3", "4", "5", "6")
            
        Returns:
            DataFrame with per-output metrics
        """
        # Multi-rate collections are phase-agnostic (per-group grids) — score
        # them the same way regardless of which phase produced them.
        if predictions_dict.get('multi_rate'):
            return self._compute_federated_multirate(
                predictions_dict, column_info, config)

        if phase in ["5", "6"]:
            return self._compute_federated(predictions_dict, column_info, config)
        elif phase == "4":
            return self._compute_domain(predictions_dict, column_info, config)
        elif phase in ["2", "3"]:
            return self._compute_hybrid(predictions_dict, column_info, config)
        else:
            return self._compute_lstm(predictions_dict, column_info, config)
    
    def _compute_lstm(
        self,
        predictions_dict: Dict[str, np.ndarray],
        column_info: Dict,
        config: Any,
    ) -> pd.DataFrame:
        """Compute metrics for Phase 1 (LSTM)."""
        pred = predictions_dict['pred_absolute']
        target = predictions_dict['target_absolute']
        last_out = predictions_dict['last_output']
        pred_norm = predictions_dict['pred_normalized']
        target_norm = predictions_dict['target_normalized']
        output_cols = column_info['output_cols']
        K = self._get_prediction_steps(config, pred)
        
        results = []
        for i, col in enumerate(output_cols):
            metrics = self._compute_single_output_metrics(
                pred[:, :, i], target[:, :, i],
                pred_norm[:, :, i], target_norm[:, :, i],
                last_out[:, i], K
            )
            
            # Add column metadata
            cdu_id = column_info.get('col_to_cdu', {}).get(col, 'Unknown')
            output_type = column_info.get('col_to_type', {}).get(col, 'Unknown')
            category = self._get_category(config, output_type)
            
            metrics.update({
                'Output': col,
                'CDU': cdu_id,
                'Type': output_type,
                'Category': category,
            })
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _compute_hybrid(
        self,
        predictions_dict: Dict[str, np.ndarray],
        column_info: Dict,
        config: Any,
    ) -> pd.DataFrame:
        """Compute metrics for Phase 2/3 (DeepONet/Hybrid)."""
        pred = predictions_dict['pred_absolute']
        target = predictions_dict['target_absolute']
        last_out = predictions_dict['last_output']
        output_cols = column_info['output_cols']
        temporal_cols = column_info.get('temporal_cols', [])
        K = self._get_prediction_steps(config, pred)
        
        results = []
        for i, col in enumerate(output_cols):
            # Get normalized predictions for delta metrics
            if col in temporal_cols:
                t_idx = temporal_cols.index(col)
                pred_norm = predictions_dict.get('pred_temporal_norm', predictions_dict.get('pred_normalized'))
                target_norm = predictions_dict.get('target_temporal_norm', predictions_dict.get('target_normalized'))
                if pred_norm is not None and t_idx < pred_norm.shape[2]:
                    p_norm = pred_norm[:, :, t_idx]
                    t_norm = target_norm[:, :, t_idx]
                else:
                    p_norm = t_norm = None
                pathway = 'temporal'
            else:
                pred_norm = predictions_dict.get('pred_algebraic_norm')
                target_norm = predictions_dict.get('target_algebraic_norm')
                algebraic_cols = column_info.get('algebraic_cols', [])
                if col in algebraic_cols and pred_norm is not None:
                    a_idx = algebraic_cols.index(col)
                    if a_idx < pred_norm.shape[2]:
                        p_norm = pred_norm[:, :, a_idx]
                        t_norm = target_norm[:, :, a_idx]
                    else:
                        p_norm = t_norm = None
                else:
                    p_norm = t_norm = None
                pathway = 'algebraic'
            
            metrics = self._compute_single_output_metrics(
                pred[:, :, i], target[:, :, i],
                p_norm, t_norm,
                last_out[:, i], K
            )
            
            # Add column metadata
            cdu_id = column_info.get('col_to_cdu', {}).get(col, 'Unknown')
            output_type = column_info.get('col_to_type', {}).get(col, 'Unknown')
            category = self._get_category(config, output_type)
            
            metrics.update({
                'Output': col,
                'CDU': cdu_id,
                'Type': output_type,
                'Category': category,
                'Pathway': pathway,
            })
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _compute_domain(
        self,
        predictions_dict: Dict[str, np.ndarray],
        column_info: Dict,
        config: Any,
        domain_name: str = "unknown",
    ) -> pd.DataFrame:
        """Compute metrics for Phase 4 (Domain-specific)."""
        pred = predictions_dict['pred_absolute']
        target = predictions_dict['target_absolute']
        last_out = predictions_dict['last_output']
        pred_norm = predictions_dict['pred_normalized']
        target_norm = predictions_dict['target_normalized']
        output_cols = column_info['output_cols']
        K = self._get_prediction_steps(config, pred)
        
        results = []
        for i, col in enumerate(output_cols):
            metrics = self._compute_single_output_metrics(
                pred[:, :, i], target[:, :, i],
                pred_norm[:, :, i], target_norm[:, :, i],
                last_out[:, i], K
            )
            
            # Add column metadata
            cdu_id = column_info.get('col_to_cdu', {}).get(col, 'Unknown')
            output_type = column_info.get('col_to_type', {}).get(col, 'Unknown')
            is_primary = col in column_info.get('primary_cols', [])
            is_secondary = col in column_info.get('secondary_cols', [])
            
            if is_primary:
                category = 'Primary'
            elif is_secondary:
                category = 'Secondary'
            else:
                category = 'Single'
            
            metrics.update({
                'Output': col,
                'Domain': domain_name,
                'CDU': cdu_id,
                'Type': output_type,
                'Category': category,
            })
            
            results.append(metrics)
        
        return pd.DataFrame(results)
    
    def _compute_federated(
        self,
        predictions_dict: Dict[str, np.ndarray],
        column_info: Dict,
        config: Any,
    ) -> pd.DataFrame:
        """Compute metrics for Phase 5/6 (Federated)."""
        if predictions_dict.get('multi_rate'):
            return self._compute_federated_multirate(
                predictions_dict, column_info, config)
        pred = predictions_dict['pred_absolute']
        target = predictions_dict['target_absolute']
        last_dyn = predictions_dict.get('last_dynamic', predictions_dict.get('last_output'))
        pred_norm = predictions_dict['pred_normalized']
        target_norm = predictions_dict['target_normalized']
        dynamic_cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
        K = self._get_prediction_steps(config, pred)
        
        # Build group lookup from column_info indices
        group_of_idx = self._build_group_lookup(column_info)
        
        results = []
        for i, col in enumerate(dynamic_cols):
            metrics = self._compute_single_output_metrics(
                pred[:, :, i], target[:, :, i],
                pred_norm[:, :, i], target_norm[:, :, i],
                last_dyn[:, i], K
            )
            
            # Add column metadata
            cdu_id = column_info.get('col_to_cdu', {}).get(col, 'Unknown')
            output_type = column_info.get('col_to_type', {}).get(col, 'Unknown')
            group = group_of_idx.get(i, 'Unknown')
            
            # Determine loop classification
            loop = 'primary'
            if group in ['G_Vs', 'G_ps', 'G_W']:
                loop = 'secondary'
            elif output_type in ['T_sec_s_C', 'T_sec_r_C']:
                loop = 'secondary'
            
            metrics.update({
                'Output': col,
                'CDU': cdu_id,
                'Output_Type': output_type,
                'Group': group,
                'Loop': loop,
            })
            
            results.append(metrics)

        return pd.DataFrame(results)

    # head -> (suffix, group name) for multi-rate metric assembly
    _MR_SUFFIX_GROUP = {
        'T': 'G_T', 'V': 'G_V', 'p': 'G_p',
        'Vs': 'G_Vs', 'ps': 'G_ps', 'W': 'G_W',
    }

    def _compute_federated_multirate(
        self,
        predictions_dict: Dict[str, Any],
        column_info: Dict,
        config: Any,
    ) -> pd.DataFrame:
        """Compute per-output metrics for the multi-rate federated model.

        Each decoder group lives on its own ``(rate, K_g)`` grid (no single combined
        tensor), so we iterate the ``per_group`` payload from
        :meth:`_collect_phase5_6_multirate` and reuse the same per-output metric
        routine on each group's arrays. The resulting DataFrame has the same columns
        as the single-rate path, plus ``Rate`` (the group's sampling stride) so plots
        and summaries can tell groups apart.
        """
        per_group = predictions_dict['per_group']
        results = []
        for suf, g in per_group.items():
            group = self._MR_SUFFIX_GROUP.get(suf, 'Unknown')
            pred = g['pred_absolute']          # (N, K_g, n_g)
            target = g['target_absolute']      # (N, K_g, n_g)
            pred_norm = g.get('pred_normalized')
            last = g['last']                   # (N, n_g)
            cols = g['cols']
            rate = int(g.get('rate', 1))
            K = pred.shape[1]
            loop = 'secondary' if group in ('G_Vs', 'G_ps', 'G_W') else 'primary'
            for i, col in enumerate(cols):
                pn = pred_norm[:, :, i] if pred_norm is not None else None
                metrics = self._compute_single_output_metrics(
                    pred[:, :, i], target[:, :, i],
                    pn, None,                  # multi-rate keeps no target_normalized
                    last[:, i], K,
                )
                output_type = column_info.get('col_to_type', {}).get(col, 'Unknown')
                cdu_id = column_info.get('col_to_cdu', {}).get(col, 'Unknown')
                # T_sec_* live in G_T but belong to the secondary loop.
                this_loop = loop
                if output_type in ('T_sec_s_C', 'T_sec_r_C'):
                    this_loop = 'secondary'
                metrics.update({
                    'Output': col,
                    'CDU': cdu_id,
                    'Output_Type': output_type,
                    'Group': group,
                    'Loop': this_loop,
                    'Rate': rate,
                })
                results.append(metrics)
        return pd.DataFrame(results)

    def _compute_single_output_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
        pred_norm: Optional[np.ndarray],
        target_norm: Optional[np.ndarray],
        last_val: np.ndarray,
        K: int,
    ) -> Dict[str, Any]:
        """Compute all metrics for a single output variable."""
        p = pred.flatten()
        t = target.flatten()
        
        # === Primary Accuracy Measures ===
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        
        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + self.EPS))
        
        var_pred = np.var(p)
        var_true = np.var(t)
        variance_ratio = float(var_pred / (var_true + self.EPS))
        
        if np.std(p) > self.EPS and np.std(t) > self.EPS:
            correlation = float(np.corrcoef(p, t)[0, 1])
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # === Delta Prediction Metrics ===
        if pred_norm is not None and target_norm is not None:
            pred_d = pred_norm.flatten()
            targ_d = target_norm.flatten()
            delta_std_ratio = float(np.std(pred_d) / (np.std(targ_d) + self.EPS))
        else:
            delta_std_ratio = np.nan
        
        # Cumulative drift: error at last step vs first step
        if K > 1:
            cumulative_drift = float(
                np.mean(np.abs(pred[:, -1] - target[:, -1])) -
                np.mean(np.abs(pred[:, 0] - target[:, 0]))
            )
        else:
            cumulative_drift = 0.0
        
        # === Persistence Baseline ===
        persistence = np.repeat(last_val[:, np.newaxis], K, axis=1).flatten()
        ss_res_pers = np.sum((t - persistence) ** 2)
        r2_pers = float(1 - ss_res_pers / (ss_tot + self.EPS))
        skill_score = float((r2 - r2_pers) / (1 - r2_pers + self.EPS))
        
        # === Per-Step R² ===
        step_r2 = {}
        for k in range(K):
            pk = pred[:, k]
            tk = target[:, k]
            ss_res_k = np.sum((tk - pk) ** 2)
            ss_tot_k = np.sum((tk - tk.mean()) ** 2)
            step_r2[f'R²_step{k+1}'] = float(1 - ss_res_k / (ss_tot_k + self.EPS))
        
        result = {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'Variance_Ratio': variance_ratio,
            'Correlation': correlation,
            'Delta_Std_Ratio': delta_std_ratio,
            'Cumulative_Drift': cumulative_drift,
            'Persistence_R²': r2_pers,
            'Beats_Persistence': r2 > r2_pers,
            'Skill_Score': skill_score,
        }
        result.update(step_r2)
        
        return result
    
    def _build_group_lookup(self, column_info: Dict) -> Dict[int, str]:
        """Build index-to-group mapping for federated models."""
        group_of_idx = {}
        
        index_keys = [
            ('G_T', 'temp_indices'),
            ('G_V', 'flow_indices'),
            ('G_p', 'pressure_indices'),
            ('G_Vs', 'flow_sec_indices'),
            ('G_ps', 'pressure_sec_indices'),
            ('G_W', 'power_indices'),
        ]
        
        for hname, key in index_keys:
            indices = column_info.get(key, [])
            for idx in indices:
                group_of_idx[idx] = hname
        
        return group_of_idx
    
    def _get_prediction_steps(self, config: Any, pred: np.ndarray) -> int:
        """Get number of prediction steps from config or array shape."""
        if hasattr(config, 'PREDICTION_STEPS'):
            return config.PREDICTION_STEPS
        elif hasattr(config, 'prediction_steps'):
            return config.prediction_steps
        else:
            return pred.shape[1]
    
    def _get_category(self, config: Any, output_type: str) -> str:
        """Get category from config for output type."""
        if hasattr(config, 'get_category'):
            return config.get_category(output_type)
        return 'Unknown'


# ───────────────────────────────────────────────────────────────────────
# Main Evaluator
# ───────────────────────────────────────────────────────────────────────

class SurrogateEvaluator:
    """
    Phase-aware evaluator for surrogate models.
    
    Handles prediction collection, metrics computation, and result summarization
    for all 6 phases of surrogate model development.
    
    Example:
        evaluator = SurrogateEvaluator()
        results = evaluator.evaluate(
            model, test_loader, normalizer, column_info, config
        )
        evaluator.print_summary(results)
    """
    
    # R² distribution thresholds
    R2_THRESHOLDS = [
        (0.99, '≥ 0.99'),
        (0.95, '≥ 0.95'),
        (0.90, '≥ 0.90'),
        (0.80, '≥ 0.80'),
        (0.50, '≥ 0.50'),
        (0.00, '≥ 0.00'),
    ]
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize evaluator.
        
        Args:
            device: Device to run evaluation on. Defaults to CUDA if available.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.collector = PredictionCollector(self.device)
        self.metrics_computer = MetricsComputer()
    
    def evaluate(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
        config: Any,
        phase: Optional[str] = None,
    ) -> EvalResults:
        """
        Evaluate surrogate model on test data.
        
        Args:
            model: Trained surrogate model
            test_loader: Test data loader
            normalizer: Normalizer(s) for inverse transformation
            column_info: Column metadata dict
            config: Configuration object
            phase: Phase identifier ("1"-"6"). If None, auto-detected.
            
        Returns:
            EvalResults containing metrics, predictions, and summaries
        """
        # Detect phase if not provided
        if phase is None:
            phase = self._detect_phase(model, column_info)
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Collect predictions
        inference_start = time.time()
        predictions_dict = self._collect_predictions(
            model, test_loader, normalizer, column_info, config, phase
        )
        inference_time = time.time() - inference_start
        
        # Compute metrics
        metrics_df = self.metrics_computer.compute(
            predictions_dict, column_info, config, phase
        )
        
        # Get sample count (multi-rate has no single combined tensor; read any group)
        if predictions_dict.get('multi_rate'):
            _g = next(iter(predictions_dict['per_group'].values()))
            n_samples = _g['pred_absolute'].shape[0]
        else:
            n_samples = predictions_dict['pred_absolute'].shape[0]
        
        # Compute summaries
        overall_summary = self._compute_overall_summary(metrics_df)
        group_summaries = self._compute_group_summaries(metrics_df, phase)
        r2_distribution = self._compute_r2_distribution(metrics_df)
        
        # Get model name
        model_name = model.__class__.__name__
        
        return EvalResults(
            metrics_df=metrics_df,
            predictions_dict=predictions_dict,
            inference_time=inference_time,
            n_samples=n_samples,
            overall_summary=overall_summary,
            group_summaries=group_summaries,
            r2_distribution=r2_distribution,
            phase=phase,
            model_name=model_name,
        )
    
    def evaluate_mc_dropout(
        self,
        model: torch.nn.Module,
        test_loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
        config: Any,
        phase: Optional[str] = None,
        n_passes: int = 50,
        dropout_p: Optional[float] = None,
        nominal_coverage: float = 0.95,
        recalibrate: bool = True,
        calibration_loader: Optional[DataLoader] = None,
        base_seed: Optional[int] = 0,
    ) -> "EvalResults":
        """
        Monte-Carlo dropout evaluation: run ``n_passes`` stochastic forward passes
        and return predictive mean + per-output uncertainty with calibration
        metrics. See :func:`fmu2ml.surrogate.evaluate.uncertainty.evaluate_mc_dropout`.

        Returns a ``UQResults`` (a subclass of ``EvalResults``) whose
        ``metrics_df`` carries the usual metrics computed on the MC mean plus
        uncertainty columns (``Mean_Sigma``, ``PICP_95``, ``MPIW_95``, ``NLL`` …),
        and whose ``predictions_dict`` includes ``pred_std``.
        """
        from .uncertainty import evaluate_mc_dropout as _mc
        return _mc(
            model, test_loader, normalizer, column_info, config,
            phase=phase, n_passes=n_passes, dropout_p=dropout_p,
            nominal_coverage=nominal_coverage, recalibrate=recalibrate,
            calibration_loader=calibration_loader, base_seed=base_seed,
            evaluator=self,
        )

    def evaluate_multi_domain(
        self,
        domain_models: Dict[str, torch.nn.Module],
        domain_loaders: Dict[str, Dict[str, DataLoader]],
        domain_normalizers: Dict[str, Any],
        domain_columns: Dict[str, Dict],
        domain_configs: Dict[str, Any],
    ) -> Tuple[EvalResults, Dict[str, EvalResults]]:
        """
        Evaluate multiple domain-specific models (Phase 4).
        
        Args:
            domain_models: Dict mapping domain name to model
            domain_loaders: Dict mapping domain name to {'test': DataLoader}
            domain_normalizers: Dict mapping domain name to normalizer
            domain_columns: Dict mapping domain name to column_info
            domain_configs: Dict mapping domain name to config
            
        Returns:
            Tuple of (combined_results, per_domain_results)
        """
        per_domain_results = {}
        all_metrics_dfs = []
        total_inference_time = 0.0
        total_samples = 0
        
        for domain_name in domain_models:
            model = domain_models[domain_name]
            loader = domain_loaders[domain_name]['test']
            normalizer = domain_normalizers[domain_name]
            col_info = domain_columns[domain_name]
            config = domain_configs[domain_name]
            
            # Evaluate single domain
            results = self.evaluate(
                model, loader, normalizer, col_info, config, phase="4"
            )
            
            # Add domain name to metrics
            results.metrics_df['Domain'] = domain_name
            
            per_domain_results[domain_name] = results
            all_metrics_dfs.append(results.metrics_df)
            total_inference_time += results.inference_time
            total_samples += results.n_samples
        
        # Combine metrics
        combined_metrics = pd.concat(all_metrics_dfs, ignore_index=True)
        
        # Compute combined summaries
        overall_summary = self._compute_overall_summary(combined_metrics)
        group_summaries = self._compute_domain_summaries(combined_metrics)
        r2_distribution = self._compute_r2_distribution(combined_metrics)
        
        combined_results = EvalResults(
            metrics_df=combined_metrics,
            predictions_dict={},  # Individual predictions in per_domain_results
            inference_time=total_inference_time,
            n_samples=total_samples,
            overall_summary=overall_summary,
            group_summaries=group_summaries,
            r2_distribution=r2_distribution,
            phase="4",
            model_name="DomainDeepONet (ensemble)",
        )
        
        return combined_results, per_domain_results
    

    def print_summary(
        self,
        results: EvalResults,
        detailed: bool = True,
        file=None,
    ) -> None:
        """
        Print formatted evaluation summary.
        
        Args:
            results: EvalResults from evaluate()
            detailed: Whether to print detailed per-group breakdowns
                    (per-category and per-output-type tables)
            file: File object to print to. Defaults to stdout.
        """
        def _print(*args, **kwargs):
            print(*args, **kwargs, file=file)
        
        metrics_df = results.metrics_df
        r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
        n_total = len(metrics_df)
        
        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        phase_names = {
            "1": "Phase 1: Baseline LSTM",
            "2": "Phase 2: Basic DeepONet",
            "3": "Phase 3: Hybrid DeepONet",
            "4": "Phase 4: Domain-Specific DeepONet",
            "5": "Phase 5: Federated DeepMMNet",
            "6": "Phase 6: Physics-Informed Federated DeepMMNet",
        }
        phase_name = phase_names.get(results.phase, f"Phase {results.phase}")
        
        _print("\n" + "=" * 70)
        _print(f"RESULTS SUMMARY — {phase_name}")
        _print("=" * 70)
        
        # ------------------------------------------------------------------
        # Inference timing
        # ------------------------------------------------------------------
        pred_shape = results.predictions_dict.get('pred_absolute', np.array([])).shape
        _print(f"\nPredictions shape: {pred_shape}")
        _print(
            f"Inference time:    {results.inference_time:.2f}s "
            f"({results.ms_per_sample:.2f} ms/sample)"
        )
        
        # ------------------------------------------------------------------
        # Overall metrics
        # ------------------------------------------------------------------
        _print(f"\n--- All {n_total} Outputs ---")
        _print(f"  Mean R²:             {metrics_df[r2_col].mean():.4f}")
        _print(f"  Median R²:           {metrics_df[r2_col].median():.4f}")
        _print(f"  Min R²:              {metrics_df[r2_col].min():.4f}")
        _print(f"  Max R²:              {metrics_df[r2_col].max():.4f}")
        _print(f"  Std R²:              {metrics_df[r2_col].std():.4f}")
        if 'Variance_Ratio' in metrics_df.columns:
            _print(f"  Mean Variance Ratio: {metrics_df['Variance_Ratio'].mean():.4f}")
        if 'Beats_Persistence' in metrics_df.columns:
            beats = metrics_df['Beats_Persistence'].sum()
            _print(
                f"  Beats Persistence:   {beats}/{n_total} "
                f"({metrics_df['Beats_Persistence'].mean():.1%})"
            )
        if 'Skill_Score' in metrics_df.columns:
            _print(f"  Mean Skill Score:    {metrics_df['Skill_Score'].mean():.4f}")
        
        # ------------------------------------------------------------------
        # Detailed per-group breakdowns
        # ------------------------------------------------------------------
        if detailed:
            # Per-category summary
            if 'Category' in metrics_df.columns:
                _print(f"\n--- Per-Category Performance ---")
                cat_aggs = {r2_col: ['mean', 'median', 'min', 'max', 'count']}
                for col in ('RMSE', 'Variance_Ratio', 'Beats_Persistence'):
                    if col in metrics_df.columns:
                        cat_aggs[col] = 'mean'
                cat_summary = metrics_df.groupby('Category').agg(cat_aggs).round(4)
                _print(cat_summary.to_string())
            
            # Per output-type summary
            if 'Type' in metrics_df.columns:
                _print(f"\n--- Per Output Type ---")
                type_aggs = {r2_col: ['mean', 'median', 'min', 'max']}
                for col in ('RMSE', 'MAE', 'Variance_Ratio',
                            'Correlation', 'Beats_Persistence'):
                    if col in metrics_df.columns:
                        type_aggs[col] = 'mean'
                type_summary = metrics_df.groupby('Type').agg(type_aggs).round(4)
                _print(type_summary.to_string())
            
            # Hook for any other custom detail the subclass wants to add
            self._print_detailed_summary(results, _print)
        
        # ------------------------------------------------------------------
        # R² distribution
        # ------------------------------------------------------------------
        _print(f"\n--- R² Distribution ---")
        for threshold, label in self.R2_THRESHOLDS:
            count = (metrics_df[r2_col] >= threshold).sum()
            _print(f"  R² {label}: {count}/{n_total} ({count/n_total:.1%})")
        count_neg = (metrics_df[r2_col] < 0).sum()
        _print(f"  R² <  0.00: {count_neg}/{n_total} ({count_neg/n_total:.1%})")
        
        # ------------------------------------------------------------------
        # Computational performance
        # ------------------------------------------------------------------
        _print(f"\n--- Computational Performance ---")
        total_params = getattr(results, 'total_params', None)
        if total_params is not None:
            _print(f"  Model Parameters:  {total_params:,}")
        _print(f"  Inference Time:    {results.inference_time:.2f}s")
        _print(f"  Throughput:        {results.ms_per_sample:.2f} ms/sample")
        
        _print("=" * 70)

    # def print_summary(
    #     self,
    #     results: EvalResults,
    #     detailed: bool = True,
    #     file=None,
    # ) -> None:
    #     """
    #     Print formatted evaluation summary.
        
    #     Args:
    #         results: EvalResults from evaluate()
    #         detailed: Whether to print detailed per-group breakdowns
    #         file: File object to print to. Defaults to stdout.
    #     """
    #     def _print(*args, **kwargs):
    #         print(*args, **kwargs, file=file)
        
    #     r2_col = 'R²' if 'R²' in results.metrics_df.columns else 'R2'
        
    #     # Header
    #     phase_names = {
    #         "1": "Phase 1: Baseline LSTM",
    #         "2": "Phase 2: Basic DeepONet",
    #         "3": "Phase 3: Hybrid DeepONet",
    #         "4": "Phase 4: Domain-Specific DeepONet",
    #         "5": "Phase 5: Federated DeepMMNet",
    #         "6": "Phase 6: Physics-Informed Federated DeepMMNet",
    #     }
    #     phase_name = phase_names.get(results.phase, f"Phase {results.phase}")
        
    #     _print("\n" + "=" * 70)
    #     _print(f"RESULTS SUMMARY — {phase_name}")
    #     _print("=" * 70)
        
    #     # Inference timing
    #     _print(f"\nPredictions shape: {results.predictions_dict.get('pred_absolute', np.array([])).shape}")
    #     _print(f"Inference time: {results.inference_time:.2f}s "
    #            f"({results.ms_per_sample:.2f} ms/sample)")
        
    #     # Overall metrics
    #     metrics_df = results.metrics_df
    #     _print(f"\n--- All {len(metrics_df)} Outputs ---")
    #     _print(f"  Mean R²:           {metrics_df[r2_col].mean():.4f}")
    #     _print(f"  Median R²:         {metrics_df[r2_col].median():.4f}")
    #     _print(f"  Min R²:            {metrics_df[r2_col].min():.4f}")
    #     _print(f"  Max R²:            {metrics_df[r2_col].max():.4f}")
    #     _print(f"  Std R²:            {metrics_df[r2_col].std():.4f}")
    #     _print(f"  Beats Persistence: {metrics_df['Beats_Persistence'].sum()}/{len(metrics_df)} "
    #            f"({metrics_df['Beats_Persistence'].mean():.1%})")
    #     _print(f"  Mean Skill Score:  {metrics_df['Skill_Score'].mean():.4f}")
        
    #     if detailed:
    #         self._print_detailed_summary(results, _print)
        
    #     # R² distribution
    #     _print(f"\n--- R² Distribution ---")
    #     for threshold, label in self.R2_THRESHOLDS:
    #         count = (metrics_df[r2_col] >= threshold).sum()
    #         _print(f"  R² {label}: {count}/{len(metrics_df)} ({count/len(metrics_df):.1%})")
    
    def _print_detailed_summary(self, results: EvalResults, _print) -> None:
        """Print detailed per-group summaries based on phase."""
        metrics_df = results.metrics_df
        r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
        
        if results.phase in ["5", "6"]:
            # Per-group summary for federated
            _print(f"\n--- Per Decoder Head Group ---")
            for group in ['G_T', 'G_V', 'G_p', 'G_Vs', 'G_ps', 'G_W']:
                grp = metrics_df[metrics_df['Group'] == group]
                if len(grp) == 0:
                    continue
                _print(f"\n  {group} ({len(grp)} outputs):")
                _print(f"    Mean R²:         {grp[r2_col].mean():.4f}")
                _print(f"    Median R²:       {grp[r2_col].median():.4f}")
                _print(f"    Variance Ratio:  {grp['Variance_Ratio'].mean():.4f}")
                _print(f"    Correlation:     {grp['Correlation'].mean():.4f}")
                _print(f"    Beats Persist:   {grp['Beats_Persistence'].sum()}/{len(grp)} "
                       f"({grp['Beats_Persistence'].mean():.1%})")
        
        elif results.phase == "4":
            # Per-domain summary
            _print(f"\n--- Per Domain ---")
            for domain_name in metrics_df['Domain'].unique():
                dom_df = metrics_df[metrics_df['Domain'] == domain_name]
                r2_mean = dom_df[r2_col].mean()
                status = 'Excellent' if r2_mean > 0.9 else (
                    'Good' if r2_mean > 0.7 else (
                    'Fair' if r2_mean > 0.3 else 'Poor'))
                _print(f"\n  {domain_name.upper()} ({len(dom_df)} outputs, {status}):")
                _print(f"    Mean R²:         {r2_mean:.4f} ± {dom_df[r2_col].std():.4f}")
                _print(f"    Median R²:       {dom_df[r2_col].median():.4f}")
                _print(f"    Min R²:          {dom_df[r2_col].min():.4f}")
                _print(f"    Beats Persist:   {dom_df['Beats_Persistence'].sum()}/{len(dom_df)} "
                       f"({dom_df['Beats_Persistence'].mean():.1%})")
        
        elif results.phase in ["2", "3"]:
            # Per-pathway summary
            _print(f"\n--- Per Pathway ---")
            for pathway in ['temporal', 'algebraic']:
                pw_df = metrics_df[metrics_df['Pathway'] == pathway]
                if len(pw_df) == 0:
                    continue
                _print(f"\n  {pathway.upper()} ({len(pw_df)} outputs):")
                _print(f"    Mean R²:         {pw_df[r2_col].mean():.4f}")
                _print(f"    Median R²:       {pw_df[r2_col].median():.4f}")
                _print(f"    Variance Ratio:  {pw_df['Variance_Ratio'].mean():.4f}")
                _print(f"    Beats Persist:   {pw_df['Beats_Persistence'].sum()}/{len(pw_df)}")
        
        # Per output type (common to all phases)
        type_col = 'Output_Type' if 'Output_Type' in metrics_df.columns else 'Type'
        if type_col in metrics_df.columns:
            _print(f"\n--- Per Output Type ---")
            type_summary = metrics_df.groupby(type_col).agg({
                r2_col: ['mean', 'median', 'min', 'max'],
                'RMSE': 'mean',
                'MAE': 'mean',
                'Variance_Ratio': 'mean',
                'Correlation': 'mean',
                'Beats_Persistence': 'mean',
            }).round(4)
            _print(type_summary.to_string())
    
    def _collect_predictions(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        normalizer: Any,
        column_info: Dict,
        config: Any,
        phase: str,
    ) -> Dict[str, np.ndarray]:
        """Dispatch to phase-specific collector."""
        # Multi-rate models share one phase-agnostic collector (see
        # PredictionCollector.collect_multirate).
        if getattr(model, "multi_rate", False) and hasattr(model, "forward_multirate"):
            return self.collector.collect_multirate(model, loader, normalizer, column_info)

        if phase in ["5", "6"]:
            return self.collector.collect_phase5_6(model, loader, normalizer, column_info)
        elif phase == "4":
            return self.collector.collect_phase4(model, loader, normalizer, column_info, config)
        elif phase in ["2", "3"]:
            # For hybrid, normalizer should be dict with 'delta' and 'output' keys
            if isinstance(normalizer, dict):
                normalizers = normalizer
            else:
                normalizers = {'delta': normalizer, 'output': normalizer}
            return self.collector.collect_phase2_3(model, loader, normalizers, column_info, config)
        else:
            return self.collector.collect_phase1(model, loader, normalizer, column_info)
    
    def _detect_phase(self, model: torch.nn.Module, column_info: Dict) -> str:
        """Auto-detect phase from model type and column_info."""
        model_name = model.__class__.__name__.lower()
        
        if 'physicsinformed' in model_name or 'pi' in model_name:
            return "6"
        elif 'federated' in model_name or 'deepmmnet' in model_name:
            return "5"
        elif 'domain' in model_name:
            return "4"
        elif 'hybrid' in model_name:
            return "3"
        elif 'deeponet' in model_name:
            return "2"
        elif 'lstm' in model_name:
            return "1"
        
        # Infer from column_info
        if 'dynamic_cols' in column_info and 'temp_indices' in column_info:
            return "5"
        elif 'temporal_cols' in column_info and 'algebraic_cols' in column_info:
            return "3"
        
        return "1"
    
    def _compute_overall_summary(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute overall summary statistics."""
        r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
        
        return {
            'n_outputs': len(metrics_df),
            'r2_summary': MetricsSummary.from_series(metrics_df[r2_col]).to_dict(),
            'rmse_mean': float(metrics_df['RMSE'].mean()),
            'mae_mean': float(metrics_df['MAE'].mean()),
            'variance_ratio_mean': float(metrics_df['Variance_Ratio'].mean()),
            'correlation_mean': float(metrics_df['Correlation'].mean()),
            'beats_persistence_count': int(metrics_df['Beats_Persistence'].sum()),
            'beats_persistence_rate': float(metrics_df['Beats_Persistence'].mean()),
            'skill_score_mean': float(metrics_df['Skill_Score'].mean()),
        }
    
    def _compute_group_summaries(
        self,
        metrics_df: pd.DataFrame,
        phase: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-group summary statistics based on phase."""
        r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
        summaries = {}
        
        if phase in ["5", "6"] and 'Group' in metrics_df.columns:
            for group in metrics_df['Group'].unique():
                grp = metrics_df[metrics_df['Group'] == group]
                summaries[group] = {
                    'n_outputs': len(grp),
                    'r2_mean': float(grp[r2_col].mean()),
                    'r2_median': float(grp[r2_col].median()),
                    'variance_ratio_mean': float(grp['Variance_Ratio'].mean()),
                    'beats_persistence_rate': float(grp['Beats_Persistence'].mean()),
                }
        
        elif phase in ["2", "3"] and 'Pathway' in metrics_df.columns:
            for pathway in metrics_df['Pathway'].unique():
                pw = metrics_df[metrics_df['Pathway'] == pathway]
                summaries[pathway] = {
                    'n_outputs': len(pw),
                    'r2_mean': float(pw[r2_col].mean()),
                    'r2_median': float(pw[r2_col].median()),
                    'variance_ratio_mean': float(pw['Variance_Ratio'].mean()),
                    'beats_persistence_rate': float(pw['Beats_Persistence'].mean()),
                }
        
        return summaries
    
    def _compute_domain_summaries(
        self,
        metrics_df: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """Compute per-domain summary statistics for Phase 4."""
        r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
        summaries = {}
        
        if 'Domain' in metrics_df.columns:
            for domain in metrics_df['Domain'].unique():
                dom = metrics_df[metrics_df['Domain'] == domain]
                summaries[domain] = {
                    'n_outputs': len(dom),
                    'r2_mean': float(dom[r2_col].mean()),
                    'r2_std': float(dom[r2_col].std()),
                    'r2_median': float(dom[r2_col].median()),
                    'r2_min': float(dom[r2_col].min()),
                    'variance_ratio_mean': float(dom['Variance_Ratio'].mean()),
                    'beats_persistence_rate': float(dom['Beats_Persistence'].mean()),
                }
        
        return summaries
    
    def _compute_r2_distribution(self, metrics_df: pd.DataFrame) -> Dict[str, int]:
        """Compute R² distribution across thresholds."""
        r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
        distribution = {}
        
        for threshold, label in self.R2_THRESHOLDS:
            count = int((metrics_df[r2_col] >= threshold).sum())
            distribution[label] = count
        
        return distribution


# ───────────────────────────────────────────────────────────────────────
# Convenience Functions
# ───────────────────────────────────────────────────────────────────────

def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    normalizer: Any,
    column_info: Dict,
    config: Any,
    phase: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> EvalResults:
    """
    Convenience function to evaluate a surrogate model.
    
    Args:
        model: Trained surrogate model
        test_loader: Test data loader
        normalizer: Normalizer(s) for inverse transformation
        column_info: Column metadata dict
        config: Configuration object
        phase: Phase identifier ("1"-"6"). If None, auto-detected.
        device: Device to run evaluation on
        
    Returns:
        EvalResults containing metrics, predictions, and summaries
    """
    evaluator = SurrogateEvaluator(device=device)
    return evaluator.evaluate(model, test_loader, normalizer, column_info, config, phase)


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    normalizer: Any,
    column_info: Dict,
    config: Any = None,
    phase: str = "1",
    device: Optional[torch.device] = None,
) -> Dict[str, np.ndarray]:
    """
    Collect predictions from model without computing metrics.
    
    Args:
        model: Trained surrogate model
        loader: Data loader
        normalizer: Normalizer(s) for inverse transformation
        column_info: Column metadata dict
        config: Configuration object (required for phases 2-4)
        phase: Phase identifier
        device: Device to run on
        
    Returns:
        Dict containing predictions and targets in absolute space
    """
    collector = PredictionCollector(device=device)
    model.eval()

    # Multi-rate models are phase-agnostic: every phase's multi-rate variant
    # exposes the same forward_multirate / branches / config contract and is
    # scored per group on its own (rate, K) grid.
    if getattr(model, "multi_rate", False) and hasattr(model, "forward_multirate"):
        return collector.collect_multirate(model, loader, normalizer, column_info)

    if phase in ["5", "6"]:
        return collector.collect_phase5_6(model, loader, normalizer, column_info)
    elif phase == "4":
        return collector.collect_phase4(model, loader, normalizer, column_info, config)
    elif phase in ["2", "3"]:
        normalizers = normalizer if isinstance(normalizer, dict) else {'delta': normalizer, 'output': normalizer}
        return collector.collect_phase2_3(model, loader, normalizers, column_info, config)
    else:
        return collector.collect_phase1(model, loader, normalizer, column_info)


def compute_metrics(
    predictions_dict: Dict[str, np.ndarray],
    column_info: Dict,
    config: Any,
    phase: str = "1",
) -> pd.DataFrame:
    """
    Compute metrics from predictions.
    
    Args:
        predictions_dict: Dict containing predictions and targets
        column_info: Column metadata dict
        config: Configuration object
        phase: Phase identifier
        
    Returns:
        DataFrame with per-output metrics
    """
    computer = MetricsComputer()
    return computer.compute(predictions_dict, column_info, config, phase)


def print_summary(results: EvalResults, detailed: bool = True, file=None) -> None:
    """
    Print evaluation summary.
    
    Args:
        results: EvalResults from evaluate()
        detailed: Whether to print detailed breakdowns
        file: File object to print to
    """
    evaluator = SurrogateEvaluator()
    evaluator.print_summary(results, detailed=detailed, file=file)


def get_r2_distribution(
    metrics_df: pd.DataFrame,
    thresholds: Optional[List[float]] = None,
) -> Dict[str, Tuple[int, float]]:
    """
    Get R² distribution across thresholds.
    
    Args:
        metrics_df: DataFrame with metrics
        thresholds: List of R² thresholds. Defaults to [0.99, 0.95, 0.90, 0.80, 0.50, 0.0]
        
    Returns:
        Dict mapping threshold label to (count, percentage)
    """
    if thresholds is None:
        thresholds = [0.99, 0.95, 0.90, 0.80, 0.50, 0.0]
    
    r2_col = 'R²' if 'R²' in metrics_df.columns else 'R2'
    n_total = len(metrics_df)
    
    distribution = {}
    for threshold in thresholds:
        label = f'≥ {threshold:.2f}'
        count = int((metrics_df[r2_col] >= threshold).sum())
        percentage = count / n_total if n_total > 0 else 0.0
        distribution[label] = (count, percentage)

    return distribution


def compute_multirate_metrics(collected: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Per-group metrics for a multi-rate collection (from ``collect_phase5_6`` when
    ``model.multi_rate``). Each group is scored on its own ``(rate, K)`` grid in
    absolute units (no single combined tensor exists).

    Parameters
    ----------
    collected : dict
        ``{'multi_rate': True, 'per_group': {suffix: {'pred_absolute',
        'target_absolute', 'rate', ...}}}``.

    Returns
    -------
    Dict[str, Dict[str, float]]
        ``{suffix: {'r2', 'mae', 'rmse', 'rate', 'n', 'K'}}``.
    """
    if not collected.get('multi_rate'):
        raise ValueError("compute_multirate_metrics expects a multi-rate collection.")
    out: Dict[str, Dict[str, float]] = {}
    for suf, g in collected['per_group'].items():
        pred = np.asarray(g['pred_absolute'], dtype=np.float64)
        targ = np.asarray(g['target_absolute'], dtype=np.float64)
        p, t = pred.reshape(-1), targ.reshape(-1)
        err = p - t
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        out[suf] = {
            'r2': 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'),
            'mae': float(np.mean(np.abs(err))),
            'rmse': float(np.sqrt(np.mean(err ** 2))),
            'rate': int(g['rate']),
            'K': int(pred.shape[1]),
            'n': int(pred.shape[0]),
        }
    return out

    return distribution