"""
Unified dataset classes for surrogate models.

Provides:
- LSTMDataset: Phase 1 dataset for baseline LSTM
- HybridDataset: Phase 2/3 dataset with temporal + algebraic pathways
- DomainDataset: Phase 4 dataset for domain-specific models
- FederatedDataset: Phase 5/6 dataset with 6 decoder head groups
- create_surrogate_dataloaders: Convenience function for data loading
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import glob
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

from .normalizer import (
    ZScoreNormalizer,
    DeltaNormalizer,
    InputWhitener,
    DomainNormalizer,
    FederatedNormalizer,
    SurrogateNormalizer,
)
from .column_info import (
    build_column_info,
    build_domain_column_info,
    build_federated_column_info,
    identify_qflow_columns,
    get_column_cdu_indices,
)
from ..architecture.configs import (
    SurrogateConfig,
    LSTMConfig,
    DeepONetConfig,
    HybridDeepONetConfig,
    DomainDeepONetConfig,
    FederatedConfig,
    PhysicsInformedConfig,
)


class LSTMDataset(Dataset):
    """
    Dataset for Phase 1: Baseline LSTM surrogate model.
    
    Provides:
    - Concatenated input + output history as model input
    - Normalized delta targets for prediction
    - Last observed output for absolute reconstruction
    """
    
    def __init__(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        config: LSTMConfig,
        normalizer: SurrogateNormalizer,
        column_info: Dict[str, Any],
    ):
        self.config = config
        self.column_info = column_info
        self.normalizer = normalizer
        
        # Subsample data
        self.input_data = input_data[::config.subsample_factor].astype(np.float32)
        self.output_data = output_data[::config.subsample_factor].astype(np.float32)
        
        # Normalize data
        self.input_normalized = normalizer.input_normalizer.transform(
            self.input_data, column_info['input_cols']
        )
        self.output_normalized = normalizer.output_normalizer.transform(
            self.output_data, column_info['output_cols']
        )
        
        # Compute number of valid samples
        total_required = config.history_steps + config.prediction_steps
        self.n_samples = max(0, len(self.input_data) - total_required)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H = self.config.history_steps
        K = self.config.prediction_steps
        input_end = idx + H
        output_start = input_end
        output_end = output_start + K
        
        # Concatenated input + output history
        u_hist = self.input_normalized[idx:input_end]
        y_hist = self.output_normalized[idx:input_end]
        x = np.concatenate([u_hist, y_hist], axis=1)
        
        # Delta targets
        last_output = self.output_data[input_end - 1]
        future_output = self.output_data[output_start:output_end]
        
        # Normalize deltas
        output_cols = self.column_info['output_cols']
        y_delta = np.zeros_like(future_output)
        for i, col in enumerate(output_cols):
            scale = self.normalizer.delta_normalizer.get_scale(col)
            y_delta[:, i] = (future_output[:, i] - last_output[i]) / scale
        
        return {
            'x': torch.from_numpy(x).float(),
            'y': torch.from_numpy(y_delta).float(),
            'last_output': torch.from_numpy(last_output).float(),
            'future_output': torch.from_numpy(future_output).float(),
        }


class HybridDataset(Dataset):
    """
    Dataset for Phase 2/3: Hybrid DeepONet with temporal + algebraic pathways.
    """
    
    def __init__(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        config: Union[DeepONetConfig, HybridDeepONetConfig],
        input_normalizer: ZScoreNormalizer,
        output_normalizer: ZScoreNormalizer,
        delta_normalizer: DeltaNormalizer,
        input_whitener: Optional[InputWhitener],
        column_info: Dict[str, Any],
        qflow_indices: List[int],
        cdu_ids: Optional[List[int]] = None,
        is_train: bool = True,
    ):
        self.config = config
        self.column_info = column_info
        self.qflow_indices = qflow_indices
        self.cdu_ids = cdu_ids or config.cdu_ids
        self.num_cdus = len(self.cdu_ids)
        self.is_train = is_train
        
        self.algebraic_indices = column_info['algebraic_indices']
        self.temporal_indices = column_info['temporal_indices']
        
        self.algebraic_cdu_map = get_column_cdu_indices(
            column_info['algebraic_cols'], self.cdu_ids
        )
        self.temporal_cdu_map = get_column_cdu_indices(
            column_info['temporal_cols'], self.cdu_ids
        )
        
        # Subsample data
        self.input_data = input_data[::config.subsample_factor].astype(np.float32)
        self.output_data = output_data[::config.subsample_factor].astype(np.float32)
        
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.delta_normalizer = delta_normalizer
        self.input_whitener = input_whitener
        
        # Normalize inputs
        self.input_normalized = input_normalizer.transform(
            self.input_data, column_info['input_cols']
        )
        
        # Apply whitening if enabled
        use_whitening = getattr(config, 'use_input_whitening', False)
        if input_whitener is not None and use_whitening:
            self.input_whitened = input_whitener.transform(self.input_normalized)
        else:
            self.input_whitened = self.input_normalized
        
        # Normalize outputs
        self.output_normalized = output_normalizer.transform(
            self.output_data, column_info['output_cols']
        )
        
        total_required = config.history_steps + config.prediction_steps
        self.n_samples = max(0, len(self.input_data) - total_required)
        
        self.algebraic_cdu_indices = torch.from_numpy(
            np.array(self.algebraic_cdu_map, dtype=np.int64)
        )
        self.temporal_cdu_indices = torch.from_numpy(
            np.array(self.temporal_cdu_map, dtype=np.int64)
        )
        
        if self.n_samples > 0:
            print(f"Dataset: {self.n_samples} samples | "
                  f"Algebraic: {len(self.algebraic_indices)} | "
                  f"Temporal: {len(self.temporal_indices)} | "
                  f"CDUs: {self.num_cdus}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H = self.config.history_steps
        K = self.config.prediction_steps
        input_end = idx + H
        output_start = input_end
        output_end = output_start + K
        output_cols = self.column_info['output_cols']
        
        # Temporal pathway
        x_temporal = self.input_whitened[idx:input_end].copy()
        
        include_output_history = getattr(self.config, 'include_output_history', False)
        if include_output_history:
            output_history_steps = getattr(self.config, 'output_history_steps', H)
            out_hist_start = max(0, input_end - output_history_steps)
            x_output_hist = self.output_normalized[out_hist_start:input_end]
            
            if x_output_hist.shape[0] < H:
                pad_len = H - x_output_hist.shape[0]
                padding = np.zeros((pad_len, x_output_hist.shape[1]), dtype=np.float32)
                x_output_hist = np.concatenate([padding, x_output_hist], axis=0)
            elif x_output_hist.shape[0] > H:
                x_output_hist = x_output_hist[-H:]
            
            temporal_hist = x_output_hist[:, self.temporal_indices]
            x_temporal = np.concatenate([x_temporal, temporal_hist], axis=1)
        
        # Temporal targets
        last_temporal_output = self.output_data[input_end - 1, self.temporal_indices]
        last_temporal_normalized = self.output_normalized[input_end - 1, self.temporal_indices]
        future_temporal = self.output_data[output_start:output_end][:, self.temporal_indices]
        temporal_deltas = future_temporal - last_temporal_output
        
        y_temporal = np.zeros_like(temporal_deltas)
        temporal_cols = self.column_info['temporal_cols']
        for i, col in enumerate(temporal_cols):
            scale = self.delta_normalizer.get_scale(col)
            y_temporal[:, i] = temporal_deltas[:, i] / scale
        
        temporal_mean = np.zeros(len(self.temporal_indices), dtype=np.float32)
        for i, col in enumerate(temporal_cols):
            temporal_mean[i] = self.output_normalizer.stats[col]['mean']
        
        # Algebraic pathway
        qflow_future = self.input_normalized[output_start:output_end][:, self.qflow_indices]
        qflow_current = self.input_normalized[input_end - 1, self.qflow_indices]
        y_algebraic = self.output_normalized[output_start:output_end][:, self.algebraic_indices]
        
        return {
            'x_temporal': torch.from_numpy(x_temporal).float(),
            'y_temporal': torch.from_numpy(y_temporal).float(),
            'last_temporal_output': torch.from_numpy(last_temporal_output).float(),
            'last_temporal_normalized': torch.from_numpy(last_temporal_normalized).float(),
            'temporal_mean': torch.from_numpy(temporal_mean).float(),
            'temporal_cdu_indices': self.temporal_cdu_indices,
            'x_algebraic': torch.from_numpy(qflow_future).float(),
            'x_algebraic_current': torch.from_numpy(qflow_current).float(),
            'y_algebraic': torch.from_numpy(y_algebraic).float(),
            'algebraic_cdu_indices': self.algebraic_cdu_indices,
            'last_output_full': torch.from_numpy(self.output_data[input_end - 1]).float(),
            'future_output_full': torch.from_numpy(
                self.output_data[output_start:output_end]).float(),
        }


class DomainDataset(Dataset):
    """
    Dataset for Phase 4: Domain-specific DeepONet models.
    
    Handles delta prediction targets with normalization.
    Includes output history concatenated with input for the branch network.
    """
    
    def __init__(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        config: DomainDeepONetConfig,
        input_normalizer: ZScoreNormalizer,
        output_normalizer: ZScoreNormalizer,
        domain_normalizer: DomainNormalizer,
        input_cols: List[str],
        output_cols: List[str],
        column_info: Dict[str, Any],
        is_train: bool = True,
    ):
        self.config = config
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.column_info = column_info
        self.is_train = is_train
        
        # Index mappings
        self.primary_indices = [
            i for i, c in enumerate(output_cols) 
            if c in column_info.get('primary_cols', [])
        ]
        self.secondary_indices = [
            i for i, c in enumerate(output_cols) 
            if c in column_info.get('secondary_cols', [])
        ]
        
        # Subsample data
        self.input_data = input_data[::config.subsample_factor].astype(np.float32)
        self.output_data = output_data[::config.subsample_factor].astype(np.float32)
        
        self.input_normalizer = input_normalizer
        self.output_normalizer = output_normalizer
        self.domain_normalizer = domain_normalizer
        
        # Normalize
        self.input_normalized = input_normalizer.transform(self.input_data, input_cols)
        self.output_normalized = output_normalizer.transform(self.output_data, output_cols)
        
        total_required = config.history_steps + config.prediction_steps
        self.n_samples = max(0, len(self.input_data) - total_required)
        
        if self.n_samples > 0:
            print(f"  {config.domain} dataset: {self.n_samples} samples "
                  f"({len(output_cols)} outputs, primary={len(self.primary_indices)}, "
                  f"secondary={len(self.secondary_indices)})")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H = self.config.history_steps
        K = self.config.prediction_steps
        input_end = idx + H
        output_start = input_end
        output_end = output_start + K
        
        # Input history (normalized)
        x_input = self.input_normalized[idx:input_end]
        
        # Include output history if configured
        if self.config.include_output_history:
            out_hist_start = max(0, input_end - self.config.output_history_steps)
            x_output_hist = self.output_normalized[out_hist_start:input_end]
            
            if x_output_hist.shape[0] < H:
                pad_len = H - x_output_hist.shape[0]
                padding = np.zeros((pad_len, x_output_hist.shape[1]), dtype=np.float32)
                x_output_hist = np.concatenate([padding, x_output_hist], axis=0)
            elif x_output_hist.shape[0] > H:
                x_output_hist = x_output_hist[-H:]
            
            x_combined = np.concatenate([x_input, x_output_hist], axis=1)
        else:
            x_combined = x_input
        
        # Last known values
        last_output = self.output_data[input_end - 1]
        
        # Future outputs (absolute)
        future_output = self.output_data[output_start:output_end]
        
        # Normalized delta targets
        future_3d = future_output[np.newaxis, :, :]
        last_2d = last_output[np.newaxis, :]
        y_normalized, _ = self.domain_normalizer.get_target(future_3d, last_2d, self.output_cols)
        y_normalized = y_normalized[0]
        
        return {
            'x': torch.from_numpy(x_combined).float(),
            'y': torch.from_numpy(y_normalized).float(),
            'last_output': torch.from_numpy(last_output).float(),
            'future_output': torch.from_numpy(future_output).float(),
        }


class FederatedDataset(Dataset):
    """
    Dataset for Phase 5/6: Federated DeepMMNet.
    
    Returns input history, output history (all dynamic), and per-group
    delta targets for each of the 6 decoder heads.
    
    Phase 6 addition: raw_inputs_last for physics constraints.
    """
    
    def __init__(
        self,
        input_data: np.ndarray,
        dynamic_output_data: np.ndarray,
        config: Union[FederatedConfig, PhysicsInformedConfig],
        normalizer: FederatedNormalizer,
        column_info: Dict[str, Any],
        include_raw_inputs: bool = False,
    ):
        self.config = config
        self.column_info = column_info
        self.normalizer = normalizer
        self.include_raw_inputs = include_raw_inputs
        
        # Subsample
        self.input_data = input_data[::config.subsample_factor].astype(np.float32)
        self.dynamic_data = dynamic_output_data[::config.subsample_factor].astype(np.float32)
        
        # Normalize inputs and dynamic outputs
        self.input_normalized = normalizer.input_normalizer.transform(
            self.input_data, column_info['input_cols']
        )
        self.dynamic_normalized = normalizer.output_normalizer.transform(
            self.dynamic_data, column_info['dynamic_cols']
        )
        
        # Index slices for each group
        self.temp_idx = column_info['temp_indices']
        self.flow_idx = column_info['flow_indices']
        self.pressure_idx = column_info['pressure_indices']
        self.flow_sec_idx = column_info['flow_sec_indices']
        self.pressure_sec_idx = column_info['pressure_sec_indices']
        self.power_idx = column_info['power_indices']
        
        total_required = config.history_steps + config.prediction_steps
        self.n_samples = max(0, len(self.input_data) - total_required)
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        H = self.config.history_steps
        K = self.config.prediction_steps
        input_end = idx + H
        output_start = input_end
        output_end = output_start + K
        
        # Encoder inputs
        u_hist = self.input_normalized[idx:input_end]
        y_hist = self.dynamic_normalized[idx:input_end]
        
        # Delta targets
        last_dynamic = self.dynamic_data[input_end - 1]
        future_dynamic = self.dynamic_data[output_start:output_end]
        
        # Compute normalized CONSECUTIVE deltas for each group:
        #   delta_k = (future_k - future_{k-1}) / scale, future_{-1} = last
        # so last + cumsum(delta * scale) reconstructs the absolute trajectory
        # (consistent with inverse_delta and the Phase 6 physics converter; equals
        # the old relative-to-last form only at K=1).
        dynamic_cols = self.column_info['dynamic_cols']
        prev_dynamic = np.concatenate(
            [last_dynamic[None, :], future_dynamic[:-1]], axis=0
        )
        consec = future_dynamic - prev_dynamic
        y_delta = np.zeros_like(future_dynamic)
        for i, col in enumerate(dynamic_cols):
            scale = self.normalizer.delta_normalizer.get_scale(col)
            y_delta[:, i] = consec[:, i] / scale
        
        # Split targets by group (6 heads)
        y_delta_T = y_delta[:, self.temp_idx]
        y_delta_V = y_delta[:, self.flow_idx]
        y_delta_p = y_delta[:, self.pressure_idx]
        y_delta_Vs = y_delta[:, self.flow_sec_idx]
        y_delta_ps = y_delta[:, self.pressure_sec_idx]
        y_delta_W = y_delta[:, self.power_idx]
        
        result = {
            'u_hist': torch.from_numpy(u_hist).float(),
            'y_hist': torch.from_numpy(y_hist).float(),
            'y_delta': torch.from_numpy(y_delta).float(),
            'y_delta_T': torch.from_numpy(y_delta_T).float(),
            'y_delta_V': torch.from_numpy(y_delta_V).float(),
            'y_delta_p': torch.from_numpy(y_delta_p).float(),
            'y_delta_Vs': torch.from_numpy(y_delta_Vs).float(),
            'y_delta_ps': torch.from_numpy(y_delta_ps).float(),
            'y_delta_W': torch.from_numpy(y_delta_W).float(),
            'last_dynamic': torch.from_numpy(last_dynamic).float(),
            'future_dynamic': torch.from_numpy(future_dynamic).float(),
        }
        
        # Phase 6: include raw inputs for physics constraints
        if self.include_raw_inputs:
            raw_inputs_last = self.input_data[input_end - 1]
            result['raw_inputs_last'] = torch.from_numpy(raw_inputs_last).float()
        
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Split strategy: chunk-shuffled vs contiguous
# ─────────────────────────────────────────────────────────────────────────────
#
# The contiguous 70/15/15 temporal split places train and val in different
# operational regimes (load, ambient T_ext, workload), which caps val loss
# regardless of architecture. The chunk-shuffled split divides the timeline into
# contiguous chunks, shuffles whole chunks across train/val/test, and confines
# every sliding window to a single chunk so there is no leakage across the
# train/val boundary. Chunk size is not hard-coded: it is solved from explicit
# constraints and a measured operational cycle (see ``suggest_chunk_size``).


def _autocorr(x: np.ndarray) -> Optional[np.ndarray]:
    """Normalized autocorrelation of a 1-D signal via FFT. None if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 4:
        return None
    x = x - x.mean()
    if not np.any(np.abs(x) > 1e-12):
        return None
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    if acf[0] <= 0:
        return None
    return acf / acf[0]


def _dominant_period(x: np.ndarray, min_lag: int, max_lag: int,
                     peak_threshold: float = 0.1) -> Optional[int]:
    """Lag (raw steps) of the first prominent autocorrelation peak, else None."""
    acf = _autocorr(x)
    if acf is None:
        return None
    hi = min(max_lag, acf.shape[0] - 1)
    lo = max(min_lag, 1)
    if hi <= lo:
        return None
    seg = acf[lo:hi]
    # locate local maxima within the search band
    is_peak = (seg[1:-1] > seg[:-2]) & (seg[1:-1] > seg[2:])
    peak_lags = np.nonzero(is_peak)[0] + 1  # offset into seg
    if peak_lags.size == 0:
        return None
    # first peak that clears the prominence threshold; fall back to tallest
    cleared = peak_lags[seg[peak_lags] >= peak_threshold]
    chosen = cleared[0] if cleared.size else peak_lags[int(np.argmax(seg[peak_lags]))]
    return int(lo + chosen)


def measure_dominant_cycle(
    data: np.ndarray,
    col_names: List[str],
    prefer_substrings: Tuple[str, ...] = ('t_ext', 'q_flow', 'qflow'),
    min_lag: int = 2,
    max_lag: Optional[int] = None,
) -> Optional[int]:
    """
    Estimate the slowest dominant operational cycle (in raw steps).

    Prefers columns matching ``prefer_substrings`` (T_ext / Q_flow drive the
    diurnal/load cycle); falls back to all columns if none match. Returns the
    longest dominant period across the selected signals, or None if no cyclic
    structure is detectable.
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[0] < 8:
        return None
    n = data.shape[0]
    max_lag = max_lag if max_lag is not None else n // 2

    lname = [c.lower() for c in col_names]
    sel = [i for i, c in enumerate(lname)
           if any(s in c for s in prefer_substrings)]
    if not sel:
        sel = list(range(data.shape[1]))

    periods = [p for p in (_dominant_period(data[:, i], min_lag, max_lag) for i in sel)
               if p is not None]
    return max(periods) if periods else None


def suggest_chunk_size(
    n_total: int,
    history_steps: int,
    prediction_steps: int,
    subsample_factor: int,
    train_ratio: float,
    val_ratio: float,
    min_windows_per_chunk: int = 15,
    min_chunks_per_split: int = 3,
    coverage_factor: float = 1.0,
    cycle_period: Optional[int] = None,
    explicit_chunk_size: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Solve for a chunk size from explicit constraints (no hard-coded sizes).

    Constraint A (statistical floor): a chunk must yield at least
    ``min_windows_per_chunk`` windows after subsampling, which also guarantees
    the chunk is wider than one window span (no boundary crossing)::

        LOWER = subsample_factor * (history + prediction + min_windows_per_chunk)

    Constraint B (regime-diversity floor): val and test must each receive at
    least ``min_chunks_per_split`` chunks, which derives the required chunk
    count from the actual split ratios::

        required_chunks = ceil(min_chunks_per_split / min(val_ratio, test_ratio))
        UPPER = n_total // required_chunks

    Selector: within ``[LOWER, UPPER]`` aim for ``coverage_factor`` times the
    measured operational cycle; if no cycle is measured, prefer maximal coverage
    (UPPER). An ``explicit_chunk_size`` bypasses the selector but is still
    validated against both floors.

    Raises ``ValueError`` when the dataset is too small to satisfy both floors.
    """
    span = (history_steps + prediction_steps) * subsample_factor
    lower = subsample_factor * (history_steps + prediction_steps + min_windows_per_chunk)
    test_ratio = 1.0 - train_ratio - val_ratio
    required_chunks = int(np.ceil(min_chunks_per_split / min(val_ratio, test_ratio)))
    upper = n_total // required_chunks

    report: Dict[str, Any] = {
        'window_span_raw': span,
        'lower_bound': lower,
        'upper_bound': upper,
        'required_chunks': required_chunks,
        'cycle_period': cycle_period,
        'coverage_factor': coverage_factor,
        'warnings': [],
    }

    if upper < lower:
        raise ValueError(
            f"Dataset too small for chunk-shuffled splits: n_total={n_total:,} "
            f"supports chunks <= {upper:,} (to give each of val/test "
            f">= {min_chunks_per_split} chunks), but the statistical floor needs "
            f">= {lower:,} (>= {min_windows_per_chunk} windows/chunk). "
            f"Reduce subsample_factor, lower min_windows_per_chunk/min_chunks_per_split, "
            f"or use k-fold cross-validation instead."
        )

    if explicit_chunk_size is not None:
        chunk_size = int(explicit_chunk_size)
        if chunk_size < lower:
            raise ValueError(
                f"explicit chunk_size={chunk_size:,} is below the statistical floor "
                f"{lower:,} ({min_windows_per_chunk} windows/chunk). Increase it."
            )
        if chunk_size > upper:
            report['warnings'].append(
                f"explicit chunk_size={chunk_size:,} exceeds {upper:,}; val/test may "
                f"receive fewer than {min_chunks_per_split} chunks."
            )
        report['chunk_size'] = chunk_size
        report['selector'] = 'explicit'
        return chunk_size, report

    if cycle_period is not None:
        target = int(np.ceil(coverage_factor * cycle_period))
        report['selector'] = 'cycle'
        if target > upper:
            report['warnings'].append(
                f"one operational cycle (~{cycle_period:,} steps x {coverage_factor}) "
                f"exceeds the largest feasible chunk {upper:,}; coverage is capped. "
                f"Consider k-fold for this dataset size."
            )
    else:
        target = upper  # no measurable cycle -> maximize coverage within feasibility
        report['selector'] = 'max_coverage'

    chunk_size = int(np.clip(target, lower, upper))
    report['chunk_size'] = chunk_size
    return chunk_size, report


def chunk_shuffled_ranges(
    n_total: int,
    chunk_size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_chunk_len: int,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Partition ``[0, n_total)`` into contiguous chunks, shuffle whole chunks, and
    assign ``train_ratio``/``val_ratio``/rest of the *chunks* to train/val/test.

    The trailing remainder is absorbed into the last chunk. Chunks shorter than
    ``min_chunk_len`` (cannot form a window) are dropped. Returns sorted index
    ranges per split so each chunk stays contiguous in time.
    """
    n_chunks = max(1, n_total // chunk_size)
    bounds = [(i * chunk_size, (i + 1) * chunk_size) for i in range(n_chunks)]
    bounds[-1] = (bounds[-1][0], n_total)  # absorb remainder into last chunk
    bounds = [(s, e) for (s, e) in bounds if (e - s) >= min_chunk_len]
    if not bounds:
        raise ValueError(
            f"No chunk of length >= {min_chunk_len:,} could be formed from "
            f"n_total={n_total:,} at chunk_size={chunk_size:,}."
        )

    return assign_chunk_ranges(bounds, train_ratio, val_ratio, seed)


def assign_chunk_ranges(
    bounds: List[Tuple[int, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Shuffle whole chunks (given as index ranges) and assign them to
    train/val/test by ``train_ratio``/``val_ratio``/rest.

    Shared by the ``chunk_shuffled`` strategy (chunks cut from one array) and the
    ``per_file`` strategy (chunks = separate files). Ranges are returned sorted
    so each chunk stays contiguous in time. Non-empty val/test are guaranteed
    whenever there are >= 3 chunks.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(bounds))
    n_train = int(round(len(bounds) * train_ratio))
    n_val = int(round(len(bounds) * val_ratio))
    # guarantee non-empty val/test when chunks allow it
    n_train = min(n_train, len(bounds) - 2) if len(bounds) >= 3 else n_train
    n_val = max(n_val, 1) if len(bounds) - n_train >= 2 else n_val

    idx = {
        'train': order[:n_train],
        'val': order[n_train:n_train + n_val],
        'test': order[n_train + n_val:],
    }
    return {split: [bounds[i] for i in sorted(ids)] for split, ids in idx.items()}


def build_split_arrays(
    input_data: np.ndarray,
    output_data: np.ndarray,
    config: Any,
    column_info: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
           Dict[str, List[Tuple[int, int]]],
           Dict[str, Any]]:
    """
    Build per-split lists of ``(input_chunk, output_chunk)`` arrays.

    Returns ``(split_arrays, ranges, report)``. For ``contiguous`` each split is
    a single chunk (legacy behavior, byte-for-byte equivalent). For
    ``chunk_shuffled`` each split is a list of shuffled chunks, with chunk size
    solved by :func:`suggest_chunk_size` and an operational cycle measured from
    the input columns.
    """
    n_total = len(input_data)
    strategy = getattr(config, 'split_strategy', 'contiguous')

    if strategy == 'contiguous':
        train_end = int(n_total * config.train_ratio)
        val_end = int(n_total * (config.train_ratio + config.val_ratio))
        ranges = {
            'train': [(0, train_end)],
            'val': [(train_end, val_end)],
            'test': [(val_end, n_total)],
        }
        report = {'strategy': 'contiguous'}
    elif strategy == 'chunk_shuffled':
        cycle = None
        if column_info is not None and 'input_cols' in column_info:
            cycle = measure_dominant_cycle(input_data, column_info['input_cols'])
        explicit = None if config.chunk_size == 'auto' else int(config.chunk_size)
        chunk_size, report = suggest_chunk_size(
            n_total,
            config.history_steps, config.prediction_steps, config.subsample_factor,
            config.train_ratio, config.val_ratio,
            min_windows_per_chunk=config.min_windows_per_chunk,
            min_chunks_per_split=config.min_chunks_per_split,
            coverage_factor=config.coverage_factor,
            cycle_period=cycle,
            explicit_chunk_size=explicit,
        )
        report['strategy'] = 'chunk_shuffled'
        span = (config.history_steps + config.prediction_steps) * config.subsample_factor
        ranges = chunk_shuffled_ranges(
            n_total, chunk_size, config.train_ratio, config.val_ratio,
            seed=config.split_seed,
            min_chunk_len=span + config.subsample_factor,  # >= 1 window
        )
    elif strategy == 'per_file':
        # Each generated chunk file is one independent observation of the system
        # (separately re-stabilized through the FMU), so its boundaries are real
        # discontinuities: windows must never cross them. Whole files are assigned
        # to train/val/test. Boundaries are the file extents in the concatenated
        # array, supplied on the config by the dataloader factory.
        boundaries = getattr(config, 'chunk_boundaries', None)
        if not boundaries:
            raise ValueError(
                "split_strategy='per_file' requires per-file boundaries. Pass "
                "chunk_dfs= or data_dir= to create_surrogate_dataloaders (it sets "
                "config.chunk_boundaries for you)."
            )
        span = (config.history_steps + config.prediction_steps) * config.subsample_factor
        min_chunk_len = span + config.subsample_factor  # >= 1 window after subsampling
        usable = [(s, e) for (s, e) in boundaries if (e - s) >= min_chunk_len]
        dropped = len(boundaries) - len(usable)
        if not usable:
            raise ValueError(
                f"No chunk file is long enough to form a window: need "
                f">= {min_chunk_len:,} rows (history+prediction span x subsample), "
                f"but all {len(boundaries)} files are shorter. Reduce "
                f"subsample_factor/history_steps/prediction_steps."
            )
        ranges = assign_chunk_ranges(
            usable, config.train_ratio, config.val_ratio, config.split_seed
        )
        report = {
            'strategy': 'per_file',
            'n_files': len(boundaries),
            'n_usable': len(usable),
            'dropped_short': dropped,
            'warnings': (
                [f"{dropped} file(s) dropped (too short to form a window)"]
                if dropped else []
            ),
        }
        if len(usable) < 3:
            report['warnings'].append(
                f"only {len(usable)} usable file(s); val/test may be empty. "
                f"Generate more chunks for a clean per-file split."
            )
    else:
        raise ValueError(f"Unknown split_strategy: {strategy!r}")

    split_arrays = {
        split: [(input_data[s:e], output_data[s:e]) for (s, e) in rgs]
        for split, rgs in ranges.items()
    }
    return split_arrays, ranges, report


def concat_train_arrays(
    split_arrays: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate train chunks into single arrays for fitting normalizers."""
    chunks = split_arrays['train']
    train_input = np.concatenate([c[0] for c in chunks], axis=0)
    train_output = np.concatenate([c[1] for c in chunks], axis=0)
    return train_input, train_output


def _predicted_windows(start: int, end: int, config: Any) -> int:
    """Windows a contiguous range yields after subsampling (matches Dataset)."""
    n = (end - start) // config.subsample_factor
    return max(0, n - (config.history_steps + config.prediction_steps))


def print_split_report(
    ranges: Dict[str, List[Tuple[int, int]]],
    config: Any,
    report: Dict[str, Any],
) -> None:
    """Print chunk/window counts per split and enforce the per-chunk floor."""
    strategy = report.get('strategy', 'contiguous')
    print(f"\nData split ({strategy}):")
    if strategy == 'chunk_shuffled':
        print(f"  chunk_size={report['chunk_size']:,} raw steps "
              f"(feasible [{report['lower_bound']:,}, {report['upper_bound']:,}], "
              f"selector={report['selector']}, cycle={report['cycle_period']})")
    elif strategy == 'per_file':
        print(f"  {report['n_usable']}/{report['n_files']} chunk files used as "
              f"independent observations (whole files assigned to splits)")
    for split in ('train', 'val', 'test'):
        rgs = ranges[split]
        wins = [_predicted_windows(s, e, config) for (s, e) in rgs]
        total = sum(wins)
        if rgs:
            print(f"  {split:5s}: {len(rgs):3d} chunk(s), {total:,} windows "
                  f"(min/median per chunk: {min(wins)}/{int(np.median(wins))})")
        else:
            print(f"  {split:5s}:   0 chunk(s)")
    for w in report.get('warnings', []):
        print(f"  [warn] {w}")

    if strategy == 'chunk_shuffled':
        all_wins = [_predicted_windows(s, e, config)
                    for split in ('train', 'val', 'test') for (s, e) in ranges[split]]
        worst = min(all_wins) if all_wins else 0
        assert worst >= config.min_windows_per_chunk, (
            f"A chunk yields only {worst} windows (< min_windows_per_chunk="
            f"{config.min_windows_per_chunk}); increase chunk_size or lower "
            f"subsample_factor."
        )


def _concat_datasets(datasets: List[Dataset]) -> Dataset:
    """Wrap per-chunk datasets, dropping empties. Single dataset passes through."""
    non_empty = [d for d in datasets if len(d) > 0]
    if not non_empty:
        raise ValueError("Split produced no usable windows.")
    return non_empty[0] if len(non_empty) == 1 else ConcatDataset(non_empty)


# ─────────────────────────────────────────────────────────────────────────────
# Loading per-chunk files as separate observations
# ─────────────────────────────────────────────────────────────────────────────

def load_chunk_dataframes(
    data_dir: str,
    pattern: str = "fmu_output_*_operational.parquet",
    time_col: str = "time",
    sort_within_chunk: bool = True,
) -> List[pd.DataFrame]:
    """
    Load one self-contained DataFrame per ``chunk_<id>/`` directory.

    The parallel FMU generator writes each chunk's record as
    ``{**cooling_inputs, **cooling_outputs}``, so every per-chunk output file
    already carries both input and output columns — no input/output merge is
    needed. Directories are returned in natural chunk-id order (chunk_2 before
    chunk_10), and rows within each chunk are sorted by ``time_col`` when present.

    Parameters
    ----------
    data_dir : str
        Directory containing ``chunk_<id>/`` subdirectories.
    pattern : str
        Glob for the output file inside each chunk dir. The lexicographically
        first match is used.
    time_col : str
        Column to sort each chunk by (ignored if absent).
    sort_within_chunk : bool
        Sort rows within each chunk by ``time_col`` before returning.

    Returns
    -------
    List[pd.DataFrame]
        One DataFrame per chunk, in chunk-id order. Empty list if none found.
    """
    chunk_dirs = glob.glob(os.path.join(data_dir, "chunk_*"))

    def _chunk_id(path: str) -> int:
        m = re.search(r"chunk_(\d+)", os.path.basename(path))
        return int(m.group(1)) if m else -1

    chunk_dirs = sorted(chunk_dirs, key=_chunk_id)

    dfs: List[pd.DataFrame] = []
    for d in chunk_dirs:
        matches = sorted(glob.glob(os.path.join(d, pattern)))
        if not matches:
            continue
        cdf = pd.read_parquet(matches[0])
        if sort_within_chunk and time_col in cdf.columns:
            cdf = cdf.sort_values(time_col).reset_index(drop=True)
        dfs.append(cdf)
    return dfs


def concat_chunk_dataframes(
    chunk_dfs: List[pd.DataFrame],
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    Concatenate per-chunk DataFrames into one frame and record file boundaries.

    Returns ``(df, boundaries)`` where ``boundaries[i] = (start, end)`` is the
    half-open row range of chunk ``i`` in the concatenated frame. The boundaries
    are what the ``per_file`` split strategy uses to keep every sliding window
    inside a single chunk. Column order is taken from the first chunk.
    """
    if not chunk_dfs:
        raise ValueError("concat_chunk_dataframes received an empty chunk list.")
    boundaries: List[Tuple[int, int]] = []
    start = 0
    for cdf in chunk_dfs:
        n = len(cdf)
        boundaries.append((start, start + n))
        start += n
    df = pd.concat(chunk_dfs, axis=0, ignore_index=True)
    return df, boundaries


# ─────────────────────────────────────────────────────────────────────────────
# Dataloader factory functions
# ─────────────────────────────────────────────────────────────────────────────

def create_dataloaders_lstm(
    df: pd.DataFrame,
    column_info: Dict[str, Any],
    config: LSTMConfig,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, SurrogateNormalizer]:
    """Create train/val/test dataloaders for Phase 1 LSTM."""
    input_cols = column_info['input_cols']
    output_cols = column_info['output_cols']
    
    input_data = df[input_cols].values.astype(np.float32)
    output_data = df[output_cols].values.astype(np.float32)

    split_arrays, ranges, report = build_split_arrays(
        input_data, output_data, config, column_info
    )
    print_split_report(ranges, config, report)

    # Fit normalizer on concatenated train chunks only
    train_input, train_output = concat_train_arrays(split_arrays)
    normalizer = SurrogateNormalizer(use_whitening=False, delta_scale_factor=10.0)
    normalizer.fit(
        train_input, train_output,
        input_cols, output_cols,
        temporal_cols=output_cols,
        subsample_factor=config.subsample_factor,
    )

    # One dataset per chunk; windows stay within a chunk via ConcatDataset
    loaders = {}
    for split in ('train', 'val', 'test'):
        per_chunk = [
            LSTMDataset(inp, out, config, normalizer, column_info)
            for (inp, out) in split_arrays[split]
        ]
        loaders[split] = _concat_datasets(per_chunk)

    train_loader = DataLoader(
        loaders['train'], batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        loaders['val'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        loaders['test'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    print(f"\nDataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Test batches:  {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, normalizer

def create_dataloaders_hybrid(
    df: pd.DataFrame,
    column_info: Dict[str, Any],
    qflow_indices: List[int],
    config: Union[DeepONetConfig, HybridDeepONetConfig],
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create train/val/test dataloaders for Phase 2/3 Hybrid DeepONet."""
    input_cols = column_info['input_cols']
    output_cols = column_info['output_cols']
    temporal_cols = column_info['temporal_cols']
    
    input_data = df[input_cols].values.astype(np.float32)
    output_data = df[output_cols].values.astype(np.float32)

    split_arrays, ranges, report = build_split_arrays(
        input_data, output_data, config, column_info
    )
    print_split_report(ranges, config, report)

    # Fit normalizers on concatenated train chunks only
    print("\nFitting normalizers...")
    train_input, train_output = concat_train_arrays(split_arrays)
    input_normalizer = ZScoreNormalizer().fit(train_input, input_cols)
    output_normalizer = ZScoreNormalizer().fit(train_output, output_cols)

    temporal_indices = [output_cols.index(c) for c in temporal_cols]
    temporal_output_data = train_output[:, temporal_indices]
    delta_normalizer = DeltaNormalizer().fit(
        temporal_output_data, temporal_cols, config.subsample_factor
    )

    # Input whitening (if enabled)
    input_whitener = None
    use_whitening = getattr(config, 'use_input_whitening', False)
    if use_whitening:
        print("\nFitting input whitener...")
        train_input_normalized = input_normalizer.transform(train_input, input_cols)
        train_input_subsampled = train_input_normalized[::config.subsample_factor]
        whitening_components = getattr(config, 'whitening_components', 0.99)
        input_whitener = InputWhitener(whitening_components).fit(train_input_subsampled)
        column_info['whitened_n_inputs'] = input_whitener._n_components_out

    # One dataset per chunk; windows stay within a chunk via ConcatDataset
    print("\nCreating datasets...")
    loaders = {}
    for split in ('train', 'val', 'test'):
        is_train = split == 'train'
        per_chunk = [
            HybridDataset(
                inp, out, config,
                input_normalizer, output_normalizer, delta_normalizer, input_whitener,
                column_info, qflow_indices,
                cdu_ids=config.cdu_ids,
                is_train=is_train,
            )
            for (inp, out) in split_arrays[split]
        ]
        loaders[split] = _concat_datasets(per_chunk)

    train_loader = DataLoader(
        loaders['train'], batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        loaders['val'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        loaders['test'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    normalizers = {
        'input': input_normalizer,
        'output': output_normalizer,
        'delta': delta_normalizer,
        'whitener': input_whitener,
    }
    
    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, normalizers


def create_dataloaders_domain(
    df: pd.DataFrame,
    input_cols: List[str],
    output_cols: List[str],
    column_info: Dict[str, Any],
    config: DomainDeepONetConfig,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Create train/val/test dataloaders for Phase 4 Domain-specific DeepONet."""
    input_data = df[input_cols].values.astype(np.float32)
    output_data = df[output_cols].values.astype(np.float32)

    split_arrays, ranges, report = build_split_arrays(
        input_data, output_data, config, column_info
    )
    print_split_report(ranges, config, report)

    # Fit normalizers on concatenated train chunks only
    train_input, train_output = concat_train_arrays(split_arrays)
    input_normalizer = ZScoreNormalizer().fit(train_input, input_cols)
    output_normalizer = ZScoreNormalizer().fit(train_output, output_cols)
    domain_normalizer = DomainNormalizer(
        subsample_factor=config.subsample_factor
    ).fit(train_output, output_cols)
    domain_normalizer.delta.print_stats(output_cols)

    # One dataset per chunk; windows stay within a chunk via ConcatDataset
    loaders = {}
    for split in ('train', 'val', 'test'):
        is_train = split == 'train'
        per_chunk = [
            DomainDataset(
                inp, out, config,
                input_normalizer, output_normalizer, domain_normalizer,
                input_cols, output_cols, column_info, is_train=is_train,
            )
            for (inp, out) in split_arrays[split]
        ]
        loaders[split] = _concat_datasets(per_chunk)

    train_loader = DataLoader(
        loaders['train'], batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )
    val_loader = DataLoader(
        loaders['val'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        loaders['test'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    normalizers = {
        'input': input_normalizer,
        'output': output_normalizer,
        'domain': domain_normalizer,
    }
    
    print(f"  DataLoader sizes: train={len(train_loader)}, "
          f"val={len(val_loader)}, test={len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader, normalizers


def create_dataloaders_federated(
    df: pd.DataFrame,
    column_info: Dict[str, Any],
    config: Union[FederatedConfig, PhysicsInformedConfig],
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, FederatedNormalizer]:
    """Create train/val/test dataloaders for Phase 5/6 Federated DeepMMNet."""
    input_cols = column_info['input_cols']
    dynamic_cols = column_info['dynamic_cols']
    
    input_data = df[input_cols].values.astype(np.float32)
    dynamic_data = df[dynamic_cols].values.astype(np.float32)

    split_arrays, ranges, report = build_split_arrays(
        input_data, dynamic_data, config, column_info
    )
    print_split_report(ranges, config, report)

    # Fit normalizers on concatenated train chunks only
    train_input, train_dynamic = concat_train_arrays(split_arrays)
    normalizer = FederatedNormalizer(subsample_factor=config.subsample_factor)
    normalizer.fit(train_input, train_dynamic, input_cols, dynamic_cols)

    # Determine if we need raw inputs (Phase 6)
    include_raw_inputs = isinstance(config, PhysicsInformedConfig)

    # One dataset per chunk; windows stay within a chunk via ConcatDataset
    loaders = {}
    for split in ('train', 'val', 'test'):
        per_chunk = [
            FederatedDataset(
                inp, dyn, config, normalizer, column_info,
                include_raw_inputs=include_raw_inputs,
            )
            for (inp, dyn) in split_arrays[split]
        ]
        loaders[split] = _concat_datasets(per_chunk)

    train_loader = DataLoader(
        loaders['train'], batch_size=config.batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    )
    val_loader = DataLoader(
        loaders['val'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        loaders['test'], batch_size=config.batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    print(f"\nDataLoader sizes:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")
    print(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader, normalizer


def create_surrogate_dataloaders(
    df: Optional[pd.DataFrame] = None,
    config: SurrogateConfig = None,
    column_info: Optional[Dict[str, Any]] = None,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    chunk_dfs: Optional[List[pd.DataFrame]] = None,
    data_dir: Optional[str] = None,
    store_dir: Optional[str] = None,
    chunk_ids: Optional[List[int]] = None,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, Any]:
    """
    Unified factory function to create dataloaders for any phase.

    Automatically detects the phase from config type and creates
    the appropriate dataset and dataloaders.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Single dataframe with all columns. Mutually exclusive with
        ``chunk_dfs``/``data_dir``. Split internally per ``config.split_strategy``
        (``contiguous`` or ``chunk_shuffled``).
    config : SurrogateConfig
        Configuration (LSTMConfig, DeepONetConfig, HybridDeepONetConfig,
        DomainDeepONetConfig, FederatedConfig, or PhysicsInformedConfig)
    column_info : Dict[str, Any], optional
        Column information (built from df if not provided)
    num_workers : int
        Number of dataloader workers
    pin_memory : bool
        Whether to pin memory for GPU transfer
    chunk_dfs : List[pd.DataFrame], optional
        One self-contained dataframe per chunk, each treated as an independent
        observation of the same system. Triggers the ``per_file`` split strategy:
        whole chunks are assigned to train/val/test and no sliding window ever
        crosses a chunk boundary. Sets ``config.split_strategy='per_file'`` and
        ``config.chunk_boundaries`` as a side effect.
    data_dir : str, optional
        Directory of ``chunk_<id>/`` folders. Loaded via
        :func:`load_chunk_dataframes` into ``chunk_dfs`` when the latter is not
        given.
    chunk_ids : List[int], optional
        Process only these chunk ids (filters the discovered chunks under
        ``data_dir``). ``None`` (default) processes all discovered chunks.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, Any]
        Train, validation, test dataloaders and normalizer(s)
    """
    # Recommended Phase 5/6 path on full data: out-of-core memmap store with
    # anti-aliased (block-mean) decimation. Single-rate by default — uses
    # config.subsample_factor; set per-group head_subsample for multi-rate. Windows
    # are sliced from disk on demand, so RAM stays bounded (no loading all chunks).
    if (data_dir is not None and chunk_dfs is None
            and isinstance(config, (FederatedConfig, PhysicsInformedConfig))):
        from . import store as _store
        from .federated_store import (
            build_store_for_config, create_dataloaders_federated_store)
        if column_info is None:
            files = _store.discover_chunk_files(data_dir)
            if not files:
                raise ValueError(f"No chunk_*/ output files found under {data_dir!r}.")
            column_info = build_federated_column_info(
                pd.read_parquet(files[0][1]), config)
        sd = store_dir or os.path.join(data_dir, "store")
        build_store_for_config(data_dir, sd, config, column_info,
                               chunk_ids=chunk_ids)
        # Memmap store path: default to NOT pinning (page-locking long 1 Hz
        # multi-key batches is unreclaimable host RAM); honor an explicit choice.
        store_pin = False if pin_memory is None else pin_memory
        return create_dataloaders_federated_store(
            sd, config, column_info, num_workers=num_workers, pin_memory=store_pin,
            include_raw_inputs=isinstance(config, PhysicsInformedConfig),
            chunk_ids=chunk_ids, prefetch_factor=prefetch_factor,
        )

    # Eager in-RAM paths below pin by default (small enough to benefit).
    if pin_memory is None:
        pin_memory = True

    # Per-file mode: each chunk file is a separate observation of the system.
    if data_dir is not None and chunk_dfs is None:
        chunk_dfs = load_chunk_dataframes(data_dir)
        if not chunk_dfs:
            raise ValueError(f"No chunk_*/ output files found under {data_dir!r}.")
    if chunk_dfs is not None:
        df, boundaries = concat_chunk_dataframes(chunk_dfs)
        # Threaded to build_split_arrays via the config (keeps the 4 factory
        # functions and their signatures untouched).
        config.split_strategy = 'per_file'
        config.chunk_boundaries = boundaries
        print(f"Per-file mode: {len(chunk_dfs)} chunk files -> "
              f"{len(df):,} total rows as independent observations")
    elif df is None:
        raise ValueError("Provide one of: df, chunk_dfs, or data_dir.")

    # Phase 5/6: Federated
    if isinstance(config, (FederatedConfig, PhysicsInformedConfig)):
        if column_info is None:
            column_info = build_federated_column_info(df, config)
        return create_dataloaders_federated(
            df, column_info, config, num_workers, pin_memory
        )
    
    # Phase 4: Domain-specific
    if isinstance(config, DomainDeepONetConfig):
        if column_info is None:
            input_cols, output_cols, column_info = build_domain_column_info(df, config)
        else:
            input_cols = column_info.get('input_cols', [])
            output_cols = column_info.get('output_cols', [])
        return create_dataloaders_domain(
            df, input_cols, output_cols, column_info, config, num_workers, pin_memory
        )
    
    # Phase 2/3: Hybrid DeepONet
    if isinstance(config, (DeepONetConfig, HybridDeepONetConfig)):
        if column_info is None:
            column_info = build_column_info(df, config)
        qflow_indices = identify_qflow_columns(column_info['input_cols'], config)
        return create_dataloaders_hybrid(
            df, column_info, qflow_indices, config, num_workers, pin_memory
        )
    
    # Phase 1: LSTM (or base config)
    if column_info is None:
        column_info = build_column_info(df, config)
    
    if isinstance(config, LSTMConfig):
        return create_dataloaders_lstm(
            df, column_info, config, num_workers, pin_memory
        )
    
    # Default: treat base SurrogateConfig as LSTM-style
    lstm_config = LSTMConfig(
        system_name=config.system_name,
        num_cdus=config.num_cdus,
        history_steps=config.history_steps,
        prediction_steps=config.prediction_steps,
        subsample_factor=config.subsample_factor,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        batch_size=config.batch_size,
    )
    return create_dataloaders_lstm(
        df, column_info, lstm_config, num_workers, pin_memory
    )


__all__ = [
    'LSTMDataset',
    'HybridDataset',
    'DomainDataset',
    'FederatedDataset',
    'create_dataloaders_lstm',
    'create_dataloaders_hybrid',
    'create_dataloaders_domain',
    'create_dataloaders_federated',
    'create_surrogate_dataloaders',
    'measure_dominant_cycle',
    'suggest_chunk_size',
    'chunk_shuffled_ranges',
    'assign_chunk_ranges',
    'build_split_arrays',
    'concat_train_arrays',
    'print_split_report',
    'load_chunk_dataframes',
    'concat_chunk_dataframes',
]