"""
Normalizers for surrogate model data preprocessing.

Provides:
- ZScoreNormalizer: Standard z-score normalization
- DeltaNormalizer: Normalization for delta (change) predictions
- InputWhitener: PCA-based input whitening
- DomainNormalizer: Phase 4 domain-specific normalizer
- FederatedNormalizer: Phase 5/6 federated normalizer
- SurrogateNormalizer: Unified normalizer manager for all phases
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import json
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ZScoreNormalizer:
    """
    Z-score (standardization) normalizer.
    
    Transforms data to have zero mean and unit variance per column.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        Initialize ZScoreNormalizer.
        
        Parameters
        ----------
        epsilon : float
            Small constant added to std to avoid division by zero
        """
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
    
    def fit(self, data: np.ndarray, col_names: List[str]) -> 'ZScoreNormalizer':
        """
        Fit normalizer to data.
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_samples, n_features)
        col_names : List[str]
            Column names corresponding to features
            
        Returns
        -------
        ZScoreNormalizer
            Self for method chaining
        """
        assert data.shape[1] == len(col_names), \
            f"Data has {data.shape[1]} columns but {len(col_names)} column names provided"
        
        for i, col in enumerate(col_names):
            col_data = data[:, i]
            self.stats[col] = {
                'mean': float(np.nanmean(col_data)),
                'std': float(np.nanstd(col_data) + self.epsilon),
                'min': float(np.nanmin(col_data)),
                'max': float(np.nanmax(col_data)),
            }
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray, col_names: List[str]) -> np.ndarray:
        """
        Transform data using fitted statistics.
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_samples, n_features)
        col_names : List[str]
            Column names corresponding to features
            
        Returns
        -------
        np.ndarray
            Normalized data
        """
        if not self.is_fitted:
            raise RuntimeError("ZScoreNormalizer not fitted. Call fit() first.")
        
        normalized = np.zeros_like(data, dtype=np.float32)
        for i, col in enumerate(col_names):
            s = self.stats[col]
            normalized[:, i] = (data[:, i] - s['mean']) / s['std']
        return normalized
    
    def inverse_transform(self, data: np.ndarray, col_names: List[str]) -> np.ndarray:
        """
        Inverse transform normalized data back to original scale.
        
        Parameters
        ----------
        data : np.ndarray
            Normalized data array
        col_names : List[str]
            Column names corresponding to features
            
        Returns
        -------
        np.ndarray
            Denormalized data
        """
        if not self.is_fitted:
            raise RuntimeError("ZScoreNormalizer not fitted. Call fit() first.")
        
        denormalized = np.zeros_like(data, dtype=np.float32)
        for i, col in enumerate(col_names):
            s = self.stats[col]
            denormalized[:, i] = data[:, i] * s['std'] + s['mean']
        return denormalized
    
    def fit_transform(self, data: np.ndarray, col_names: List[str]) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data, col_names).transform(data, col_names)
    
    def get_mean(self, col: str) -> float:
        """Get mean for a column."""
        return self.stats[col]['mean']
    
    def get_std(self, col: str) -> float:
        """Get std for a column."""
        return self.stats[col]['std']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'ZScoreNormalizer',
            'epsilon': self.epsilon,
            'stats': self.stats,
            'is_fitted': self.is_fitted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ZScoreNormalizer':
        """Create from dictionary."""
        normalizer = cls(epsilon=d.get('epsilon', 1e-8))
        normalizer.stats = d['stats']
        normalizer.is_fitted = d.get('is_fitted', True)
        return normalizer


class DeltaNormalizer:
    """
    Normalizer for delta (change) predictions.
    
    Computes statistics on temporal differences for proper scaling
    of delta predictions in temporal outputs.
    """
    
    def __init__(self, scale_factor: float = 10.0, epsilon: float = 1e-10):
        """
        Initialize DeltaNormalizer.
        
        Parameters
        ----------
        scale_factor : float
            Multiplier for delta_std to get scale (default 10x)
        epsilon : float
            Small constant to avoid division by zero
        """
        self.scale_factor = scale_factor
        self.epsilon = epsilon
        self.stats: Dict[str, Dict[str, float]] = {}
        self.is_fitted = False
    
    def fit(
        self,
        data: np.ndarray,
        col_names: List[str],
        subsample_factor: int = 1,
    ) -> 'DeltaNormalizer':
        """
        Fit normalizer to data by computing delta statistics.
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_samples, n_features)
        col_names : List[str]
            Column names corresponding to features
        subsample_factor : int
            Factor to subsample before computing deltas
            
        Returns
        -------
        DeltaNormalizer
            Self for method chaining
        """
        for i, col in enumerate(col_names):
            col_data = data[::subsample_factor, i]
            deltas = np.diff(col_data)
            
            self.stats[col] = {
                'delta_mean': float(np.nanmean(deltas)),
                'delta_std': float(np.nanstd(deltas) + self.epsilon),
                'delta_abs_max': float(np.abs(deltas).max() + self.epsilon),
                'abs_mean': float(np.nanmean(col_data)),
                'abs_std': float(np.nanstd(col_data) + self.epsilon),
            }
        
        self.is_fitted = True
        return self
    
    def get_scale(self, col: str) -> float:
        """
        Get scale factor for normalizing deltas.
        
        Parameters
        ----------
        col : str
            Column name
            
        Returns
        -------
        float
            Scale factor (delta_std * scale_factor)
        """
        if not self.is_fitted:
            raise RuntimeError("DeltaNormalizer not fitted. Call fit() first.")
        return self.stats[col]['delta_std'] * self.scale_factor
    
    def normalize_delta(self, delta: np.ndarray, col: str) -> np.ndarray:
        """Normalize a delta value."""
        scale = self.get_scale(col)
        return delta / scale
    
    def denormalize_delta(self, normalized_delta: np.ndarray, col: str) -> np.ndarray:
        """Denormalize a delta value."""
        scale = self.get_scale(col)
        return normalized_delta * scale
    
    def print_stats(self, col_names: List[str], n_sample: int = 3) -> None:
        """Print delta statistics for sample columns."""
        print(f"  Delta normalizer fitted ({len(col_names)} cols):")
        sample = col_names[:n_sample]
        for col in sample:
            s = self.stats[col]
            print(f"    {col}: delta_std={s['delta_std']:.8f}, scale={self.get_scale(col):.8f}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'DeltaNormalizer',
            'scale_factor': self.scale_factor,
            'epsilon': self.epsilon,
            'stats': self.stats,
            'is_fitted': self.is_fitted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DeltaNormalizer':
        """Create from dictionary."""
        normalizer = cls(
            scale_factor=d.get('scale_factor', 10.0),
            epsilon=d.get('epsilon', 1e-10),
        )
        normalizer.stats = d['stats']
        normalizer.is_fitted = d.get('is_fitted', True)
        return normalizer


class InputWhitener:
    """
    PCA-based input whitening for decorrelation.
    
    Transforms inputs to have unit variance and zero correlation
    between features, which can improve training stability.
    """
    
    def __init__(self, n_components: Union[int, float] = 0.99):
        """
        Initialize InputWhitener.
        
        Parameters
        ----------
        n_components : int or float
            If int, number of components to keep.
            If float (0-1), fraction of variance to retain.
        """
        if not HAS_SKLEARN:
            raise ImportError("InputWhitener requires scikit-learn. Install with: pip install scikit-learn")
        
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.mean: Optional[np.ndarray] = None
        self.is_fitted = False
        self._n_components_out: int = 0
    
    def fit(self, data: np.ndarray) -> 'InputWhitener':
        """
        Fit whitener to data.
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_samples, n_features)
            
        Returns
        -------
        InputWhitener
            Self for method chaining
        """
        self.mean = np.mean(data, axis=0)
        centered = data - self.mean
        
        self.pca = PCA(n_components=self.n_components, whiten=True)
        self.pca.fit(centered)
        
        self._n_components_out = self.pca.n_components_
        self.is_fitted = True
        
        print(f"  InputWhitener: {data.shape[1]} → {self._n_components_out} components")
        print(f"  Explained variance: {self.pca.explained_variance_ratio_.sum():.2%}")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted whitening.
        
        Parameters
        ----------
        data : np.ndarray
            Data array of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Whitened data of shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise RuntimeError("InputWhitener not fitted. Call fit() first.")
        centered = data - self.mean
        return self.pca.transform(centered).astype(np.float32)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(data).transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform whitened data."""
        if not self.is_fitted:
            raise RuntimeError("InputWhitener not fitted. Call fit() first.")
        return self.pca.inverse_transform(data) + self.mean
    
    @property
    def output_dim(self) -> int:
        """Number of output dimensions after whitening."""
        return self._n_components_out if self.is_fitted else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = {
            'type': 'InputWhitener',
            'n_components': self.n_components,
            'is_fitted': self.is_fitted,
        }
        if self.is_fitted:
            d['mean'] = self.mean.tolist()
            d['components'] = self.pca.components_.tolist()
            d['explained_variance'] = self.pca.explained_variance_.tolist()
            d['n_components_out'] = self._n_components_out
        return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'InputWhitener':
        """Create from dictionary."""
        whitener = cls(n_components=d.get('n_components', 0.99))
        if d.get('is_fitted', False):
            whitener.mean = np.array(d['mean'])
            whitener.pca = PCA(n_components=d['n_components_out'], whiten=True)
            whitener.pca.components_ = np.array(d['components'])
            whitener.pca.explained_variance_ = np.array(d['explained_variance'])
            whitener.pca.n_components_ = d['n_components_out']
            whitener._n_components_out = d['n_components_out']
            whitener.is_fitted = True
        return whitener


class DomainNormalizer:
    """
    Domain-specific normalizer for Phase 4.
    
    Handles delta prediction targets with proper normalization
    for domain-specific DeepONet models.
    """
    
    def __init__(self, subsample_factor: int = 10, scale_factor: float = 10.0):
        """
        Initialize DomainNormalizer.
        
        Parameters
        ----------
        subsample_factor : int
            Factor to subsample before computing delta statistics
        scale_factor : float
            Multiplier for delta_std to get scale
        """
        self.subsample_factor = subsample_factor
        self.scale_factor = scale_factor
        self.delta = DeltaNormalizer(scale_factor=scale_factor)
        self.is_fitted = False
    
    def fit(self, output_data: np.ndarray, output_cols: List[str]) -> 'DomainNormalizer':
        """
        Fit normalizer to output data.
        
        Parameters
        ----------
        output_data : np.ndarray
            Output data array
        output_cols : List[str]
            Output column names
            
        Returns
        -------
        DomainNormalizer
            Self for method chaining
        """
        self.delta.fit(output_data, output_cols, self.subsample_factor)
        self.is_fitted = True
        return self
    
    def get_target(
        self,
        future_output: np.ndarray,
        last_output: np.ndarray,
        col_names: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute normalized delta targets.
        
        Parameters
        ----------
        future_output : np.ndarray
            Future outputs of shape (batch, pred_steps, n_outputs)
        last_output : np.ndarray
            Last known outputs of shape (batch, n_outputs)
        col_names : List[str]
            Output column names
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (normalized_deltas, raw_deltas)
        """
        # Compute raw deltas
        deltas = future_output - last_output[:, np.newaxis, :]
        
        # Normalize
        y_normalized = np.zeros_like(deltas)
        for i, col in enumerate(col_names):
            scale = self.delta.get_scale(col)
            y_normalized[:, :, i] = deltas[:, :, i] / scale
        
        return y_normalized, deltas
    
    def inverse_transform(
        self,
        predictions: np.ndarray,
        last_output: np.ndarray,
        col_names: List[str],
    ) -> np.ndarray:
        """
        Convert normalized delta predictions to absolute values.
        
        Parameters
        ----------
        predictions : np.ndarray
            Normalized delta predictions of shape (batch, pred_steps, n_outputs)
        last_output : np.ndarray
            Last known outputs of shape (batch, n_outputs)
        col_names : List[str]
            Output column names
            
        Returns
        -------
        np.ndarray
            Absolute predictions
        """
        absolute = np.zeros_like(predictions)
        for i, col in enumerate(col_names):
            scale = self.delta.get_scale(col)
            raw_deltas = predictions[:, :, i] * scale
            cumulative = np.cumsum(raw_deltas, axis=1)
            absolute[:, :, i] = last_output[:, i:i+1] + cumulative
        return absolute
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'DomainNormalizer',
            'subsample_factor': self.subsample_factor,
            'scale_factor': self.scale_factor,
            'delta': self.delta.to_dict(),
            'is_fitted': self.is_fitted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'DomainNormalizer':
        """Create from dictionary."""
        normalizer = cls(
            subsample_factor=d.get('subsample_factor', 10),
            scale_factor=d.get('scale_factor', 10.0),
        )
        normalizer.delta = DeltaNormalizer.from_dict(d['delta'])
        normalizer.is_fitted = d.get('is_fitted', True)
        return normalizer


class FederatedNormalizer:
    """
    Federated normalizer for Phase 5/6.
    
    Manages normalizers for all 6 output groups in the federated model.
    Provides both numpy and torch-based inverse transforms for
    physics-informed training.
    """
    
    def __init__(self, subsample_factor: int = 10, scale_factor: float = 10.0):
        """
        Initialize FederatedNormalizer.
        
        Parameters
        ----------
        subsample_factor : int
            Factor to subsample before computing delta statistics
        scale_factor : float
            Multiplier for delta_std to get scale
        """
        self.subsample_factor = subsample_factor
        self.scale_factor = scale_factor
        self.input_normalizer = ZScoreNormalizer()
        self.output_normalizer = ZScoreNormalizer()
        self.delta_normalizer = DeltaNormalizer(scale_factor=scale_factor)
        # Multi-rate: one DeltaNormalizer per sampling rate, each fit on the
        # already-decimated series at that rate (subsample_factor=1). Empty in the
        # single-rate path, where ``delta_normalizer`` alone is used.
        self.delta_by_rate: Dict[int, DeltaNormalizer] = {}
        self.input_cols: List[str] = []
        self.output_cols: List[str] = []
        self.is_fitted = False
    
    def fit(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_cols: List[str],
        output_cols: List[str],
    ) -> 'FederatedNormalizer':
        """
        Fit all normalizers to training data.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input data array
        output_data : np.ndarray
            Output data array (all dynamic outputs)
        input_cols : List[str]
            Input column names
        output_cols : List[str]
            Output column names (all dynamic)
            
        Returns
        -------
        FederatedNormalizer
            Self for method chaining
        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        
        self.input_normalizer.fit(input_data, input_cols)
        self.output_normalizer.fit(output_data, output_cols)
        self.delta_normalizer.fit(output_data, output_cols, self.subsample_factor)

        self.is_fitted = True
        return self

    def fit_multirate(
        self,
        input_base: np.ndarray,
        dynamic_by_rate: Dict[int, np.ndarray],
        input_cols: List[str],
        output_cols: List[str],
    ) -> 'FederatedNormalizer':
        """
        Fit normalizers from already-decimated memmap-store arrays.

        ZScore stats (rate-insensitive) are fit on the finest available rate;
        a separate :class:`DeltaNormalizer` is fit per rate on that rate's
        decimated dynamic data with ``subsample_factor=1`` — which equals
        ``[::rate]``-then-diff on the native series, so per-group delta scales are
        correct for each group's own grid. ``delta_normalizer`` is set to the
        finest rate so single-rate / rate-agnostic callers keep working.

        Parameters
        ----------
        input_base : np.ndarray
            Input data at the finest rate (concatenated over train chunks).
        dynamic_by_rate : Dict[int, np.ndarray]
            Dynamic (output) data per rate, concatenated over train chunks.
        input_cols, output_cols : List[str]
            Column names (output_cols = all dynamic, in store order).
        """
        if not dynamic_by_rate:
            raise ValueError("fit_multirate requires at least one rate.")
        self.input_cols = input_cols
        self.output_cols = output_cols
        base = min(dynamic_by_rate)

        self.input_normalizer.fit(input_base, input_cols)
        self.output_normalizer.fit(dynamic_by_rate[base], output_cols)

        self.delta_by_rate = {}
        for r, dyn in dynamic_by_rate.items():
            self.delta_by_rate[r] = DeltaNormalizer(scale_factor=self.scale_factor).fit(
                dyn, output_cols, subsample_factor=1
            )
        # Default for rate-agnostic callers (legacy single-rate eval/UQ).
        self.delta_normalizer = self.delta_by_rate[base]
        self.is_fitted = True
        return self

    def _delta_for(self, rate: Optional[int]) -> DeltaNormalizer:
        """Pick the DeltaNormalizer for ``rate`` (falls back to the default)."""
        if rate is not None and rate in self.delta_by_rate:
            return self.delta_by_rate[rate]
        return self.delta_normalizer

    def get_delta_target(
        self,
        future_data: np.ndarray,
        last_data: np.ndarray,
        col_names: List[str],
        rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute normalized **consecutive** delta targets.

        ``delta_k = (future_k - future_{k-1}) / scale`` with ``future_{-1} = last``,
        so ``last + cumsum(deltas * scale)`` (see :meth:`inverse_delta`) exactly
        reconstructs the absolute trajectory. This is the consistent convention
        across the codebase; it coincides with the old relative-to-last form only
        at K=1.

        Parameters
        ----------
        future_data : np.ndarray
            Future outputs of shape (batch, pred_steps, n_outputs)
        last_data : np.ndarray
            Last known outputs of shape (batch, n_outputs)
        col_names : List[str]
            Output column names
        rate : int, optional
            Sampling rate selecting the per-rate delta scale.

        Returns
        -------
        np.ndarray
            Normalized consecutive-delta targets
        """
        delta = self._delta_for(rate)
        # prev_k = future_{k-1}, with prev_0 = last
        prev = np.concatenate(
            [last_data[:, None, :], future_data[:, :-1, :]], axis=1
        )
        consec = (future_data - prev).astype(np.float32)
        targets = np.zeros_like(future_data, dtype=np.float32)
        for i, col in enumerate(col_names):
            targets[:, :, i] = consec[:, :, i] / delta.get_scale(col)
        return targets

    def inverse_delta(
        self,
        predictions: np.ndarray,
        last_data: np.ndarray,
        col_names: List[str],
        rate: Optional[int] = None,
    ) -> np.ndarray:
        """
        Convert normalized deltas back to absolute values.
        
        Parameters
        ----------
        predictions : np.ndarray
            Normalized delta predictions of shape (batch, pred_steps, n_outputs)
        last_data : np.ndarray
            Last known outputs of shape (batch, n_outputs)
        col_names : List[str]
            Output column names
            
        Returns
        -------
        np.ndarray
            Absolute predictions
        """
        delta = self._delta_for(rate)
        absolute = np.zeros_like(predictions, dtype=np.float32)
        for i, col in enumerate(col_names):
            scale = delta.get_scale(col)
            denorm_deltas = predictions[:, :, i] * scale
            absolute[:, :, i] = last_data[:, i:i+1] + np.cumsum(denorm_deltas, axis=1)
        return absolute

    def inverse_delta_torch(
        self,
        predictions: 'torch.Tensor',
        last_data: 'torch.Tensor',
        col_names: List[str],
        rate: Optional[int] = None,
    ) -> 'torch.Tensor':
        """
        Convert normalized deltas back to absolute values (differentiable torch version).
        
        Used for physics-informed training where gradients need to flow through
        the inverse transform.
        
        Parameters
        ----------
        predictions : torch.Tensor
            Normalized delta predictions of shape (batch, pred_steps, n_outputs)
        last_data : torch.Tensor
            Last known outputs of shape (batch, n_outputs)
        col_names : List[str]
            Output column names
            
        Returns
        -------
        torch.Tensor
            Absolute predictions
        """
        if not HAS_TORCH:
            raise ImportError("inverse_delta_torch requires PyTorch")
        
        B, K, D = predictions.shape
        delta = self._delta_for(rate)
        absolute = torch.zeros_like(predictions)

        for i, col in enumerate(col_names):
            scale = delta.get_scale(col)
            denorm_deltas = predictions[:, :, i] * scale
            absolute[:, :, i] = last_data[:, i].unsqueeze(1) + torch.cumsum(denorm_deltas, dim=1)

        return absolute
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'FederatedNormalizer',
            'subsample_factor': self.subsample_factor,
            'scale_factor': self.scale_factor,
            'input_cols': self.input_cols,
            'output_cols': self.output_cols,
            'input_normalizer': self.input_normalizer.to_dict(),
            'output_normalizer': self.output_normalizer.to_dict(),
            'delta_normalizer': self.delta_normalizer.to_dict(),
            'delta_by_rate': {str(r): d.to_dict() for r, d in self.delta_by_rate.items()},
            'is_fitted': self.is_fitted,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FederatedNormalizer':
        """Create from dictionary."""
        normalizer = cls(
            subsample_factor=d.get('subsample_factor', 10),
            scale_factor=d.get('scale_factor', 10.0),
        )
        normalizer.input_cols = d['input_cols']
        normalizer.output_cols = d['output_cols']
        normalizer.input_normalizer = ZScoreNormalizer.from_dict(d['input_normalizer'])
        normalizer.output_normalizer = ZScoreNormalizer.from_dict(d['output_normalizer'])
        normalizer.delta_normalizer = DeltaNormalizer.from_dict(d['delta_normalizer'])
        normalizer.delta_by_rate = {
            int(r): DeltaNormalizer.from_dict(dd)
            for r, dd in d.get('delta_by_rate', {}).items()
        }
        normalizer.is_fitted = d.get('is_fitted', True)
        return normalizer
    
    def save(self, path: Union[str, Path]) -> None:
        """Save normalizer state to file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"FederatedNormalizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FederatedNormalizer':
        """Load normalizer state from file."""
        path = Path(path)
        with open(path, 'r') as f:
            state = json.load(f)
        normalizer = cls.from_dict(state)
        print(f"FederatedNormalizer loaded from {path}")
        return normalizer


class SurrogateNormalizer:
    """
    Unified normalizer manager for surrogate models.
    
    Manages input normalization, output normalization, delta normalization,
    and optional input whitening. Supports save/load for reproducibility.
    Works across all phases (1-6).
    """
    
    def __init__(
        self,
        use_whitening: bool = False,
        whitening_components: float = 0.99,
        delta_scale_factor: float = 10.0,
    ):
        """
        Initialize SurrogateNormalizer.
        
        Parameters
        ----------
        use_whitening : bool
            Whether to apply PCA whitening to inputs
        whitening_components : float
            Fraction of variance to retain in PCA
        delta_scale_factor : float
            Scale factor for delta normalization
        """
        self.use_whitening = use_whitening
        self.whitening_components = whitening_components
        self.delta_scale_factor = delta_scale_factor
        
        self.input_normalizer = ZScoreNormalizer()
        self.output_normalizer = ZScoreNormalizer()
        self.delta_normalizer = DeltaNormalizer(scale_factor=delta_scale_factor)
        self.input_whitener: Optional[InputWhitener] = None
        
        self.input_cols: List[str] = []
        self.output_cols: List[str] = []
        self.temporal_cols: List[str] = []
        self.is_fitted = False
    
    def fit(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray,
        input_cols: List[str],
        output_cols: List[str],
        temporal_cols: Optional[List[str]] = None,
        subsample_factor: int = 1,
    ) -> 'SurrogateNormalizer':
        """
        Fit all normalizers to training data.
        
        Parameters
        ----------
        input_data : np.ndarray
            Input data array
        output_data : np.ndarray
            Output data array
        input_cols : List[str]
            Input column names
        output_cols : List[str]
            Output column names
        temporal_cols : List[str], optional
            Temporal output column names (for delta normalizer)
        subsample_factor : int
            Subsample factor for delta computation
            
        Returns
        -------
        SurrogateNormalizer
            Self for method chaining
        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.temporal_cols = temporal_cols or output_cols
        
        # Fit input normalizer
        print("Fitting input normalizer...")
        self.input_normalizer.fit(input_data, input_cols)
        
        # Fit output normalizer
        print("Fitting output normalizer...")
        self.output_normalizer.fit(output_data, output_cols)
        
        # Fit delta normalizer on temporal columns
        if temporal_cols:
            print("Fitting delta normalizer...")
            temporal_indices = [output_cols.index(c) for c in temporal_cols if c in output_cols]
            temporal_data = output_data[:, temporal_indices]
            temporal_col_names = [output_cols[i] for i in temporal_indices]
            self.delta_normalizer.fit(temporal_data, temporal_col_names, subsample_factor)
        
        # Fit input whitener if enabled
        if self.use_whitening:
            print("Fitting input whitener...")
            input_normalized = self.input_normalizer.transform(input_data, input_cols)
            input_subsampled = input_normalized[::subsample_factor]
            self.input_whitener = InputWhitener(self.whitening_components)
            self.input_whitener.fit(input_subsampled)
        
        self.is_fitted = True
        return self
    
    def transform_input(self, data: np.ndarray, apply_whitening: bool = True) -> np.ndarray:
        """
        Transform input data.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array
        apply_whitening : bool
            Whether to apply whitening (if fitted)
            
        Returns
        -------
        np.ndarray
            Normalized (and optionally whitened) input
        """
        normalized = self.input_normalizer.transform(data, self.input_cols)
        if apply_whitening and self.input_whitener is not None:
            return self.input_whitener.transform(normalized)
        return normalized
    
    def transform_output(self, data: np.ndarray) -> np.ndarray:
        """Transform output data."""
        return self.output_normalizer.transform(data, self.output_cols)
    
    def inverse_output(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform output data."""
        return self.output_normalizer.inverse_transform(data, self.output_cols)
    
    def inverse_delta(
        self,
        predictions: np.ndarray,
        last_values: np.ndarray,
        col_names: List[str],
    ) -> np.ndarray:
        """
        Convert normalized delta predictions to absolute values.
        
        Parameters
        ----------
        predictions : np.ndarray
            Normalized delta predictions of shape (batch, pred_steps, n_outputs)
        last_values : np.ndarray
            Last observed values of shape (batch, n_outputs)
        col_names : List[str]
            Column names for the outputs
            
        Returns
        -------
        np.ndarray
            Absolute predictions
        """
        absolute = np.zeros_like(predictions, dtype=np.float32)
        for i, col in enumerate(col_names):
            scale = self.delta_normalizer.get_scale(col)
            denorm_deltas = predictions[:, :, i] * scale
            absolute[:, :, i] = last_values[:, i:i+1] + np.cumsum(denorm_deltas, axis=1)
        return absolute
    
    @property
    def whitened_input_dim(self) -> int:
        """Get whitened input dimension (or original if no whitening)."""
        if self.input_whitener is not None:
            return self.input_whitener.output_dim
        return len(self.input_cols)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save normalizer state to file.
        
        Parameters
        ----------
        path : str or Path
            Output file path (JSON format)
        """
        path = Path(path)
        state = {
            'use_whitening': self.use_whitening,
            'whitening_components': self.whitening_components,
            'delta_scale_factor': self.delta_scale_factor,
            'input_cols': self.input_cols,
            'output_cols': self.output_cols,
            'temporal_cols': self.temporal_cols,
            'is_fitted': self.is_fitted,
            'input_normalizer': self.input_normalizer.to_dict(),
            'output_normalizer': self.output_normalizer.to_dict(),
            'delta_normalizer': self.delta_normalizer.to_dict(),
        }
        if self.input_whitener is not None:
            state['input_whitener'] = self.input_whitener.to_dict()
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Normalizer saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'SurrogateNormalizer':
        """
        Load normalizer state from file.
        
        Parameters
        ----------
        path : str or Path
            Input file path
            
        Returns
        -------
        SurrogateNormalizer
            Loaded normalizer
        """
        path = Path(path)
        with open(path, 'r') as f:
            state = json.load(f)
        
        normalizer = cls(
            use_whitening=state['use_whitening'],
            whitening_components=state['whitening_components'],
            delta_scale_factor=state['delta_scale_factor'],
        )
        normalizer.input_cols = state['input_cols']
        normalizer.output_cols = state['output_cols']
        normalizer.temporal_cols = state['temporal_cols']
        normalizer.is_fitted = state['is_fitted']
        
        normalizer.input_normalizer = ZScoreNormalizer.from_dict(state['input_normalizer'])
        normalizer.output_normalizer = ZScoreNormalizer.from_dict(state['output_normalizer'])
        normalizer.delta_normalizer = DeltaNormalizer.from_dict(state['delta_normalizer'])
        
        if 'input_whitener' in state:
            normalizer.input_whitener = InputWhitener.from_dict(state['input_whitener'])
        
        print(f"Normalizer loaded from {path}")
        return normalizer


__all__ = [
    'ZScoreNormalizer',
    'DeltaNormalizer',
    'InputWhitener',
    'DomainNormalizer',
    'FederatedNormalizer',
    'SurrogateNormalizer',
]