"""
Phase 2: Basic DeepONet with separate temporal and algebraic pathways.

Architecture:
- Temporal pathway: DeepONet (LSTM branch + Fourier trunk) for dynamic outputs
- Algebraic pathway: MLP for instantaneous Q_flow → pressure relationships
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from .configs import DeepONetConfig
from .multirate import MultiRateOperatorModel


class FourierFeatures(nn.Module):
    """
    Fourier feature encoding for query times.
    
    Maps scalar time values to a higher-dimensional representation
    using sine and cosine features at multiple frequencies.
    """
    
    def __init__(self, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        freqs = torch.linspace(1, n_freqs, n_freqs) * np.pi
        self.register_buffer('freqs', freqs)
    
    @property
    def output_dim(self) -> int:
        """Output dimension: 1 (raw time) + 2 * n_freqs (sin + cos)."""
        return 1 + 2 * self.n_freqs
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (n_times, 1) or (n_times,) time values in [0, 1]
        Returns:
            (n_times, 1 + 2*n_freqs) Fourier features
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        sin_features = torch.sin(t * self.freqs)
        cos_features = torch.cos(t * self.freqs)
        return torch.cat([t, sin_features, cos_features], dim=-1)


class BranchNetwork(nn.Module):
    """
    Branch network: LSTM encoder with attention for history encoding.
    
    Encodes the input history sequence into basis coefficients.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        n_basis: int = 32,
        output_size: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_basis = n_basis
        self.output_size = output_size
        
        # LSTM encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Output head: produces basis coefficients per output
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_basis * output_size),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.3)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size) history sequence
        Returns:
            (batch, n_basis, output_size) basis coefficients
        """
        batch_size = x.shape[0]
        
        # Encode sequence
        lstm_out, _ = self.encoder(x)
        lstm_out = self.norm(lstm_out)
        
        # Attention pooling
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden)
        
        # Produce basis coefficients
        out = self.head(context)  # (batch, n_basis * output_size)
        out = out.view(batch_size, self.n_basis, self.output_size)
        
        return out


class TrunkNetwork(nn.Module):
    """
    Trunk network: MLP that maps query times to basis functions.
    
    Uses Fourier features for improved time representation.
    """
    
    def __init__(
        self,
        n_basis: int = 32,
        hidden_size: int = 64,
        n_fourier_freqs: int = 8,
    ):
        super().__init__()
        self.n_basis = n_basis
        
        # Fourier feature encoding
        self.fourier = FourierFeatures(n_freqs=n_fourier_freqs)
        
        # MLP
        self.net = nn.Sequential(
            nn.Linear(self.fourier.output_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_basis),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (n_times, 1) query times in [0, 1]
        Returns:
            (n_times, n_basis) basis function values
        """
        features = self.fourier(t)
        return self.net(features)


class TemporalDeepONet(nn.Module):
    """
    DeepONet for temporal/dynamical outputs.
    
    Uses history to predict future deltas via branch-trunk decomposition:
    - Branch: encodes input history into basis coefficients
    - Trunk: encodes query times into basis functions
    - Output: inner product of branch and trunk outputs
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        prediction_steps: int,
        lstm_hidden: int = 64,
        trunk_hidden: int = 64,
        n_basis: int = 32,
        n_fourier_freqs: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.output_size = output_size
        self.prediction_steps = prediction_steps
        self.n_basis = n_basis

        # Branch network
        self.branch = BranchNetwork(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            n_basis=n_basis,
            output_size=output_size,
            dropout=dropout,
        )

        # Trunk network
        self.trunk = TrunkNetwork(
            n_basis=n_basis,
            hidden_size=trunk_hidden,
            n_fourier_freqs=n_fourier_freqs,
        )

        # Register query times as buffer
        self.register_buffer(
            'query_times',
            torch.linspace(0, 1, prediction_steps).view(-1, 1),
        )

        # Output scaling and bias
        self.output_scale = nn.Parameter(torch.ones(output_size) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, history_steps, input_size)
        Returns:
            (batch, prediction_steps, output_size) predicted deltas
        """
        # Branch: encode history
        branch_out = self.branch(x)  # (batch, n_basis, output_size)

        # Trunk: encode query times
        trunk_out = self.trunk(self.query_times)  # (pred_steps, n_basis)

        # Combine via einsum: (pred_steps, n_basis) @ (batch, n_basis, output_size)
        out = torch.einsum('pn,bno->bpo', trunk_out, branch_out)

        # Scale and bias
        out = out * self.output_scale + self.output_bias
        return out


class AlgebraicPathway(nn.Module):
    """
    MLP for instantaneous (algebraic) relationships.
    
    Maps current Q_flow directly to secondary pressures.
    Based on analysis: p_sec = f(Q_flow) with R² > 0.96.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        layers = []
        # First layer
        layers.extend([
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        # Hidden layers
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_qflow: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_qflow: (batch, prediction_steps, n_qflow_inputs)
        Returns:
            (batch, prediction_steps, n_algebraic_outputs)
        """
        batch_size, pred_steps, _ = x_qflow.shape
        x_flat = x_qflow.reshape(batch_size * pred_steps, -1)
        out_flat = self.mlp(x_flat)
        return out_flat.reshape(batch_size, pred_steps, -1)


class BasicDeepONet(nn.Module):
    """
    Phase 2: Basic Hybrid DeepONet.
    
    Combines:
    1. Temporal pathway: DeepONet for history → temporal output deltas
    2. Algebraic pathway: MLP for Q_flow → algebraic outputs (instantaneous)
    
    Parameters
    ----------
    temporal_input_size : int
        Input size for temporal pathway (may include output history)
    temporal_output_size : int
        Number of temporal outputs
    algebraic_input_size : int
        Number of Q_flow inputs for algebraic pathway
    algebraic_output_size : int
        Number of algebraic outputs
    prediction_steps : int
        Number of future steps to predict
    lstm_hidden : int
        Hidden size for LSTM in branch network
    trunk_hidden : int
        Hidden size for trunk network MLP
    n_basis : int
        Number of basis functions
    algebraic_hidden : int
        Hidden size for algebraic MLP
    algebraic_layers : int
        Number of layers in algebraic MLP
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        temporal_input_size: int,
        temporal_output_size: int,
        algebraic_input_size: int,
        algebraic_output_size: int,
        prediction_steps: int,
        lstm_hidden: int = 64,
        trunk_hidden: int = 64,
        n_basis: int = 32,
        algebraic_hidden: int = 64,
        algebraic_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.temporal_output_size = temporal_output_size
        self.algebraic_output_size = algebraic_output_size
        self.prediction_steps = prediction_steps

        self.temporal_pathway = TemporalDeepONet(
            input_size=temporal_input_size,
            output_size=temporal_output_size,
            prediction_steps=prediction_steps,
            lstm_hidden=lstm_hidden,
            trunk_hidden=trunk_hidden,
            n_basis=n_basis,
            dropout=dropout,
        )

        self.algebraic_pathway = AlgebraicPathway(
            input_size=algebraic_input_size,
            output_size=algebraic_output_size,
            hidden_size=algebraic_hidden,
            n_layers=algebraic_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_algebraic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_temporal: (batch, history_steps, temporal_input_size)
            x_algebraic: (batch, prediction_steps, algebraic_input_size)
        Returns:
            y_temporal: (batch, prediction_steps, temporal_output_size)
            y_algebraic: (batch, prediction_steps, algebraic_output_size)
        """
        y_temporal = self.temporal_pathway(x_temporal)
        y_algebraic = self.algebraic_pathway(x_algebraic)
        return y_temporal, y_algebraic

    @classmethod
    def from_config(
        cls,
        config: DeepONetConfig,
        temporal_input_size: int,
        temporal_output_size: int,
        algebraic_input_size: int,
        algebraic_output_size: int,
    ) -> 'BasicDeepONet':
        """Create model from configuration."""
        return cls(
            temporal_input_size=temporal_input_size,
            temporal_output_size=temporal_output_size,
            algebraic_input_size=algebraic_input_size,
            algebraic_output_size=algebraic_output_size,
            prediction_steps=config.prediction_steps,
            lstm_hidden=config.branch_lstm_hidden,
            trunk_hidden=config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 64,
            n_basis=config.basis_dim,
            algebraic_hidden=config.algebraic_mlp_hidden[0] if config.algebraic_mlp_hidden else 64,
            algebraic_layers=len(config.algebraic_mlp_hidden) + 1 if config.algebraic_mlp_hidden else 3,
            dropout=config.branch_lstm_dropout,
        )


class MultiRateDeepONet(MultiRateOperatorModel):
    """Phase 2 on the multi-rate task: per-branch LSTM basis coefficients x
    per-head Fourier trunk inner product — the raw DeepONet identity
    (``head_mode='plain'``, no gates, no scale/bias decoration)."""

    def __init__(self, config: DeepONetConfig, column_info: Dict[str, Any]):
        super().__init__(
            config, column_info, head_mode='plain',
            encoder_hidden=config.branch_lstm_hidden,
            encoder_layers=config.branch_lstm_layers,
            n_basis=config.basis_dim,
            trunk_hidden=config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 64,
            n_fourier=getattr(config, 'trunk_n_fourier', 8),
            dropout=config.branch_lstm_dropout,
        )


def deeponet(
    config: DeepONetConfig,
    column_info: Optional[Dict[str, Any]] = None,
    temporal_input_size: Optional[int] = None,
    temporal_output_size: Optional[int] = None,
    algebraic_input_size: Optional[int] = None,
    algebraic_output_size: Optional[int] = None,
) -> BasicDeepONet:
    """
    Factory function to create Phase 2 BasicDeepONet model.
    
    Parameters
    ----------
    config : DeepONetConfig
        Model configuration
    column_info : Dict[str, Any], optional
        Column information for inferring sizes
    temporal_input_size : int, optional
        Input size for temporal pathway
    temporal_output_size : int, optional
        Number of temporal outputs
    algebraic_input_size : int, optional
        Number of Q_flow inputs
    algebraic_output_size : int, optional
        Number of algebraic outputs
        
    Returns
    -------
    BasicDeepONet
        Instantiated model (``MultiRateDeepONet`` when ``config.multi_rate``)
    """
    # Multi-rate mode: group-structured model on the chunk-store task
    if getattr(config, 'multi_rate', False):
        if column_info is None:
            raise ValueError("multi-rate deeponet requires column_info")
        return MultiRateDeepONet(config, column_info)

    if column_info is not None:
        n_inputs = column_info.get('n_inputs', len(column_info.get('input_cols', [])))
        n_temporal = column_info.get('n_temporal', len(column_info.get('temporal_cols', [])))
        n_algebraic = column_info.get('n_algebraic', len(column_info.get('algebraic_cols', [])))
        
        if temporal_input_size is None:
            n_inputs_actual = column_info.get('whitened_n_inputs', n_inputs)
            if config.include_output_history:
                temporal_input_size = n_inputs_actual + n_temporal
            else:
                temporal_input_size = n_inputs_actual
        
        if temporal_output_size is None:
            temporal_output_size = n_temporal
        
        if algebraic_input_size is None:
            # Number of Q_flow columns (typically num_cdus)
            algebraic_input_size = config.num_cdus
        
        if algebraic_output_size is None:
            algebraic_output_size = n_algebraic
    
    if any(x is None for x in [temporal_input_size, temporal_output_size, 
                                algebraic_input_size, algebraic_output_size]):
        raise ValueError("All sizes must be provided or inferable from column_info")
    
    return BasicDeepONet.from_config(
        config,
        temporal_input_size=temporal_input_size,
        temporal_output_size=temporal_output_size,
        algebraic_input_size=algebraic_input_size,
        algebraic_output_size=algebraic_output_size,
    )


__all__ = [
    'FourierFeatures',
    'BranchNetwork',
    'TrunkNetwork',
    'TemporalDeepONet',
    'AlgebraicPathway',
    'BasicDeepONet',
    'MultiRateDeepONet',
    'deeponet',
]