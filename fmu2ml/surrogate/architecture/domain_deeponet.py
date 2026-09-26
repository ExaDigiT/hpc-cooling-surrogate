"""
Phase 4: Domain-specific DeepONet models.

Each domain (temperature, flow, pressure, power) has its own expert DeepONet
with shared architecture but independent parameters.

Architecture:
    Branch: LSTM → LayerNorm → Attention Pooling → Linear → N_BASIS × output_size
    Trunk: Fourier Features → 2-layer MLP → N_BASIS basis weights
    Output: einsum(trunk, branch) → scale + bias → (optional) skip connection
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from .configs import DomainDeepONetConfig, DOMAIN_OUTPUTS
from .deeponet import FourierFeatures
from .multirate import MultiRateOperatorModel


class DomainBranchNetwork(nn.Module):
    """
    Branch network for domain-specific DeepONet.
    
    Encodes input history via LSTM with attention pooling.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        n_basis: int,
        output_size: int,
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
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False,
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
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
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # Produce basis coefficients
        out = self.head(context)
        return out.view(batch_size, self.n_basis, self.output_size)


class DomainTrunkNetwork(nn.Module):
    """
    Trunk network for domain-specific DeepONet.
    
    Uses Fourier features for temporal encoding.
    """
    
    def __init__(
        self,
        n_basis: int,
        hidden_size: int,
        n_fourier_freqs: int,
        prediction_steps: int,
    ):
        super().__init__()
        self.n_basis = n_basis
        
        # Fourier feature encoding
        freqs = torch.linspace(1, n_fourier_freqs, n_fourier_freqs) * np.pi
        self.register_buffer('freqs', freqs)
        self.register_buffer(
            'query_times',
            torch.linspace(0, 1, prediction_steps).view(-1, 1),
        )
        
        trunk_input_size = 1 + 2 * n_fourier_freqs
        self.net = nn.Sequential(
            nn.Linear(trunk_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_basis),
        )
    
    def _get_trunk_features(self) -> torch.Tensor:
        """Compute Fourier features for query times."""
        t = self.query_times
        sin_features = torch.sin(t * self.freqs)
        cos_features = torch.cos(t * self.freqs)
        return torch.cat([t, sin_features, cos_features], dim=-1)
    
    def forward(self) -> torch.Tensor:
        """
        Returns:
            (prediction_steps, n_basis) basis function values
        """
        trunk_features = self._get_trunk_features()
        return self.net(trunk_features)


class DomainDeepONet(nn.Module):
    """
    Domain-specific DeepONet architecture.
    
    A unified DeepONet for domain-specific models (temperature, flow, pressure, power).
    Each domain gets its own instance with independent parameters.
    
    Architecture:
        Branch: LSTM → LayerNorm → Attention Pooling → Linear → N_BASIS × output_size
        Trunk: Fourier Features → 2-layer MLP → N_BASIS basis weights
        Output: einsum(trunk, branch) → scale + bias → (optional) skip connection
    
    Parameters
    ----------
    input_size : int
        Size of input features (inputs + output history if enabled)
    output_size : int
        Number of outputs for this domain
    prediction_steps : int
        Number of future steps to predict
    domain : str
        Domain name ('temperature', 'flow', 'pressure', 'power')
    branch_hidden : int
        Hidden size for branch LSTM
    trunk_hidden : int
        Hidden size for trunk MLP
    n_basis : int
        Number of basis functions
    n_lstm_layers : int
        Number of LSTM layers
    n_fourier_freqs : int
        Number of Fourier frequencies
    dropout : float
        Dropout rate
    use_skip_connection : bool
        Whether to use learnable skip connection
    initial_skip_alpha : float
        Initial value for skip connection weight
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        prediction_steps: int,
        domain: str = "temperature",
        branch_hidden: int = 128,
        trunk_hidden: int = 64,
        n_basis: int = 32,
        n_lstm_layers: int = 2,
        n_fourier_freqs: int = 8,
        dropout: float = 0.3,
        use_skip_connection: bool = True,
        initial_skip_alpha: float = 0.5,
        output_cols: Optional[List[str]] = None,
        column_info: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.prediction_steps = prediction_steps
        self.domain = domain
        self.n_basis = n_basis
        self.use_skip_connection = use_skip_connection
        self.output_cols = output_cols
        self.column_info = column_info

        # Branch network
        self.branch = DomainBranchNetwork(
            input_size=input_size,
            hidden_size=branch_hidden,
            n_layers=n_lstm_layers,
            n_basis=n_basis,
            output_size=output_size,
            dropout=dropout,
        )

        # Trunk network
        self.trunk = DomainTrunkNetwork(
            n_basis=n_basis,
            hidden_size=trunk_hidden,
            n_fourier_freqs=n_fourier_freqs,
            prediction_steps=prediction_steps,
        )

        # Optional skip connection
        if use_skip_connection:
            initial_logit = np.log(initial_skip_alpha / (1 - initial_skip_alpha + 1e-8))
            self.skip_alpha_logit = nn.Parameter(
                torch.ones(output_size) * initial_logit
            )
        else:
            self.skip_alpha_logit = None

        # Output scaling and bias per output
        self.output_scale = nn.Parameter(torch.ones(output_size) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(output_size))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.3)
            elif 'bias' in name and 'output_bias' not in name and 'skip' not in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, history_steps, input_size)
            return_components: Whether to return intermediate outputs
            
        Returns:
            Dict with:
            - 'predictions': (batch, pred_steps, output_size)
            - 'alpha': skip connection weights (if enabled)
            - 'attention_weights', 'context', 'trunk_out' (if return_components)
        """
        batch_size = x.shape[0]

        # Branch: encode history
        branch_out = self.branch(x)  # (batch, n_basis, output_size)

        # Trunk: encode query times
        trunk_out = self.trunk()  # (pred_steps, n_basis)

        # Combine via einsum: (pred_steps, n_basis) @ (batch, n_basis, output_size)
        out = torch.einsum('pn,bno->bpo', trunk_out, branch_out)

        # Scale and bias
        out = out * self.output_scale + self.output_bias

        # Apply skip connection if enabled
        result = {}
        if self.skip_alpha_logit is not None:
            alpha = torch.sigmoid(self.skip_alpha_logit)
            out = alpha.view(1, 1, -1) * out
            result['alpha'] = alpha
        else:
            result['alpha'] = None

        result['predictions'] = out

        if return_components:
            # Note: for full component access, would need to modify branch network
            result['trunk_out'] = trunk_out

        return result

    @classmethod
    def from_config(
        cls,
        config: DomainDeepONetConfig,
        input_size: int,
        output_size: int,
        output_cols: Optional[List[str]] = None,
        column_info: Optional[Dict[str, Any]] = None,
    ) -> 'DomainDeepONet':
        """Create model from configuration."""
        return cls(
            input_size=input_size,
            output_size=output_size,
            prediction_steps=config.prediction_steps,
            domain=config.domain,
            branch_hidden=config.branch_lstm_hidden,
            trunk_hidden=config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 64,
            n_basis=config.basis_dim,
            n_lstm_layers=config.branch_lstm_layers,
            n_fourier_freqs=config.trunk_n_fourier,
            dropout=config.branch_lstm_dropout,
            use_skip_connection=True,
            initial_skip_alpha=0.5,
            output_cols=output_cols,
            column_info=column_info,
        )

    def __repr__(self) -> str:
        return (
            f"DomainDeepONet(\n"
            f"  domain={self.domain},\n"
            f"  input_size={self.input_size},\n"
            f"  output_size={self.output_size},\n"
            f"  prediction_steps={self.prediction_steps},\n"
            f"  n_basis={self.n_basis},\n"
            f"  use_skip_connection={self.use_skip_connection}\n"
            f")"
        )


class MultiRateDomainDeepONet(MultiRateOperatorModel):
    """Phase 4 on the multi-rate task: the operator core with phase 4's
    domain-head identity on every group — per-output scale+bias and a sigmoid
    skip gate (``head_mode='skip'``). The single-domain restriction of the
    legacy model is lifted: each canonical group IS a domain head here."""

    def __init__(self, config: DomainDeepONetConfig, column_info: Dict[str, Any]):
        super().__init__(
            config, column_info, head_mode='skip',
            encoder_hidden=config.branch_lstm_hidden,
            encoder_layers=config.branch_lstm_layers,
            n_basis=config.basis_dim,
            trunk_hidden=config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 64,
            n_fourier=getattr(config, 'trunk_n_fourier', 8),
            dropout=config.branch_lstm_dropout,
        )


def domain_deeponet(
    config: DomainDeepONetConfig,
    input_size: Optional[int] = None,
    output_size: Optional[int] = None,
    column_info: Optional[Dict[str, Any]] = None,
) -> DomainDeepONet:
    """
    Factory function to create Phase 4 DomainDeepONet model.
    
    Parameters
    ----------
    config : DomainDeepONetConfig
        Domain-specific configuration
    input_size : int, optional
        Input size (inferred from column_info if not provided)
    output_size : int, optional
        Output size (inferred from column_info if not provided)
    column_info : Dict[str, Any], optional
        Column information dictionary
        
    Returns
    -------
    DomainDeepONet
        Instantiated model (``MultiRateDomainDeepONet`` when ``config.multi_rate``)
    """
    # Multi-rate mode: group-structured model on the chunk-store task
    if getattr(config, 'multi_rate', False):
        if column_info is None:
            raise ValueError("multi-rate domain_deeponet requires column_info")
        return MultiRateDomainDeepONet(config, column_info)

    output_cols = None
    
    if column_info is not None:
        n_inputs = len(column_info.get('input_cols', []))
        n_outputs = len(column_info.get('output_cols', []))
        output_cols = column_info.get('output_cols', [])
        
        if input_size is None:
            if config.include_output_history:
                input_size = n_inputs + n_outputs
            else:
                input_size = n_inputs
        
        if output_size is None:
            output_size = n_outputs
    
    if input_size is None or output_size is None:
        raise ValueError("input_size and output_size must be provided or inferable from column_info")
    
    return DomainDeepONet.from_config(
        config,
        input_size=input_size,
        output_size=output_size,
        output_cols=output_cols,
        column_info=column_info,
    )


def create_domain_models(
    configs: Dict[str, DomainDeepONetConfig],
    domain_column_info: Dict[str, Tuple[List[str], List[str], Dict[str, Any]]],
) -> Dict[str, DomainDeepONet]:
    """
    Create models for all domains.
    
    Parameters
    ----------
    configs : Dict[str, DomainDeepONetConfig]
        Dictionary mapping domain name to config
    domain_column_info : Dict[str, Tuple]
        Dictionary mapping domain name to (input_cols, output_cols, column_info)
        
    Returns
    -------
    Dict[str, DomainDeepONet]
        Dictionary mapping domain name to model
    """
    models = {}
    
    for domain_name, config in configs.items():
        if domain_name in domain_column_info:
            input_cols, output_cols, col_info = domain_column_info[domain_name]
            n_inputs = len(input_cols)
            n_outputs = len(output_cols)
            
            if config.include_output_history:
                input_size = n_inputs + n_outputs
            else:
                input_size = n_inputs
            
            models[domain_name] = DomainDeepONet.from_config(
                config,
                input_size=input_size,
                output_size=n_outputs,
                output_cols=output_cols,
                column_info=col_info,
            )
            print(f"Created {domain_name} model: input={input_size}, output={n_outputs}")
    
    return models


__all__ = [
    'DomainBranchNetwork',
    'DomainTrunkNetwork',
    'DomainDeepONet',
    'MultiRateDomainDeepONet',
    'domain_deeponet',
    'create_domain_models',
]