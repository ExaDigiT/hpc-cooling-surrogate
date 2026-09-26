"""
Phase 5: Federated DeepMMNet.

Shared temporal encoder with 6 specialized decoder heads:
- G_T: Temperature (standard decoder)
- G_V: Primary flow (standard decoder)
- G_p: Primary pressure (standard decoder)
- G_Vs: Secondary flow (skip decoder)
- G_ps: Secondary pressure (skip decoder)
- G_W: Pump power (skip decoder)

Fusion: z[k] = branch(u) ⊙ tbranch(y) ⊙ trunk(t_k) (Hadamard product)
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn

from .configs import FederatedConfig, zero_persistence_heads


class BranchNetwork(nn.Module):
    """
    Branch network: Encodes input history u(t) → latent basis coefficients.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        n_basis: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_basis),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            (batch, n_basis) basis coefficients
        """
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm(lstm_out)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_w, dim=1)
        return self.projection(context)


class TBranchNetwork(nn.Module):
    """
    T-Branch network: Encodes output state history y(t) → latent basis coefficients.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        n_layers: int,
        n_basis: int,
        dropout: float,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_basis),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: (batch, seq_len, input_size) output history
        Returns:
            (batch, n_basis) basis coefficients
        """
        lstm_out, _ = self.lstm(y)
        lstm_out = self.norm(lstm_out)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        context = torch.sum(lstm_out * attn_w, dim=1)
        return self.projection(context)


class AttentionEncoder(nn.Module):
    """
    Transformer-encoder temporal backbone with attention pooling.

    Drop-in for ``BranchNetwork``/``TBranchNetwork`` (same ``(B, seq, input_size)
    -> (B, n_basis)`` contract); selected by ``config.encoder_type == "attention"``.
    Handles the variable-length per-branch windows of the multi-rate model.
    """

    def __init__(self, input_size: int, hidden_size: int, n_layers: int,
                 n_basis: int, dropout: float, n_heads: int = 4):
        super().__init__()
        if hidden_size % n_heads != 0:
            n_heads = max(1, _largest_divisor_leq(hidden_size, 8))
        self.in_proj = nn.Linear(input_size, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, n_basis),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.in_proj(x))
        h = self.norm(h)
        w = torch.softmax(self.attention(h), dim=1)
        return self.projection(torch.sum(h * w, dim=1))


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is <= ``cap`` (>=1)."""
    for d in range(min(cap, n), 0, -1):
        if n % d == 0:
            return d
    return 1


def make_temporal_encoder(encoder_type: str, input_size: int, hidden_size: int,
                          n_layers: int, n_basis: int, dropout: float) -> nn.Module:
    """Build a temporal encoder backbone (``"lstm"`` or ``"attention"``)."""
    if encoder_type == 'attention':
        return AttentionEncoder(input_size, hidden_size, n_layers, n_basis, dropout)
    return BranchNetwork(input_size, hidden_size, n_layers, n_basis, dropout)


class FourierTrunkNetwork(nn.Module):
    """
    Trunk network: Fourier temporal encoding → basis functions.
    """

    def __init__(
        self,
        n_fourier: int,
        hidden_size: int,
        n_basis: int,
        prediction_steps: int,
    ):
        super().__init__()
        self.n_basis = n_basis
        
        trunk_input_size = 1 + 2 * n_fourier
        self.net = nn.Sequential(
            nn.Linear(trunk_input_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, n_basis),
        )
        
        freqs = torch.linspace(1, n_fourier, n_fourier) * np.pi
        self.register_buffer('freqs', freqs)
        self.register_buffer(
            'query_times',
            torch.linspace(0, 1, prediction_steps).view(-1, 1),
        )

    def forward(self, K: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            K: Optional number of time steps (uses all if None)
        Returns:
            (K, n_basis) basis function values
        """
        t = self.query_times[:K] if K else self.query_times
        sin_f = torch.sin(t * self.freqs)
        cos_f = torch.cos(t * self.freqs)
        features = torch.cat([t, sin_f, cos_f], dim=-1)
        return self.net(features)


class DecoderHead(nn.Module):
    """
    Standard decoder head for dynamic output groups (G_T, G_V, G_p).
    """

    def __init__(
        self,
        n_basis: int,
        hidden_dim: int,
        n_outputs: int,
        dropout: float,
        name: str = '',
    ):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.net = nn.Sequential(
            nn.Linear(n_basis, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs),
        )
        self.output_scale = nn.Parameter(torch.ones(n_outputs) * 0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, K, n_basis) fused latent representation
        Returns:
            (batch, K, n_outputs) predictions
        """
        return self.net(z) * self.output_scale


class SkipDecoderHead(nn.Module):
    """
    Decoder head with skip connection for near-constant outputs (G_Vs, G_ps, G_W).
    
    output = bias + net(z) * scale
    Bias captures near-constant mean; MLP learns tiny deviations.
    """

    def __init__(
        self,
        n_basis: int,
        hidden_dim: int,
        n_outputs: int,
        dropout: float,
        name: str = '',
    ):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.net = nn.Sequential(
            nn.Linear(n_basis, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_outputs),
        )
        self.bias = nn.Parameter(torch.zeros(n_outputs))
        self.output_scale = nn.Parameter(torch.ones(n_outputs) * 0.01)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (batch, K, n_basis) fused latent representation
        Returns:
            (batch, K, n_outputs) predictions
        """
        return self.bias + self.net(z) * self.output_scale


class FederatedDeepMMNet(nn.Module):
    """
    Federated DeepM&Mnet with shared temporal encoder and 6 specialized decoder heads.
    
    Architecture:
    - Shared encoder: BranchNetwork (u-branch), TBranchNetwork (y-branch), FourierTrunkNetwork
    - 6 decoder heads: G_T, G_V, G_p (standard), G_Vs, G_ps, G_W (skip)
    - Fusion: z[k] = branch(u) ⊙ tbranch(y) ⊙ trunk(t_k) (Hadamard product)
    
    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_dynamic : int
        Total number of dynamic output features
    column_info : Dict[str, Any]
        Column information with head group indices
    config : FederatedConfig
        Model configuration
    """

    HEAD_NAMES = ['G_T', 'G_V', 'G_p', 'G_Vs', 'G_ps', 'G_W']

    def __init__(
        self,
        n_inputs: int,
        n_dynamic: int,
        column_info: Dict[str, Any],
        config: FederatedConfig,
    ):
        super().__init__()
        self.config = config
        self.n_inputs = n_inputs
        self.n_dynamic = n_dynamic

        # Extract head sizes from column_info
        n_T = len(column_info['temp_cols'])
        n_V = len(column_info['flow_cols'])
        n_p = len(column_info['pressure_cols'])
        n_Vs = len(column_info['flow_sec_cols'])
        n_ps = len(column_info['pressure_sec_cols'])
        n_W = len(column_info['power_cols'])

        # Store index ranges for assembly
        self.temp_indices = column_info['temp_indices']
        self.flow_indices = column_info['flow_indices']
        self.pressure_indices = column_info['pressure_indices']
        self.flow_sec_indices = column_info['flow_sec_indices']
        self.pressure_sec_indices = column_info['pressure_sec_indices']
        self.power_indices = column_info['power_indices']

        # Get config values
        branch_hidden = config.u_branch_lstm_hidden
        tbranch_hidden = config.y_branch_lstm_hidden
        trunk_hidden = config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 128
        n_basis = config.basis_dim
        n_layers = config.u_branch_lstm_layers
        dropout = config.u_branch_lstm_dropout
        n_fourier = config.trunk_n_fourier
        decoder_hidden = config.head_hidden_sizes.get('G_T', [256, 128])[0]
        decoder_hidden_small = config.head_hidden_sizes.get('G_Vs', [128, 64])[0]

        self.multi_rate = bool(getattr(config, 'multi_rate', False))
        enc_type = getattr(config, 'encoder_type', 'lstm')

        if not self.multi_rate:
            # ── Single shared encoder (legacy; byte-for-byte the original) ──
            self.branch = BranchNetwork(
                input_size=n_inputs, hidden_size=branch_hidden,
                n_layers=n_layers, n_basis=n_basis, dropout=dropout,
            )
            self.tbranch = TBranchNetwork(
                input_size=n_dynamic, hidden_size=tbranch_hidden,
                n_layers=n_layers, n_basis=n_basis, dropout=dropout,
            )
            self.trunk = FourierTrunkNetwork(
                n_fourier=n_fourier, hidden_size=trunk_hidden,
                n_basis=n_basis, prediction_steps=config.prediction_steps,
            )
        else:
            # ── Multi-rate: one (u,y) encoder per distinct branch id, one trunk
            # per group (its own K), plus a shared physics-grid trunk (W5b). ──
            branch_ids = sorted(set(config.encoder_groups.values()))
            self.branches = nn.ModuleDict({
                b: make_temporal_encoder(enc_type, n_inputs, branch_hidden,
                                         n_layers, n_basis, dropout)
                for b in branch_ids
            })
            self.tbranches = nn.ModuleDict({
                b: make_temporal_encoder(enc_type, n_dynamic, tbranch_hidden,
                                         n_layers, n_basis, dropout)
                for b in branch_ids
            })
            self.trunks = nn.ModuleDict({
                h: FourierTrunkNetwork(
                    n_fourier=n_fourier, hidden_size=trunk_hidden, n_basis=n_basis,
                    prediction_steps=config.head_prediction_steps[h])
                for h in self.HEAD_NAMES
            })
            self.trunk_phys = FourierTrunkNetwork(
                n_fourier=n_fourier, hidden_size=trunk_hidden, n_basis=n_basis,
                prediction_steps=config.physics_K,
            )

        # Primary decoder heads (standard)
        self.head_T = DecoderHead(n_basis, decoder_hidden, n_T, dropout, name='G_T')
        self.head_V = DecoderHead(n_basis, decoder_hidden, n_V, dropout, name='G_V')
        self.head_p = DecoderHead(n_basis, decoder_hidden, n_p, dropout, name='G_p')

        # Near-constant decoder heads (skip connection)
        self.head_Vs = SkipDecoderHead(n_basis, decoder_hidden_small, n_Vs, dropout, name='G_Vs')
        self.head_ps = SkipDecoderHead(n_basis, decoder_hidden_small, n_ps, dropout, name='G_ps')
        self.head_W = SkipDecoderHead(n_basis, decoder_hidden_small, n_W, dropout, name='G_W')

        # Map names → modules
        self._heads = {
            'G_T': self.head_T, 'G_V': self.head_V, 'G_p': self.head_p,
            'G_Vs': self.head_Vs, 'G_ps': self.head_ps, 'G_W': self.head_W,
        }
        self._pred_keys = {
            'G_T': 'pred_T', 'G_V': 'pred_V', 'G_p': 'pred_p',
            'G_Vs': 'pred_Vs', 'G_ps': 'pred_ps', 'G_W': 'pred_W',
        }
        self._index_map = {
            'G_T': self.temp_indices, 'G_V': self.flow_indices,
            'G_p': self.pressure_indices, 'G_Vs': self.flow_sec_indices,
            'G_ps': self.pressure_sec_indices, 'G_W': self.power_indices,
        }

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.3)
            elif 'bias' in name and 'output_scale' not in name:
                nn.init.zeros_(param)

    def get_encoder_params(self) -> List[nn.Parameter]:
        """Get parameters from shared encoder components (both modes)."""
        if self.multi_rate:
            params: List[nn.Parameter] = []
            for m in (self.branches, self.tbranches, self.trunks):
                params += list(m.parameters())
            params += list(self.trunk_phys.parameters())
            return params
        return (
            list(self.branch.parameters()) +
            list(self.tbranch.parameters()) +
            list(self.trunk.parameters())
        )

    def get_head_params(self, head_name: str) -> List[nn.Parameter]:
        """Get parameters for a specific decoder head."""
        return list(self._heads[head_name].parameters())

    def get_all_head_params(self) -> List[nn.Parameter]:
        """Get parameters from all decoder heads."""
        params = []
        for head in self._heads.values():
            params.extend(head.parameters())
        return params

    def encode(
        self,
        u_hist: torch.Tensor,
        y_hist: torch.Tensor,
        K: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode inputs and produce fused latent representation.
        
        Args:
            u_hist: (batch, H, n_inputs) input history
            y_hist: (batch, H, n_dynamic) output history
            K: Optional number of prediction steps
            
        Returns:
            z: (batch, K, n_basis) fused latent representation
        """
        b = self.branch(u_hist)       # (B, n_basis)
        tb = self.tbranch(y_hist)     # (B, n_basis)
        T = self.trunk(K=K)           # (K, n_basis)
        
        # Hadamard fusion: z[k] = b ⊙ tb ⊙ T[k]
        z = b.unsqueeze(1) * tb.unsqueeze(1) * T.unsqueeze(0)  # (B, K, n_basis)
        return z

    def forward(
        self,
        u_hist: torch.Tensor,
        y_hist: Optional[torch.Tensor] = None,
        K: Optional[int] = None,
        active_heads: Optional[List[str]] = None,
        physics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            u_hist: (batch, H, n_inputs) input history
            y_hist: (batch, H, n_dynamic) output history
            K: Optional number of prediction steps
            active_heads: Optional list of heads to compute (all if None)
            
        Returns:
            Dict with:
            - 'pred_T', 'pred_V', etc.: Individual head predictions
            - 'predictions': (batch, K, n_dynamic) assembled full predictions

        Multi-rate dispatch: when ``u_hist`` is the ``{branch_id: (u, y)}`` dict
        this delegates to :meth:`forward_multirate`. DistributedDataParallel
        drives the model through ``__call__`` -> ``forward``, so the multi-rate
        entry point has to be reachable that way; calling
        ``model.module.forward_multirate()`` under DDP would run correctly but
        silently skip the gradient all-reduce, leaving each rank with its own
        divergent weights.
        """
        if isinstance(u_hist, dict):
            return self.forward_multirate(
                u_hist, active_heads=active_heads, physics=physics)

        if y_hist is None:
            raise TypeError(
                "forward() needs y_hist for the single-rate path; pass the "
                "{branch_id: (u_hist_b, y_hist_b)} dict for multi-rate.")

        z = self.encode(u_hist, y_hist, K)
        B, K_actual, _ = z.shape
        result = {}

        # Compute predictions for each head
        for hname in self.HEAD_NAMES:
            if active_heads is None or hname in active_heads:
                pred_key = self._pred_keys[hname]
                result[pred_key] = self._heads[hname](z)

        # Assemble full dynamic prediction if all heads are active
        all_active = active_heads is None or set(active_heads) == set(self.HEAD_NAMES)
        if all_active:
            predictions = torch.zeros(B, K_actual, self.n_dynamic, device=z.device)
            for hname in self.HEAD_NAMES:
                pred_key = self._pred_keys[hname]
                indices = self._index_map[hname]
                if pred_key in result:
                    predictions[:, :, indices] = result[pred_key]
            result['predictions'] = predictions

        return result

    def _encode_branches(
        self, branch_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Per-branch fused code b⊙tb (B, n_basis), one entry per encoder branch."""
        enc: Dict[str, torch.Tensor] = {}
        for b, (u, y) in branch_inputs.items():
            enc[b] = self.branches[b](u) * self.tbranches[b](y)
        return enc

    def forward_multirate(
        self,
        branch_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        active_heads: Optional[List[str]] = None,
        physics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-rate forward.

        Args:
            branch_inputs: ``{branch_id: (u_hist_b, y_hist_b)}`` at each branch's
                own (rate, history).
            active_heads: subset of heads to compute (all if None).
            physics: also emit co-located predictions on the shared physics grid
                (``predictions_phys`` of shape ``(B, physics_K, n_dynamic)``) for
                the Phase 6 physics loss (W5b).

        Returns:
            Per-group ``pred_T``..``pred_W`` on each group's own ``K`` grid; plus
            ``predictions_phys`` when ``physics`` is set. (No single combined
            ``predictions`` tensor — group horizons differ.)
        """
        if not self.multi_rate:
            raise RuntimeError("forward_multirate requires config.multi_rate=True.")
        enc = self._encode_branches(branch_inputs)
        result: Dict[str, torch.Tensor] = {}
        for hname in self.HEAD_NAMES:
            if active_heads is not None and hname not in active_heads:
                continue
            b = self.config.encoder_groups[hname]
            T = self.trunks[hname]()                       # (K_h, n_basis)
            z = enc[b].unsqueeze(1) * T.unsqueeze(0)       # (B, K_h, n_basis)
            result[self._pred_keys[hname]] = self._heads[hname](z)

        if physics:
            ref = next(iter(enc.values()))
            B = ref.shape[0]
            Tp = self.trunk_phys()                         # (K_phys, n_basis)
            K_phys = Tp.shape[0]
            pred_phys = torch.zeros(B, K_phys, self.n_dynamic, device=ref.device)
            _persist = set(self.config.persistence_heads)
            for hname in self.HEAD_NAMES:
                # A persistence-locked head contributes zero delta on the
                # physics grid too. Leaving its learned output here would make
                # the physics residuals constrain a quantity the model does not
                # actually emit (forward_multirate zeroes it below).
                if hname in _persist:
                    continue
                b = self.config.encoder_groups[hname]
                zc = enc[b].unsqueeze(1) * Tp.unsqueeze(0)
                pred_phys[:, :, self._index_map[hname]] = self._heads[hname](zc)
            result['predictions_phys'] = pred_phys

        return zero_persistence_heads(
            result, self._pred_keys, self.config.persistence_heads)

    @classmethod
    def from_config(
        cls,
        config: FederatedConfig,
        column_info: Dict[str, Any],
    ) -> 'FederatedDeepMMNet':
        """Create model from configuration."""
        n_inputs = len(column_info['input_cols'])
        n_dynamic = len(column_info['dynamic_cols'])
        return cls(n_inputs, n_dynamic, column_info, config)


def federated(
    config: FederatedConfig,
    column_info: Dict[str, Any],
) -> FederatedDeepMMNet:
    """
    Factory function to create Phase 5 FederatedDeepMMNet model.
    
    Parameters
    ----------
    config : FederatedConfig
        Model configuration
    column_info : Dict[str, Any]
        Column information with head group indices
        
    Returns
    -------
    FederatedDeepMMNet
        Instantiated model
    """
    return FederatedDeepMMNet.from_config(config, column_info)


__all__ = [
    'BranchNetwork',
    'TBranchNetwork',
    'FourierTrunkNetwork',
    'DecoderHead',
    'SkipDecoderHead',
    'FederatedDeepMMNet',
    'federated',
]