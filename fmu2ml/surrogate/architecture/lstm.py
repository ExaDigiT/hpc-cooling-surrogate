"""
Phase 1: Baseline LSTM with temporal attention.

Architecture:
1. Input projection: maps concatenated features to hidden dimension
2. LSTM encoder: 2-layer LSTM captures temporal dependencies
3. Temporal attention: learned weighted average over time steps
4. Decoder MLP: projects to delta predictions for each future step
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn

from .configs import LSTMConfig, zero_persistence_heads


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for weighted aggregation over time steps.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    
    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (batch, seq_len, hidden_size)
        Returns:
            context: (batch, hidden_size) weighted sum over time
        """
        attn_weights = self.attention(lstm_out)  # (B, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (B, hidden_size)
        return context


class BaselineLSTM(nn.Module):
    """
    Phase 1 Baseline LSTM with temporal attention.

    Architecture:
    1. Input projection: maps concatenated features to hidden dimension
    2. LSTM encoder: 2-layer LSTM captures temporal dependencies
    3. Temporal attention: learned weighted average over time steps
    4. Decoder MLP: projects to delta predictions for each future step
    
    Parameters
    ----------
    input_size : int
        Size of input features (n_inputs + n_outputs for concatenated history)
    output_size : int
        Number of output features to predict
    hidden_size : int
        Hidden dimension for LSTM and MLP layers
    num_layers : int
        Number of LSTM layers
    dropout : float
        Dropout rate
    prediction_steps : int
        Number of future time steps to predict
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        prediction_steps: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.prediction_steps = prediction_steps

        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)

        # Temporal attention
        self.attention = TemporalAttention(hidden_size)

        # Decoder MLP
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size * prediction_steps),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_size) concatenated input + output history
            
        Returns:
            (batch, prediction_steps, output_size) predicted deltas
        """
        batch_size = x.shape[0]

        # Project input
        x = self.input_proj(x)  # (B, H, hidden)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (B, H, hidden)
        lstm_out = self.lstm_norm(lstm_out)

        # Attention over time steps
        context = self.attention(lstm_out)  # (B, hidden)

        # Decode to predictions
        out = self.decoder(context)  # (B, output_size * pred_steps)
        out = out.view(batch_size, self.prediction_steps, self.output_size)

        return out

    @classmethod
    def from_config(
        cls,
        config: LSTMConfig,
        input_size: int,
        output_size: int,
    ) -> 'BaselineLSTM':
        """
        Create model from configuration.
        
        Parameters
        ----------
        config : LSTMConfig
            Model configuration
        input_size : int
            Input feature size
        output_size : int
            Output feature size
            
        Returns
        -------
        BaselineLSTM
            Instantiated model
        """
        return cls(
            input_size=input_size,
            output_size=output_size,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            prediction_steps=config.prediction_steps,
        )


class _BranchLSTMEncoder(nn.Module):
    """The BaselineLSTM encoder stack (proj -> LSTM -> norm -> attention)
    packaged per branch for multi-rate mode."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.lstm_norm = nn.LayerNorm(hidden_size)
        self.attention = TemporalAttention(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        out, _ = self.lstm(x)
        return self.attention(self.lstm_norm(out))       # (B, hidden)


class MultiRateLSTM(nn.Module):
    """Phase 1 on the multi-rate task, keeping its identity: LSTM+attention
    encoding and DIRECT MLP decoding (no basis expansion, no trunk).

    One encoder per branch id over concat(u_hist, y_hist) at that branch's
    (rate, history); branch contexts are concatenated and each head decodes
    the fused context straight to its own ``(K_h, n_h)`` delta grid.
    """

    def __init__(self, config: LSTMConfig, column_info: Dict[str, Any]):
        super().__init__()
        from .multirate import head_layout

        self.config = config
        self.multi_rate = True
        self.layout = head_layout(config, column_info)

        n_in = len(column_info['input_cols'])
        n_dyn = len(column_info['dynamic_cols'])
        hidden = config.lstm_hidden_size
        branch_ids = sorted(set(config.encoder_groups.values()))
        self.branch_ids = branch_ids

        self.branches = nn.ModuleDict({
            b: _BranchLSTMEncoder(n_in + n_dyn, hidden,
                                  config.lstm_num_layers, config.lstm_dropout)
            for b in branch_ids
        })

        fused = hidden * len(branch_ids)
        self.heads = nn.ModuleDict({
            h: nn.Sequential(
                nn.Linear(fused, hidden * 2),
                nn.LayerNorm(hidden * 2),
                nn.ReLU(),
                nn.Dropout(config.lstm_dropout),
                nn.Linear(hidden * 2, hidden),
                nn.ReLU(),
                nn.Dropout(config.lstm_dropout),
                nn.Linear(hidden, spec['K'] * spec['n_outputs']),
            )
            for h, spec in self.layout.items()
        })
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward_multirate(
        self, branch_inputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        ctx = torch.cat(
            [self.branches[b](torch.cat([u, y], dim=-1))
             for b, (u, y) in sorted(branch_inputs.items())],
            dim=-1,
        )
        out: Dict[str, torch.Tensor] = {}
        for h, spec in self.layout.items():
            pred = self.heads[h](ctx)
            out[spec['pred_key']] = pred.view(-1, spec['K'], spec['n_outputs'])
        return zero_persistence_heads(
            out, {h: s['pred_key'] for h, s in self.layout.items()},
            self.config.persistence_heads)

    def forward(self, *args, **kwargs):
        # See MultiRateOperator.forward: DDP drives the model through
        # __call__ -> forward(), so the multi-rate entry point must be
        # reachable that way or gradients are never all-reduced.
        if args and isinstance(args[0], dict):
            return self.forward_multirate(*args, **kwargs)
        raise RuntimeError(
            "MultiRateLSTM is a multi-rate model; use "
            "forward_multirate({branch_id: (u_hist_b, y_hist_b)}).")


def lstm(
    config: LSTMConfig,
    input_size: Optional[int] = None,
    output_size: Optional[int] = None,
    column_info: Optional[Dict[str, Any]] = None,
) -> BaselineLSTM:
    """
    Factory function to create Phase 1 BaselineLSTM model.
    
    Parameters
    ----------
    config : LSTMConfig
        Model configuration
    input_size : int, optional
        Input feature size (inferred from column_info if not provided)
    output_size : int, optional
        Output feature size (inferred from column_info if not provided)
    column_info : Dict[str, Any], optional
        Column information dictionary
        
    Returns
    -------
    BaselineLSTM
        Instantiated model (``MultiRateLSTM`` when ``config.multi_rate``)
    """
    # Multi-rate mode: group-structured model on the chunk-store task
    if getattr(config, 'multi_rate', False):
        if column_info is None:
            raise ValueError("multi-rate lstm requires column_info")
        return MultiRateLSTM(config, column_info)

    # Infer sizes from column_info if not provided
    if column_info is not None:
        if input_size is None:
            n_inputs = column_info.get('n_inputs', len(column_info.get('input_cols', [])))
            n_outputs = column_info.get('n_outputs', len(column_info.get('output_cols', [])))
            input_size = n_inputs + n_outputs  # Concatenated history
        if output_size is None:
            output_size = column_info.get('n_outputs', len(column_info.get('output_cols', [])))
    
    if input_size is None or output_size is None:
        raise ValueError("input_size and output_size must be provided or inferable from column_info")
    
    return BaselineLSTM.from_config(config, input_size, output_size)


__all__ = [
    'BaselineLSTM',
    'MultiRateLSTM',
    'TemporalAttention',
    'lstm',
]