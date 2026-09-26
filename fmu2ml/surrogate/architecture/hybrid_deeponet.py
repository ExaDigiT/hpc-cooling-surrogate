"""
Phase 3: Hybrid DeepONet with CDU embeddings, drift correction, and skip connections.

Improvements over Phase 2:
- Fix 1: CDU embedding in both pathways for per-CDU adaptation
- Fix 2: Drift correction in temporal pathway to prevent divergence
- Fix 3: Learnable skip connection weights for stability
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn

from .configs import HybridDeepONetConfig
from .deeponet import FourierFeatures
from .multirate import MultiRateOperatorModel


class CDUEmbeddingLayer(nn.Module):
    """
    CDU embedding layer for per-CDU conditioning.
    """
    
    def __init__(self, num_cdus: int, embedding_dim: int = 16):
        super().__init__()
        self.num_cdus = num_cdus
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_cdus, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
    
    def forward(self, cdu_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Get embedding for a specific CDU, expanded to batch size."""
        idx_tensor = torch.tensor([cdu_idx], device=device)
        emb = self.embedding(idx_tensor)  # (1, embedding_dim)
        return emb.expand(batch_size, -1)  # (batch, embedding_dim)


class AlgebraicPathwayWithCDU(nn.Module):
    """
    MLP for instantaneous (algebraic) relationships with CDU embedding.
    
    Each CDU gets its own Q_flow value + embedding, processed through shared layers
    then CDU-specific output heads.
    """

    def __init__(
        self,
        num_cdus: int,
        outputs_per_cdu: int = 2,
        hidden_size: int = 64,
        embedding_dim: int = 16,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_cdus = num_cdus
        self.outputs_per_cdu = outputs_per_cdu
        self.output_size = num_cdus * outputs_per_cdu
        self.embedding_dim = embedding_dim

        # CDU embedding layer
        self.cdu_embedding = CDUEmbeddingLayer(num_cdus, embedding_dim)

        # Input: 1 Q_flow + CDU embedding
        shared_input_size = 1 + embedding_dim

        # Shared feature extractor
        layers = []
        layers.extend([
            nn.Linear(shared_input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        for _ in range(n_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        self.shared_mlp = nn.Sequential(*layers)

        # CDU-specific output heads
        self.cdu_heads = nn.ModuleList([
            nn.Linear(hidden_size, outputs_per_cdu)
            for _ in range(num_cdus)
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x_qflow: torch.Tensor,
        cdu_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_qflow: (batch, prediction_steps, num_cdus) Q_flow per CDU
            cdu_indices: (num_outputs,) CDU index for each output column
        Returns:
            (batch, prediction_steps, num_outputs) predicted algebraic values
        """
        batch_size, pred_steps, n_qflow_inputs = x_qflow.shape
        device = x_qflow.device

        output = torch.zeros(batch_size, pred_steps, self.output_size, device=device)

        for cdu_idx in range(self.num_cdus):
            # CDU embedding
            cdu_emb = self.cdu_embedding(cdu_idx, batch_size * pred_steps, device)

            # Get Q_flow for this CDU
            if n_qflow_inputs == self.num_cdus:
                qflow_cdu = x_qflow[:, :, cdu_idx:cdu_idx+1]
            elif n_qflow_inputs == 1:
                qflow_cdu = x_qflow
            else:
                qflow_cdu = x_qflow[:, :, 0:1]

            qflow_flat = qflow_cdu.reshape(batch_size * pred_steps, 1)
            combined = torch.cat([qflow_flat, cdu_emb], dim=-1)

            features = self.shared_mlp(combined)
            cdu_output = self.cdu_heads[cdu_idx](features)
            cdu_output = cdu_output.view(batch_size, pred_steps, self.outputs_per_cdu)

            start_idx = cdu_idx * self.outputs_per_cdu
            end_idx = start_idx + self.outputs_per_cdu
            output[:, :, start_idx:end_idx] = cdu_output

        return output


class TemporalDeepONetWithFixes(nn.Module):
    """
    DeepONet for temporal/dynamical outputs with Phase 3 fixes:
    
    - CDU embedding for per-output conditioning
    - Skip connection (learnable alpha)
    - Drift correction toward operating mean
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        prediction_steps: int,
        num_cdus: int,
        lstm_hidden: int = 64,
        trunk_hidden: int = 64,
        n_basis: int = 32,
        embedding_dim: int = 16,
        n_fourier_freqs: int = 8,
        dropout: float = 0.3,
        drift_correction_strength: float = 0.1,
    ):
        super().__init__()
        self.output_size = output_size
        self.prediction_steps = prediction_steps
        self.n_basis = n_basis
        self.num_cdus = num_cdus
        self.drift_correction_strength = drift_correction_strength

        # CDU embedding for temporal outputs
        self.cdu_embedding = CDUEmbeddingLayer(num_cdus, embedding_dim)

        # Branch: LSTM encoder
        self.branch_encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.branch_norm = nn.LayerNorm(lstm_hidden)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.Tanh(),
            nn.Linear(lstm_hidden // 2, 1),
        )

        # Branch head with CDU conditioning
        self.branch_head = nn.Sequential(
            nn.Linear(lstm_hidden + embedding_dim, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, n_basis),
        )

        # Trunk: Fourier features
        self.fourier = FourierFeatures(n_freqs=n_fourier_freqs)
        self.register_buffer(
            'query_times',
            torch.linspace(0, 1, prediction_steps).view(-1, 1),
        )

        self.trunk_net = nn.Sequential(
            nn.Linear(self.fourier.output_dim, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, n_basis),
        )

        # Learnable skip connection weights (alpha)
        # Shape: (prediction_steps, output_size) — per step and output
        self.skip_alpha = nn.Parameter(torch.ones(prediction_steps, output_size) * 0.5)

        # Learnable drift correction strength per output
        self.drift_beta = nn.Parameter(torch.ones(output_size) * drift_correction_strength)

        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(output_size) * 0.1)
        self.output_bias = nn.Parameter(torch.zeros(output_size))

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param, gain=0.3)
            elif 'bias' in name and 'output_bias' not in name:
                nn.init.zeros_(param)

    def _get_trunk_features(self) -> torch.Tensor:
        """Get Fourier features for query times."""
        return self.fourier(self.query_times)

    def forward(
        self,
        x: torch.Tensor,
        last_output_normalized: torch.Tensor,
        output_mean: torch.Tensor,
        cdu_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, history_steps, input_size)
            last_output_normalized: (batch, output_size) last known output (normalized)
            output_mean: (batch, output_size) mean of each output (for drift correction)
            cdu_indices: (output_size,) CDU index for each output
        Returns:
            y_delta: (batch, prediction_steps, output_size) predicted deltas
            alpha: (prediction_steps, output_size) skip connection weights
        """
        batch_size = x.shape[0]
        device = x.device

        # Encode history
        lstm_out, _ = self.branch_encoder(x)
        lstm_out = self.branch_norm(lstm_out)

        # Attention over time
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, lstm_hidden)

        # Trunk features
        trunk_features = self._get_trunk_features()
        trunk_out = self.trunk_net(trunk_features)  # (pred_steps, n_basis)

        # Process each output with its CDU embedding
        raw_predictions = torch.zeros(
            batch_size, self.prediction_steps, self.output_size, device=device
        )

        # DataLoader collates (n_temporal,) → (batch_size, n_temporal); flatten back
        if cdu_indices.dim() == 2:
            cdu_indices = cdu_indices[0]

        for out_idx in range(self.output_size):
            cdu_idx = cdu_indices[out_idx].item()
            cdu_emb = self.cdu_embedding(cdu_idx, batch_size, device)

            combined = torch.cat([context, cdu_emb], dim=-1)
            branch_out = self.branch_head(combined)  # (batch, n_basis)
            out = torch.matmul(branch_out, trunk_out.T)  # (batch, pred_steps)
            out = out * self.output_scale[out_idx] + self.output_bias[out_idx]
            raw_predictions[:, :, out_idx] = out

        # ── Drift Correction ──────────────────────────────────────────────
        deviation_from_mean = last_output_normalized - output_mean
        step_weights = torch.arange(1, self.prediction_steps + 1, device=device).float()
        step_weights = step_weights.view(-1, 1)

        drift_correction = (
            -self.drift_beta.view(1, 1, -1) *
            deviation_from_mean.unsqueeze(1) *
            step_weights.unsqueeze(0)
        )
        corrected_predictions = raw_predictions + drift_correction * 0.01

        # ── Skip Connection ───────────────────────────────────────────────
        alpha = torch.sigmoid(self.skip_alpha)
        final_deltas = alpha.unsqueeze(0) * corrected_predictions

        return final_deltas, alpha


class HybridDeepONet(nn.Module):
    """
    Phase 3: Hybrid DeepONet with all fixes.
    
    Improvements:
    - Fix 1: CDU embedding in both pathways
    - Fix 2: Drift correction in temporal pathway
    - Fix 3: Skip connection in temporal pathway
    
    Parameters
    ----------
    temporal_input_size : int
        Input size for temporal pathway
    temporal_output_size : int
        Number of temporal outputs
    algebraic_output_size : int
        Number of algebraic outputs
    prediction_steps : int
        Number of future steps to predict
    num_cdus : int
        Number of CDUs in the system
    lstm_hidden : int
        Hidden size for LSTM
    trunk_hidden : int
        Hidden size for trunk network
    n_basis : int
        Number of basis functions
    embedding_dim : int
        Dimension of CDU embeddings
    algebraic_hidden : int
        Hidden size for algebraic MLP
    algebraic_layers : int
        Number of layers in algebraic MLP
    dropout : float
        Dropout rate
    drift_correction_strength : float
        Initial drift correction strength
    """

    def __init__(
        self,
        temporal_input_size: int,
        temporal_output_size: int,
        algebraic_output_size: int,
        prediction_steps: int,
        num_cdus: int,
        lstm_hidden: int = 64,
        trunk_hidden: int = 64,
        n_basis: int = 32,
        embedding_dim: int = 16,
        algebraic_hidden: int = 64,
        algebraic_layers: int = 3,
        dropout: float = 0.3,
        drift_correction_strength: float = 0.1,
    ):
        super().__init__()
        self.temporal_output_size = temporal_output_size
        self.algebraic_output_size = algebraic_output_size
        self.prediction_steps = prediction_steps
        self.num_cdus = num_cdus

        self.temporal_pathway = TemporalDeepONetWithFixes(
            input_size=temporal_input_size,
            output_size=temporal_output_size,
            prediction_steps=prediction_steps,
            num_cdus=num_cdus,
            lstm_hidden=lstm_hidden,
            trunk_hidden=trunk_hidden,
            n_basis=n_basis,
            embedding_dim=embedding_dim,
            dropout=dropout,
            drift_correction_strength=drift_correction_strength,
        )

        outputs_per_cdu = algebraic_output_size // num_cdus if num_cdus > 0 else 1
        self.algebraic_pathway = AlgebraicPathwayWithCDU(
            num_cdus=num_cdus,
            outputs_per_cdu=outputs_per_cdu,
            hidden_size=algebraic_hidden,
            embedding_dim=embedding_dim,
            n_layers=algebraic_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x_temporal: torch.Tensor,
        x_algebraic: torch.Tensor,
        last_temporal_normalized: torch.Tensor,
        temporal_mean: torch.Tensor,
        temporal_cdu_indices: torch.Tensor,
        algebraic_cdu_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x_temporal: (batch, history_steps, temporal_input_size)
            x_algebraic: (batch, prediction_steps, num_cdus) Q_flow per CDU
            last_temporal_normalized: (batch, temporal_output_size)
            temporal_mean: (batch, temporal_output_size)
            temporal_cdu_indices: (temporal_output_size,) CDU index per temporal output
            algebraic_cdu_indices: (algebraic_output_size,) CDU index per algebraic output
            
        Returns:
            y_temporal: (batch, prediction_steps, temporal_output_size)
            y_algebraic: (batch, prediction_steps, algebraic_output_size)
            alpha: (prediction_steps, temporal_output_size) skip connection weights
        """
        y_temporal, alpha = self.temporal_pathway(
            x_temporal, last_temporal_normalized, temporal_mean, temporal_cdu_indices
        )
        y_algebraic = self.algebraic_pathway(x_algebraic, algebraic_cdu_indices)
        return y_temporal, y_algebraic, alpha

    @classmethod
    def from_config(
        cls,
        config: HybridDeepONetConfig,
        temporal_input_size: int,
        temporal_output_size: int,
        algebraic_output_size: int,
    ) -> 'HybridDeepONet':
        """Create model from configuration."""
        return cls(
            temporal_input_size=temporal_input_size,
            temporal_output_size=temporal_output_size,
            algebraic_output_size=algebraic_output_size,
            prediction_steps=config.prediction_steps,
            num_cdus=config.num_cdus,
            lstm_hidden=config.branch_lstm_hidden,
            trunk_hidden=config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 64,
            n_basis=config.basis_dim,
            embedding_dim=config.cdu_embedding_dim,
            algebraic_hidden=config.algebraic_decoder_hidden[0] if config.algebraic_decoder_hidden else 64,
            algebraic_layers=len(config.algebraic_decoder_hidden) + 1 if config.algebraic_decoder_hidden else 3,
            dropout=config.branch_lstm_dropout,
            drift_correction_strength=0.1,  # Default from notebook
        )


class MultiRateHybridDeepONet(MultiRateOperatorModel):
    """Phase 3 on the multi-rate task: the phase 2 operator core plus phase
    3's identity — a learnable per-output blend gate on every head
    (``head_mode='alpha'``, sigmoid init 0.5). The CDU-embedding / algebraic
    pathway of the legacy per-CDU dataset has no analogue on the all-CDU
    chunk-store task and is not part of this mode."""

    def __init__(self, config: HybridDeepONetConfig, column_info: Dict[str, Any]):
        super().__init__(
            config, column_info, head_mode='alpha',
            encoder_hidden=config.branch_lstm_hidden,
            encoder_layers=config.branch_lstm_layers,
            n_basis=config.basis_dim,
            trunk_hidden=config.trunk_hidden_sizes[0] if config.trunk_hidden_sizes else 64,
            n_fourier=getattr(config, 'trunk_n_fourier', 8),
            dropout=config.branch_lstm_dropout,
        )


def hybrid_deeponet(
    config: HybridDeepONetConfig,
    column_info: Optional[Dict[str, Any]] = None,
    temporal_input_size: Optional[int] = None,
    temporal_output_size: Optional[int] = None,
    algebraic_output_size: Optional[int] = None,
) -> HybridDeepONet:
    """
    Factory function to create Phase 3 HybridDeepONet model.
    
    Parameters
    ----------
    config : HybridDeepONetConfig
        Model configuration
    column_info : Dict[str, Any], optional
        Column information for inferring sizes
    temporal_input_size : int, optional
        Input size for temporal pathway
    temporal_output_size : int, optional
        Number of temporal outputs
    algebraic_output_size : int, optional
        Number of algebraic outputs
        
    Returns
    -------
    HybridDeepONet
        Instantiated model (``MultiRateHybridDeepONet`` when ``config.multi_rate``)
    """
    # Multi-rate mode: group-structured model on the chunk-store task
    if getattr(config, 'multi_rate', False):
        if column_info is None:
            raise ValueError("multi-rate hybrid_deeponet requires column_info")
        return MultiRateHybridDeepONet(config, column_info)

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
        
        if algebraic_output_size is None:
            algebraic_output_size = n_algebraic
    
    if any(x is None for x in [temporal_input_size, temporal_output_size, algebraic_output_size]):
        raise ValueError("All sizes must be provided or inferable from column_info")
    
    return HybridDeepONet.from_config(
        config,
        temporal_input_size=temporal_input_size,
        temporal_output_size=temporal_output_size,
        algebraic_output_size=algebraic_output_size,
    )


__all__ = [
    'CDUEmbeddingLayer',
    'AlgebraicPathwayWithCDU',
    'TemporalDeepONetWithFixes',
    'HybridDeepONet',
    'MultiRateHybridDeepONet',
    'hybrid_deeponet',
]