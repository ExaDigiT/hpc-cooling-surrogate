"""
Custom architecture builder — fluent API for assembling models from
shared components (LSTM encoder, Fourier trunk, decoder heads, etc.).

Example
-------
>>> import fmu2ml.surrogate as surrogate
>>> model = (
...     surrogate.architecture.build()
...     .with_encoder("lstm", hidden_size=128, num_layers=2)
...     .with_trunk("fourier", n_fourier=8, hidden=64)
...     .with_decoder_heads(["temperature", "flow"])
...     .with_fusion("hadamard")
...     .build(input_size=512, output_sizes={"temperature": 1028, "flow": 514})
... )
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from .lstm import TemporalAttention


# ───────────────────────────────────────────────────────────────────────
# Reusable components
# ───────────────────────────────────────────────────────────────────────

class _LSTMEncoder(nn.Module):
    """LSTM encoder with optional temporal attention."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_attention: bool = True,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.use_attention = use_attention
        if use_attention:
            self.attention = TemporalAttention(hidden_size)
        self.output_dim = hidden_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h, _ = self.lstm(h)
        h = self.norm(h)
        if self.use_attention:
            return self.attention(h)  # (B, hidden)
        return h[:, -1, :]            # last step


class _FourierTrunk(nn.Module):
    """Fourier-feature trunk network."""

    def __init__(
        self,
        n_fourier: int = 8,
        hidden: int = 64,
        n_layers: int = 2,
        output_dim: int = 64,
    ):
        super().__init__()
        self.n_fourier = n_fourier
        freqs = torch.linspace(1, n_fourier, n_fourier) * math.pi
        self.register_buffer("freqs", freqs)
        trunk_in = 1 + 2 * n_fourier
        layers: list = []
        for i in range(n_layers):
            in_d = trunk_in if i == 0 else hidden
            layers += [nn.Linear(in_d, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (K, 1) → (K, output_dim)."""
        sin_f = torch.sin(t * self.freqs)
        cos_f = torch.cos(t * self.freqs)
        ff = torch.cat([t, sin_f, cos_f], dim=-1)
        return self.net(ff)


class _DecoderHead(nn.Module):
    """Simple MLP decoder head."""

    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ───────────────────────────────────────────────────────────────────────
# Custom assembled model
# ───────────────────────────────────────────────────────────────────────

class _CustomSurrogate(nn.Module):
    """Model assembled by ``CustomArchitectureBuilder``."""

    _surrogate_phase = 0  # custom

    def __init__(
        self,
        encoder: nn.Module,
        trunk: Optional[nn.Module],
        heads: nn.ModuleDict,
        fusion: str,
        prediction_steps: int,
    ):
        super().__init__()
        self.encoder = encoder
        self.trunk = trunk
        self.heads = heads
        self.fusion = fusion
        self.prediction_steps = prediction_steps

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        ctx = self.encoder(x)  # (B, D_enc)
        outputs: Dict[str, torch.Tensor] = {}

        if self.trunk is not None:
            K = self.prediction_steps
            t = torch.linspace(0, 1, K, device=x.device).unsqueeze(-1)
            trunk_out = self.trunk(t)  # (K, D_trunk)

            for name, head in self.heads.items():
                if self.fusion == "hadamard":
                    z = ctx.unsqueeze(1) * trunk_out.unsqueeze(0)  # (B,K,D)
                elif self.fusion == "additive":
                    z = ctx.unsqueeze(1) + trunk_out.unsqueeze(0)
                else:  # concat
                    ctx_exp = ctx.unsqueeze(1).expand(-1, K, -1)
                    trunk_exp = trunk_out.unsqueeze(0).expand(
                        ctx.shape[0], -1, -1
                    )
                    z = torch.cat([ctx_exp, trunk_exp], dim=-1)
                B, K2, D = z.shape
                out = head(z.reshape(B * K2, D)).reshape(B, K2, -1)
                outputs[name] = out
        else:
            for name, head in self.heads.items():
                outputs[name] = head(ctx)

        return outputs


# ───────────────────────────────────────────────────────────────────────
# Builder
# ───────────────────────────────────────────────────────────────────────

class CustomArchitectureBuilder:
    """
    Fluent builder for custom surrogate architectures.

    Example
    -------
    >>> model = (
    ...     CustomArchitectureBuilder()
    ...     .with_encoder("lstm", hidden_size=128)
    ...     .with_trunk("fourier", n_fourier=8, hidden=64)
    ...     .with_decoder_heads(["temperature", "flow"],
    ...                         output_sizes={"temperature": 1028, "flow": 514})
    ...     .with_fusion("hadamard")
    ...     .build(input_size=3100, prediction_steps=1)
    ... )
    """

    def __init__(self):
        self._encoder_type: str = "lstm"
        self._encoder_kwargs: Dict[str, Any] = {}
        self._trunk_type: Optional[str] = None
        self._trunk_kwargs: Dict[str, Any] = {}
        self._head_names: List[str] = []
        self._head_output_sizes: Dict[str, int] = {}
        self._fusion: str = "hadamard"
        self._decoder_hidden: int = 128

    # ── Encoder ──────────────────────────────────────────────────────

    def with_encoder(self, encoder_type: str = "lstm", **kwargs) -> "CustomArchitectureBuilder":
        self._encoder_type = encoder_type
        self._encoder_kwargs = kwargs
        return self

    # ── Trunk ────────────────────────────────────────────────────────

    def with_trunk(self, trunk_type: str = "fourier", **kwargs) -> "CustomArchitectureBuilder":
        self._trunk_type = trunk_type
        self._trunk_kwargs = kwargs
        return self

    # ── Decoder heads ────────────────────────────────────────────────

    def with_decoder_heads(
        self,
        names: List[str],
        output_sizes: Optional[Dict[str, int]] = None,
        hidden: int = 128,
    ) -> "CustomArchitectureBuilder":
        self._head_names = names
        if output_sizes:
            self._head_output_sizes = output_sizes
        self._decoder_hidden = hidden
        return self

    # ── Fusion ───────────────────────────────────────────────────────

    def with_fusion(self, fusion: str = "hadamard") -> "CustomArchitectureBuilder":
        assert fusion in ("hadamard", "additive", "concat")
        self._fusion = fusion
        return self

    # ── Build ────────────────────────────────────────────────────────

    def build(
        self,
        input_size: int,
        prediction_steps: int = 1,
        output_sizes: Optional[Dict[str, int]] = None,
    ) -> _CustomSurrogate:
        if output_sizes:
            self._head_output_sizes.update(output_sizes)

        # Encoder
        enc_hidden = self._encoder_kwargs.get("hidden_size", 128)
        enc_layers = self._encoder_kwargs.get("num_layers", 2)
        enc_drop = self._encoder_kwargs.get("dropout", 0.2)

        if self._encoder_type == "lstm":
            encoder = _LSTMEncoder(
                input_size, enc_hidden, enc_layers, enc_drop, True
            )
        else:
            raise ValueError(f"Unknown encoder type: {self._encoder_type}")

        # Trunk
        trunk = None
        trunk_dim = enc_hidden  # fallback
        if self._trunk_type == "fourier":
            nf = self._trunk_kwargs.get("n_fourier", 8)
            th = self._trunk_kwargs.get("hidden", 64)
            tl = self._trunk_kwargs.get("n_layers", 2)
            trunk = _FourierTrunk(nf, th, tl, output_dim=enc_hidden)
            trunk_dim = enc_hidden

        # Head input dimension
        if self._fusion == "concat" and trunk is not None:
            head_input = enc_hidden + trunk_dim
        else:
            head_input = enc_hidden

        # Heads
        heads = nn.ModuleDict()
        for name in self._head_names:
            out_d = self._head_output_sizes.get(name, 1)
            heads[name] = _DecoderHead(head_input, out_d, self._decoder_hidden)

        return _CustomSurrogate(
            encoder=encoder,
            trunk=trunk,
            heads=heads,
            fusion=self._fusion,
            prediction_steps=prediction_steps,
        )


def build() -> CustomArchitectureBuilder:
    """Return a fresh ``CustomArchitectureBuilder``."""
    return CustomArchitectureBuilder()