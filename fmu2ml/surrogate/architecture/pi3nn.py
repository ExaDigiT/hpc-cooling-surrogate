"""
PI3NN: distribution-free, OOD-aware prediction intervals for the federated model.

Reference: Liu et al., "PI3NN: Prediction intervals from three independently
trained neural networks" (ICLR 2022, arXiv:2108.02327).

The idea, adapted to the Phase-5 ``FederatedDeepMMNet``:

* The trained federated model is the **mean** network ``f(x)`` — it is frozen.
* Two auxiliary heads consume the shared encoder latent ``z = f.encode(u, y)``
  (detached) and emit **non-negative** upward / downward deviation magnitudes
  ``u(x) >= 0`` and ``l(x) >= 0``.
* The aux heads are trained on the **positive / negative residuals** of the
  frozen mean model (see ``evaluate.pi3nn.train_pi3nn_bounds``).
* At calibration time, per-output scalars ``c_u, c_l`` are found by root-finding
  so the interval ``[f - c_l·l, f + c_u·u]`` attains the target coverage
  (see ``evaluate.pi3nn.calibrate_pi3nn``).

Why it helps with OOD: the deviation heads extrapolate away from the training
distribution, so intervals widen automatically on out-of-distribution inputs
(e.g. a Summit-trained model run zero-shot on Marconi100/Lassen) — a signal MC
dropout typically misses (it tends to be over-confident off-distribution).

Reusing the encoder makes this an "add two heads" change rather than a new model;
other phases can be wrapped the same way by exposing an ``encode`` method.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .federated import FederatedDeepMMNet
from .configs import FederatedConfig


class PI3NNBoundHead(nn.Module):
    """
    Non-negative deviation head: latent ``z`` → ``>= 0`` deviation per output.

    Mirrors the structure of ``federated.DecoderHead`` but applies a softplus so
    the output (an interval half-width contribution) is strictly non-negative.
    """

    def __init__(
        self,
        n_basis: int,
        hidden_dim: int,
        n_outputs: int,
        dropout: float = 0.0,
    ):
        super().__init__()
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
        # Small positive init bias so deviations start non-degenerate.
        self.bias = nn.Parameter(torch.zeros(n_outputs))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, K, n_basis) → (B, K, n_outputs) non-negative deviations."""
        return F.softplus(self.net(z) + self.bias)


class FederatedPI3NN(nn.Module):
    """
    PI3NN wrapper around a (trained, frozen) ``FederatedDeepMMNet``.

    Produces, per forward pass, the mean prediction (from the frozen model, in
    normalized-delta space) plus non-negative upward/downward deviations from the
    two auxiliary heads. Calibration scalars are applied downstream, not here.

    Parameters
    ----------
    mean_model : FederatedDeepMMNet
        Trained Phase-5 model used as the (frozen) mean network.
    n_basis : int
        Encoder basis dimension (``config.basis_dim``).
    n_dynamic : int
        Number of dynamic outputs (matches the mean model).
    hidden_dim : int
        Hidden width of each deviation head.
    dropout : float
        Dropout inside the deviation heads (0 by default — PI3NN uncertainty
        comes from the residual fit, not from stochastic passes).
    """

    def __init__(
        self,
        mean_model: FederatedDeepMMNet,
        n_basis: int,
        n_dynamic: int,
        hidden_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mean_model = mean_model
        self.n_dynamic = n_dynamic
        self.upper = PI3NNBoundHead(n_basis, hidden_dim, n_dynamic, dropout)
        self.lower = PI3NNBoundHead(n_basis, hidden_dim, n_dynamic, dropout)
        self.freeze_mean()

    def freeze_mean(self) -> None:
        """Freeze the mean model so only the deviation heads train."""
        self.mean_model.eval()
        for p in self.mean_model.parameters():
            p.requires_grad_(False)

    def bound_parameters(self):
        """Trainable parameters: the two deviation heads only."""
        return list(self.upper.parameters()) + list(self.lower.parameters())

    def forward(
        self,
        u_hist: torch.Tensor,
        y_hist: torch.Tensor,
        K: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with
            'mean'      : (B, K, n_dynamic) frozen mean prediction (norm. delta)
            'upper_dev' : (B, K, n_dynamic) non-negative upward deviation
            'lower_dev' : (B, K, n_dynamic) non-negative downward deviation
        """
        with torch.no_grad():
            out = self.mean_model(u_hist, y_hist, K)
            mean = out['predictions']
            z = self.mean_model.encode(u_hist, y_hist, K)
        z = z.detach()
        return {
            'mean': mean,
            'upper_dev': self.upper(z),
            'lower_dev': self.lower(z),
        }


def federated_pi3nn(
    mean_model: FederatedDeepMMNet,
    config: FederatedConfig,
    column_info: Dict[str, Any],
    hidden_dim: int = 128,
    dropout: float = 0.0,
) -> FederatedPI3NN:
    """Factory: wrap a trained federated model with PI3NN deviation heads."""
    n_basis = config.basis_dim
    n_dynamic = len(column_info['dynamic_cols'])
    return FederatedPI3NN(mean_model, n_basis, n_dynamic,
                          hidden_dim=hidden_dim, dropout=dropout)


__all__ = [
    'PI3NNBoundHead',
    'FederatedPI3NN',
    'federated_pi3nn',
]
