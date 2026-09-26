"""
fmu2ml.surrogate.train.loss - Phase-aware loss functions for surrogate models.

Provides loss functions for all six phases:
- Phase 1 (LSTM): MSE/Huber loss
- Phase 2 (Basic DeepONet): HybridLoss (temporal + algebraic)
- Phase 3 (Hybrid DeepONet): HybridLoss with alpha regularization
- Phase 4 (Domain DeepONet): DomainLoss with optional derivative matching
- Phase 5 (Federated): HeadLoss for 6 decoder heads
- Phase 6 (Physics-Informed): HeadLoss + PhysicsConstraintLoss
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
import torch.nn as nn

__all__ = [
    "BaseLoss",
    "MSELoss",
    "HuberLoss",
    "HybridLoss",
    "DomainLoss",
    "HeadLoss",
    "PhysicsConstraintLoss",
    "SurrogateLoss",
    "create_loss",
]


# ───────────────────────────────────────────────────────────────────────
# Base Loss Interface
# ───────────────────────────────────────────────────────────────────────


class BaseLoss(nn.Module, ABC):
    """Abstract base class for all surrogate loss functions."""

    @abstractmethod
    def forward(self, *args, **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute the loss.

        Returns:
            Either a single loss tensor, or a tuple of (loss_tensor, loss_dict)
            where loss_dict contains component losses for logging.
        """
        pass


# ───────────────────────────────────────────────────────────────────────
# Basic Loss Functions (Phase 1)
# ───────────────────────────────────────────────────────────────────────


class MSELoss(BaseLoss):
    """Mean Squared Error loss wrapper."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute MSE loss.

        Args:
            pred: Predictions (batch, pred_steps, n_outputs)
            target: Targets (batch, pred_steps, n_outputs)

        Returns:
            Tuple of (loss_tensor, loss_dict)
        """
        loss = self.loss_fn(pred, target)
        return loss, {"total": loss.item(), "mse": loss.item()}


class HuberLoss(BaseLoss):
    """Huber loss wrapper with configurable delta."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.loss_fn = nn.HuberLoss(delta=delta)
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute Huber loss.

        Args:
            pred: Predictions (batch, pred_steps, n_outputs)
            target: Targets (batch, pred_steps, n_outputs)

        Returns:
            Tuple of (loss_tensor, loss_dict)
        """
        loss = self.loss_fn(pred, target)
        return loss, {"total": loss.item(), "huber": loss.item()}


# ───────────────────────────────────────────────────────────────────────
# Hybrid Loss (Phase 2 & 3)
# ───────────────────────────────────────────────────────────────────────


class HybridLoss(BaseLoss):
    """
    Combined loss for hybrid architectures with temporal and algebraic pathways.

    Used in Phase 2 (Basic DeepONet) and Phase 3 (Hybrid DeepONet).

    Components:
    - Temporal: Huber loss + variance matching
    - Algebraic: MSE loss
    - Alpha regularization (Phase 3): penalizes extreme skip connection weights
    """

    def __init__(
        self,
        temporal_weight: float = 1.0,
        algebraic_weight: float = 1.0,
        huber_delta: float = 0.5,
        var_weight: float = 0.1,
        alpha_reg_weight: float = 0.0,
    ):
        """
        Initialize HybridLoss.

        Args:
            temporal_weight: Weight for temporal pathway loss
            algebraic_weight: Weight for algebraic pathway loss
            huber_delta: Delta parameter for Huber loss
            var_weight: Weight for variance matching loss
            alpha_reg_weight: Weight for alpha regularization (0 = disabled)
        """
        super().__init__()
        self.temporal_weight = temporal_weight
        self.algebraic_weight = algebraic_weight
        self.huber = nn.HuberLoss(delta=huber_delta)
        self.mse = nn.MSELoss()
        self.var_weight = var_weight
        self.alpha_reg_weight = alpha_reg_weight

    def forward(
        self,
        pred_temporal: torch.Tensor,
        target_temporal: torch.Tensor,
        pred_algebraic: torch.Tensor,
        target_algebraic: torch.Tensor,
        alpha: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss.

        Args:
            pred_temporal: Temporal predictions (batch, pred_steps, n_temporal)
            target_temporal: Temporal targets
            pred_algebraic: Algebraic predictions (batch, pred_steps, n_algebraic)
            target_algebraic: Algebraic targets
            alpha: Optional skip connection weights for regularization

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Algebraic loss (MSE)
        algebraic_loss = self.mse(pred_algebraic, target_algebraic)

        # Temporal loss (Huber + variance matching)
        temporal_huber = self.huber(pred_temporal, target_temporal)

        # Variance matching: encourage predictions to have similar variance as targets
        pred_var = torch.var(pred_temporal, dim=(0, 1)) + 1e-8
        target_var = torch.var(target_temporal, dim=(0, 1)) + 1e-8
        var_ratio = torch.log(pred_var / target_var)
        var_loss = torch.mean(var_ratio ** 2)

        temporal_loss = temporal_huber + self.var_weight * var_loss

        # Alpha regularization: penalize extreme values (encourage near 0.5)
        alpha_reg = torch.tensor(0.0, device=pred_temporal.device)
        if alpha is not None and self.alpha_reg_weight > 0:
            alpha_reg = torch.mean((alpha - 0.5) ** 2)

        # Combined loss
        total_loss = (
            self.temporal_weight * temporal_loss
            + self.algebraic_weight * algebraic_loss
            + self.alpha_reg_weight * alpha_reg
        )

        loss_dict = {
            "total": total_loss.item(),
            "temporal": temporal_loss.item(),
            "algebraic": algebraic_loss.item(),
            "temporal_huber": temporal_huber.item(),
            "var_match": var_loss.item(),
            "alpha_reg": alpha_reg.item() if isinstance(alpha_reg, torch.Tensor) else alpha_reg,
        }

        return total_loss, loss_dict


# ───────────────────────────────────────────────────────────────────────
# Domain Loss (Phase 4)
# ───────────────────────────────────────────────────────────────────────


class DomainLoss(BaseLoss):
    """
    Loss function for domain-specific DeepONet models.

    Supports configurable data loss (Huber/MSE/MAE) with optional derivative matching.
    Used in Phase 4 (Domain DeepONet).
    """

    def __init__(
        self,
        loss_type: str = "huber",
        huber_delta: float = 0.5,
        use_derivative_loss: bool = False,
        derivative_weight: float = 0.1,
    ):
        """
        Initialize DomainLoss.

        Args:
            loss_type: Type of data loss ('huber', 'mse', 'mae')
            huber_delta: Delta parameter for Huber loss
            use_derivative_loss: Whether to include rate-of-change matching
            derivative_weight: Weight for derivative loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.use_derivative_loss = use_derivative_loss
        self.derivative_weight = derivative_weight

        if loss_type == "huber":
            self.data_loss_fn = nn.HuberLoss(delta=huber_delta)
        elif loss_type == "mse":
            self.data_loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.data_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute domain loss.

        Args:
            predictions: Model predictions (batch, pred_steps, output_size)
            targets: Ground truth targets (batch, pred_steps, output_size)

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        loss_dict = {}

        # Data loss
        data_loss = self.data_loss_fn(predictions, targets)
        loss_dict["data"] = data_loss.item()
        total_loss = data_loss

        # Derivative loss (rate-of-change matching)
        if self.use_derivative_loss and predictions.shape[1] > 1:
            pred_deriv = predictions[:, 1:, :] - predictions[:, :-1, :]
            target_deriv = targets[:, 1:, :] - targets[:, :-1, :]
            deriv_loss = self.data_loss_fn(pred_deriv, target_deriv)
            total_loss = total_loss + self.derivative_weight * deriv_loss
            loss_dict["derivative"] = deriv_loss.item()
        else:
            loss_dict["derivative"] = 0.0

        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ───────────────────────────────────────────────────────────────────────
# Head Loss (Phase 5 & 6)
# ───────────────────────────────────────────────────────────────────────


class HeadLoss(BaseLoss):
    """
    Loss function for individual decoder heads in federated models.

    Used in Phase 5 (Federated DeepMMNet) and Phase 6 (Physics-Informed).
    """

    def __init__(
        self,
        loss_type: str = "huber",
        huber_delta: float = 0.5,
    ):
        """
        Initialize HeadLoss.

        Args:
            loss_type: Type of loss ('huber', 'mse', 'mae')
            huber_delta: Delta parameter for Huber loss
        """
        super().__init__()
        self.loss_type = loss_type

        if loss_type == "huber":
            self.loss_fn = nn.HuberLoss(delta=huber_delta)
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        else:
            self.loss_fn = nn.L1Loss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for a single head.

        Args:
            pred: Predictions from one decoder head
            target: Ground truth for that head

        Returns:
            Loss tensor (scalar)
        """
        return self.loss_fn(pred, target)


# ───────────────────────────────────────────────────────────────────────
# Physics Constraint Loss (Phase 6)
# ───────────────────────────────────────────────────────────────────────


@dataclass
class PhysicsConstraintConfig:
    """
    Configuration for physics constraints.

    This is a thin wrapper that provides backward-compatible access to the
    canonical PhysicsConfig from ``architecture.federated_pi``. For new code,
    prefer importing ``PhysicsConfig`` directly.

    The canonical implementation defines 17 constraints across three tiers:
    - Tier 1 (8 hard): temperature ordering, HX feasibility, pressure drops,
      PUE bounds, Carnot limit
    - Tier 2 (9 soft): energy conservation (primary/secondary/HX), approach
      temps, wet bulb, temperature/pressure bounds, COP bounds
    - Tier 3 (3 monitor): pump power consistency, load-temperature
      monotonicity, external temp sensitivity
    """

    # ── Physical Constants ────────────────────────────────────────────────
    water_density: float = 997.0          # kg/m³
    water_specific_heat: float = 4186.0   # J/(kg·K)

    # ── Unit Conversions ──────────────────────────────────────────────────
    gpm_to_m3_s: float = 6.30902e-5
    kw_to_w: float = 1000.0
    psi_to_pa: float = 6894.76
    kelvin_to_celsius: float = -273.15

    # ── Temperature Constraints ───────────────────────────────────────────
    min_approach_temp: float = 3.0        # Min HX approach temp (°C)
    wet_bulb_approach_min: float = 5.0    # T_prim_s must be > T_ext + this
    temp_min_c: float = 30.0
    temp_max_c: float = 75.0

    # ── Pressure Constraints ──────────────────────────────────────────────
    pressure_drop_min_psi: float = 0.5
    pressure_min_psig: float = 0.0
    pressure_max_psig: float = 100.0

    # ── Energy Balance ────────────────────────────────────────────────────
    energy_balance_tolerance: float = 0.30   # 30% relative for secondary
    energy_balance_tol_primary: float = 0.35 # 35% relative for primary
    hx_balance_tolerance: float = 0.20       # 20% HX mismatch

    # ── COP/Carnot ────────────────────────────────────────────────────────
    cop_min: float = 5.0
    cop_max: float = 15.0
    cop_carnot_fraction: float = 0.6

    # ── PUE ───────────────────────────────────────────────────────────────
    pue_min: float = 1.0
    pue_max: float = 1.5

    # ── Pump ──────────────────────────────────────────────────────────────
    pump_eff_typical: float = 0.70

    # ── Numerical ─────────────────────────────────────────────────────────
    epsilon: float = 1e-8
    softplus_beta: float = 10.0   # For smooth constraint activations

    # ── Tier 1: Hard Constraints (Always Active, High Weight) ─────────────
    tier1_weights: Dict[str, float] = field(default_factory=lambda: {
        'temp_ordering_primary': 3.0,
        'temp_ordering_secondary': 3.0,
        'hx_feasibility_hot': 3.0,
        'hx_feasibility_cold': 3.0,
        'pressure_drop_primary': 2.0,
        'pressure_drop_secondary': 2.0,
        'pue_bounds': 2.0,
        'carnot_limit': 1.0,
    })

    # ── Tier 2: Soft Constraints (Active with Tolerance, Medium Weight) ───
    tier2_weights: Dict[str, float] = field(default_factory=lambda: {
        'energy_conservation_secondary': 1.5,
        'energy_conservation_primary': 1.0,
        'energy_balance_hx': 0.5,
        'approach_temp_hot': 1.0,
        'approach_temp_cold': 1.0,
        'wet_bulb_constraint': 0.5,
        'temperature_bounds': 1.0,
        'pressure_bounds': 1.0,
        'cop_bounds': 1.0,
    })

    # ── Tier 3: Monitor Only (NOT used in training, logged) ───────────────
    tier3_names: List[str] = field(default_factory=lambda: [
        'pump_power_consistency',
        'load_temperature_monotonicity',
        'external_temp_sensitivity',
    ])

    def to_physics_config(self) -> 'PhysicsConfig':
        """Convert to the canonical PhysicsConfig used by PhysicsLossCalculator."""
        try:
            from ..architecture.federated_pi import PhysicsConfig
        except ImportError:
            raise ImportError(
                "Cannot convert to PhysicsConfig — "
                "architecture.federated_pi not available"
            )
        return PhysicsConfig(
            water_density=self.water_density,
            water_specific_heat=self.water_specific_heat,
            gpm_to_m3_s=self.gpm_to_m3_s,
            kw_to_w=self.kw_to_w,
            psi_to_pa=self.psi_to_pa,
            kelvin_to_celsius=self.kelvin_to_celsius,
            min_approach_temp=self.min_approach_temp,
            wet_bulb_approach_min=self.wet_bulb_approach_min,
            temp_min_c=self.temp_min_c,
            temp_max_c=self.temp_max_c,
            pressure_drop_min_psi=self.pressure_drop_min_psi,
            pressure_min_psig=self.pressure_min_psig,
            pressure_max_psig=self.pressure_max_psig,
            energy_balance_tolerance=self.energy_balance_tolerance,
            energy_balance_tol_primary=self.energy_balance_tol_primary,
            hx_balance_tolerance=self.hx_balance_tolerance,
            cop_min=self.cop_min,
            cop_max=self.cop_max,
            cop_carnot_fraction=self.cop_carnot_fraction,
            pue_min=self.pue_min,
            pue_max=self.pue_max,
            pump_eff_typical=self.pump_eff_typical,
            epsilon=self.epsilon,
            softplus_beta=self.softplus_beta,
            tier1_weights=dict(self.tier1_weights),
            tier2_weights=dict(self.tier2_weights),
            tier3_names=list(self.tier3_names),
        )


class PhysicsConstraintLoss(BaseLoss):
    """
    Physics-informed loss function with tiered constraints.

    Delegates to the canonical ``PhysicsLossCalculator`` from
    ``architecture.federated_pi``, which implements all 17 physics
    constraints from the Phase 6 notebook:

    **Tier 1 — Hard (8 constraints, high weight):**
    Temperature ordering (primary/secondary), HX feasibility (hot/cold end),
    pressure drops (primary/secondary), PUE bounds, Carnot limit.

    **Tier 2 — Soft (9 constraints, medium weight, with tolerance):**
    Energy conservation (secondary/primary/HX balance), approach temps
    (hot/cold), wet bulb, temperature bounds, pressure bounds, COP bounds.

    **Tier 3 — Monitor only (3 metrics, logged but not backpropagated):**
    Pump power consistency, load-temperature monotonicity,
    external temp sensitivity.

    Used in Phase 6 (Physics-Informed Federated DeepMMNet).
    """

    def __init__(self, config: Optional[PhysicsConstraintConfig] = None):
        """
        Initialize PhysicsConstraintLoss.

        Args:
            config: Physics constraint configuration. If None, uses defaults
                    matching the canonical Phase 6 notebook.
        """
        super().__init__()
        self.config = config or PhysicsConstraintConfig()

        # Import and instantiate the canonical calculator
        from ..architecture.federated_pi import PhysicsConfig, PhysicsLossCalculator

        # Build the canonical PhysicsConfig from our config
        self._physics_config = self.config.to_physics_config()
        self._calculator = PhysicsLossCalculator(self._physics_config)

    def compute_training_losses(
        self,
        physics_vars: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute all physics constraint losses.

        Delegates to the canonical ``PhysicsLossCalculator`` which implements
        all Tier 1 (hard) and Tier 2 (soft) constraints from the Phase 6
        notebook, plus Tier 3 monitoring metrics.

        Args:
            physics_vars: Dictionary of physics variables from PredictionToPhysics.
                Expected keys include per-CDU tensors like ``T_prim_s``,
                ``T_prim_r``, ``T_sec_s``, ``T_sec_r``, ``p_prim_s``,
                ``p_prim_r``, ``p_sec_s``, ``p_sec_r``, ``V_flow_prim``,
                ``V_flow_sec``, ``W_flow``, ``Q_flow_total_kW``, ``T_ext_C``.

        Returns:
            Tuple of:
            - training_losses: Dict of constraint name -> loss tensor (Tier 1 & 2)
            - monitoring_losses: Dict of constraint name -> metric value (Tier 3)
        """
        return self._calculator.compute_training_losses(physics_vars)

    def compute_weighted_total(
        self,
        training_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute weighted sum of training losses using canonical tier weights.

        Args:
            training_losses: Dict from compute_training_losses

        Returns:
            Total physics loss tensor
        """
        return self._calculator.compute_weighted_total(training_losses)

    def forward(
        self,
        physics_vars: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total physics constraint loss.

        Args:
            physics_vars: Dictionary of physics variables

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        training_losses, monitoring_losses = self.compute_training_losses(physics_vars)
        total_loss = self.compute_weighted_total(training_losses)

        loss_dict = {
            "physics_total": total_loss.item(),
            **{k: v.item() for k, v in training_losses.items()},
            **{k: v for k, v in monitoring_losses.items()},
        }

        return total_loss, loss_dict


# ───────────────────────────────────────────────────────────────────────
# Unified Surrogate Loss (Dispatcher)
# ───────────────────────────────────────────────────────────────────────


class SurrogateLoss(BaseLoss):
    """
    Unified loss function that dispatches to the correct phase-specific loss.

    Automatically selects the appropriate loss based on model phase:
    - Phase 1 (LSTM): Huber loss
    - Phase 2 (Basic DeepONet): HybridLoss
    - Phase 3 (Hybrid DeepONet): HybridLoss with alpha regularization
    - Phase 4 (Domain DeepONet): DomainLoss
    - Phase 5 (Federated): HeadLoss for 6 heads
    - Phase 6 (Physics-Informed): HeadLoss + PhysicsConstraintLoss
    """

    # Head name mappings for Phase 5/6
    ALL_HEAD_NAMES = ["G_T", "G_V", "G_p", "G_Vs", "G_ps", "G_W"]
    HEAD_TARGET_MAP = {
        "G_T": "y_delta_T",
        "G_V": "y_delta_V",
        "G_p": "y_delta_p",
        "G_Vs": "y_delta_Vs",
        "G_ps": "y_delta_ps",
        "G_W": "y_delta_W",
    }
    HEAD_PRED_MAP = {
        "G_T": "pred_T",
        "G_V": "pred_V",
        "G_p": "pred_p",
        "G_Vs": "pred_Vs",
        "G_ps": "pred_ps",
        "G_W": "pred_W",
    }

    def __init__(
        self,
        phase: str,
        config: Optional[Any] = None,
        physics_config: Optional[PhysicsConstraintConfig] = None,
    ):
        """
        Initialize SurrogateLoss.

        Args:
            phase: Model phase ('lstm', 'deeponet', 'hybrid_deeponet',
                   'domain_deeponet', 'federated', 'federated_pi')
            config: Phase-specific configuration
            physics_config: Physics constraint configuration (Phase 6 only)
        """
        super().__init__()
        self.phase = phase
        self.config = config

        def _cfg(*keys: str, default: Any = None) -> Any:
            if config is None:
                return default
            for key in keys:
                if hasattr(config, key):
                    return getattr(config, key)
            return default

        # Initialize phase-specific loss
        if phase == "lstm":
            self._loss = HuberLoss(
                delta=_cfg("huber_delta", "HUBER_DELTA", default=1.0)
            )
        elif phase in ("deeponet", "hybrid_deeponet"):
            self._loss = HybridLoss(
                temporal_weight=_cfg("temporal_weight", "TEMPORAL_WEIGHT", default=1.0),
                algebraic_weight=_cfg("algebraic_loss_weight", "algebraic_weight", "ALGEBRAIC_WEIGHT", default=2.0),
                huber_delta=_cfg("temporal_huber_delta", "huber_delta", "HUBER_DELTA", default=0.5),
                var_weight=_cfg("variance_loss_weight", "var_weight", "VAR_WEIGHT", default=0.1),
                alpha_reg_weight=(
                    _cfg("alpha_reg_weight", "ALPHA_REG_WEIGHT", default=0.01)
                    if config and phase == "hybrid_deeponet"
                    else 0.0
                ),
            )
        elif phase == "domain_deeponet":
            self._loss = DomainLoss(
                loss_type=_cfg("loss_type", "LOSS_TYPE", default="huber"),
                huber_delta=_cfg("huber_delta", "HUBER_DELTA", default=0.5),
                use_derivative_loss=_cfg("use_derivative_loss", "USE_DERIVATIVE_LOSS", default=False),
                derivative_weight=_cfg("derivative_weight", "DERIVATIVE_WEIGHT", default=0.1),
            )
        elif phase in ("federated", "federated_pi"):
            self._loss = HeadLoss(
                loss_type=_cfg("loss_type", "LOSS_TYPE", default="huber"),
                huber_delta=_cfg("huber_delta", "HUBER_DELTA", default=0.5),
            )
            if phase == "federated_pi":
                self._physics_loss = PhysicsConstraintLoss(physics_config)
            else:
                self._physics_loss = None
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss based on phase.

        Args vary by phase:
        - Phase 1: (pred, target)
        - Phase 2/3: (pred_temporal, target_temporal, pred_algebraic, target_algebraic, [alpha])
        - Phase 4: (predictions, targets)
        - Phase 5/6: (output_dict, batch_dict, [physics_vars, physics_weight])

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if self.phase == "lstm":
            return self._loss(*args, **kwargs)

        elif self.phase in ("deeponet", "hybrid_deeponet"):
            return self._loss(*args, **kwargs)

        elif self.phase == "domain_deeponet":
            return self._loss(*args, **kwargs)

        elif self.phase in ("federated", "federated_pi"):
            return self._compute_federated_loss(*args, **kwargs)

        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def _compute_federated_loss(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        physics_vars: Optional[Dict[str, torch.Tensor]] = None,
        physics_weight: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute federated loss across all 6 decoder heads.

        Args:
            output: Model output dict with pred_T, pred_V, etc.
            batch: Batch dict with y_delta_T, y_delta_V, etc.
            physics_vars: Physics variables for physics-informed loss
            physics_weight: Weight for physics loss
            device: Target device

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        if device is None:
            device = next(iter(output.values())).device

        loss_dict = {}
        data_loss = torch.tensor(0.0, device=device)

        # Compute loss for each head
        for head_name in self.ALL_HEAD_NAMES:
            target_key = self.HEAD_TARGET_MAP[head_name]
            pred_key = self.HEAD_PRED_MAP[head_name]

            if pred_key not in output or target_key not in batch:
                continue

            target = batch[target_key]
            if not isinstance(target, torch.Tensor):
                target = torch.tensor(target, device=device)
            elif target.device != device:
                target = target.to(device)

            pred = output[pred_key]
            head_loss = self._loss(pred, target)
            data_loss = data_loss + head_loss
            loss_dict[head_name] = head_loss.item()

        loss_dict["data_total"] = data_loss.item()

        # Physics loss (Phase 6 only)
        physics_loss = torch.tensor(0.0, device=device)
        if self._physics_loss is not None and physics_vars is not None and physics_weight > 0:
            physics_loss, physics_dict = self._physics_loss(physics_vars)
            loss_dict["physics_total"] = physics_loss.item()
            loss_dict.update(physics_dict)

        # Combined loss
        total_loss = data_loss + physics_weight * physics_loss
        loss_dict["total"] = total_loss.item()
        loss_dict["combined_total"] = total_loss.item()

        return total_loss, loss_dict


# ───────────────────────────────────────────────────────────────────────
# Factory Function
# ───────────────────────────────────────────────────────────────────────


def create_loss(
    phase: str,
    config: Optional[Any] = None,
    physics_config: Optional[PhysicsConstraintConfig] = None,
) -> SurrogateLoss:
    """
    Factory function to create a phase-appropriate loss function.

    Args:
        phase: Model phase ('lstm', 'deeponet', 'hybrid_deeponet',
               'domain_deeponet', 'federated', 'federated_pi')
        config: Phase-specific configuration
        physics_config: Physics constraint configuration (Phase 6 only)

    Returns:
        SurrogateLoss instance configured for the specified phase
    """
    return SurrogateLoss(phase=phase, config=config, physics_config=physics_config)