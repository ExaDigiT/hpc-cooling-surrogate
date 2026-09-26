"""
Phase 6: Physics-Informed Federated DeepMMNet.

Extends Phase 5 with:
- PhysicsConfig: Three-tier physics constraint configuration
- PredictionToPhysics: Converts model predictions to physical quantities
- PhysicsLossCalculator: Differentiable physics loss computation
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .configs import PhysicsInformedConfig, FederatedConfig
from .federated import FederatedDeepMMNet


@dataclass
class PhysicsConfig:
    """
    Physics configuration calibrated for datacenter cooling systems.
    
    Three-tier approach:
    - Tier 1: Hard constraints (thermodynamic laws, always active)
    - Tier 2: Soft constraints (with tolerance, medium weight)
    - Tier 3: Monitor only (logged but not backpropagated)
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


class PredictionToPhysics:
    """
    Converts model predictions (normalized deltas) → absolute physics quantities.

    The model outputs normalized deltas for dynamic outputs.
    This class reshapes them into per-CDU tensors for physics computations:
    - T_prim: (B, num_cdus), T_prim_r: (B, num_cdus), etc.
    Also extracts input features (Q_flow, T_ext) from raw input data.
    """

    def __init__(
        self,
        column_info: Dict[str, Any],
        normalizer: Any,  # FederatedNormalizer
        config: FederatedConfig,
        rate: Optional[int] = None,
    ):
        self.column_info = column_info
        self.normalizer = normalizer
        self.config = config
        self.num_cdus = config.num_cdus
        # Delta scale to use when denormalizing predictions. For the multi-rate
        # physics grid this is the physics_rate scale; rate=None keeps the legacy
        # single-rate default.
        self.rate = rate

        # Pre-compute index maps: output_type → indices in dynamic_cols
        self.type_indices = column_info.get('type_indices', {})

        # Pre-compute input column indices for Q_flow and T_ext
        input_cols = column_info['input_cols']
        self.q_flow_input_indices = []
        self.t_air_input_indices = []
        self.t_ext_input_idx = None

        for i, col in enumerate(input_cols):
            col_lower = col.lower()
            if 'q_flow' in col_lower or 'qflow' in col_lower:
                self.q_flow_input_indices.append(i)
            elif 't_air' in col_lower:
                self.t_air_input_indices.append(i)
            elif 't_ext' in col_lower:
                self.t_ext_input_idx = i

        # Pre-compute delta scales as tensor for fast conversion (rate-aware)
        dynamic_cols = column_info['dynamic_cols']
        if rate is not None and hasattr(normalizer, '_delta_for'):
            _delta = normalizer._delta_for(rate)
        else:
            _delta = normalizer.delta_normalizer
        self.delta_scales = torch.tensor(
            [_delta.get_scale(col) for col in dynamic_cols],
            dtype=torch.float32,
        )

    def convert(
        self,
        pred_normalized: torch.Tensor,
        last_dynamic: torch.Tensor,
        raw_inputs_last: torch.Tensor,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Convert batch predictions to physics-ready per-CDU tensors.

        Args:
            pred_normalized: (B, K, n_dynamic) — model output (normalized deltas)
            last_dynamic: (B, n_dynamic) — raw absolute values at last history step
            raw_inputs_last: (B, n_inputs) — raw input features at last history step
            device: target device

        Returns:
            Dict of per-CDU tensors ready for physics loss computation.
            For K=1, shapes are (B, num_cdus).
        """
        B, K, D = pred_normalized.shape
        scales = self.delta_scales.to(device)

        # Convert normalized deltas → absolute values (differentiable)
        denorm_deltas = pred_normalized * scales.unsqueeze(0).unsqueeze(0)
        pred_absolute = last_dynamic.unsqueeze(1) + torch.cumsum(denorm_deltas, dim=1)

        # For K=1, squeeze time dimension; otherwise take last step
        if K == 1:
            pred_abs = pred_absolute.squeeze(1)
        else:
            pred_abs = pred_absolute[:, -1, :]

        # Extract per-CDU output tensors
        result = {}
        for output_type, indices in self.type_indices.items():
            if indices:
                idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                result[output_type] = pred_abs[:, idx_tensor]

        # Extract input features
        if self.q_flow_input_indices:
            q_idx = torch.tensor(self.q_flow_input_indices, dtype=torch.long, device=device)
            q_flow = raw_inputs_last[:, q_idx]
            # Convert to kW if in W (check magnitude)
            if q_flow.abs().max() > 10000:
                q_flow = q_flow / 1000.0
            result['Q_flow_total_kW'] = q_flow
        else:
            result['Q_flow_total_kW'] = torch.ones(B, self.num_cdus, device=device) * 30.0

        if self.t_ext_input_idx is not None:
            t_ext = raw_inputs_last[:, self.t_ext_input_idx]
            result['T_ext_K'] = t_ext
            # Convert Kelvin to Celsius if needed
            if t_ext.mean() > 200:  # Likely in Kelvin
                result['T_ext_C'] = t_ext - 273.15
            else:
                result['T_ext_C'] = t_ext
        else:
            result['T_ext_C'] = torch.ones(B, device=device) * 20.0

        return result

    def __call__(
        self,
        pred_normalized: torch.Tensor,
        last_dynamic: torch.Tensor,
        raw_inputs_last: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Make instance callable — delegates to convert()."""
        device = pred_normalized.device
        return self.convert(pred_normalized, last_dynamic, raw_inputs_last, device)


class PhysicsLossCalculator:
    """
    Physics-based loss calculator for training.
    
    All losses return differentiable torch tensors.
    Losses are organized into Tier 1 (hard), Tier 2 (soft), and Tier 3 (monitor only).
    """

    def __init__(self, physics_config: PhysicsConfig):
        self.pcfg = physics_config
        self.eps = physics_config.epsilon

    def compute_training_losses(
        self,
        phys: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute all physics losses for training.

        Args:
            phys: Dict from PredictionToPhysics.convert()

        Returns:
            training_losses: Dict of differentiable tensors (Tier 1 + Tier 2)
            monitoring_losses: Dict of float values (Tier 3, for logging only)
        """
        training_losses = {}
        monitoring_losses = {}

        # Check if we have the required fields
        required_fields = ['T_prim_s', 'T_prim_r', 'T_sec_s', 'T_sec_r']
        if not all(f in phys for f in required_fields):
            # Return empty losses if physics data is incomplete
            return training_losses, monitoring_losses

        # ── Tier 1: Hard Constraints ──────────────────────────────────────
        training_losses.update(self._tier1_temperature_ordering(phys))
        training_losses.update(self._tier1_hx_feasibility(phys))
        
        if 'p_prim_s' in phys and 'p_sec_s' in phys:
            training_losses.update(self._tier1_pressure_drops(phys))
        
        if 'Q_flow_total_kW' in phys and 'W_flow' in phys:
            training_losses.update(self._tier1_pue_bounds(phys))
            training_losses.update(self._tier1_carnot_limit(phys))

        # ── Tier 2: Soft Constraints ──────────────────────────────────────
        if 'V_flow_sec' in phys:
            training_losses.update(self._tier2_energy_conservation(phys))
        
        training_losses.update(self._tier2_approach_temps(phys))
        
        if 'T_ext_C' in phys:
            training_losses.update(self._tier2_wet_bulb(phys))
        
        training_losses.update(self._tier2_bounds(phys))
        
        if 'Q_flow_total_kW' in phys and 'W_flow' in phys:
            training_losses.update(self._tier2_cop_bounds(phys))

        # ── Tier 3: Monitoring Only ───────────────────────────────────────
        monitoring_losses.update(self._tier3_monitoring(phys))

        return training_losses, monitoring_losses

    def compute_weighted_total(
        self,
        training_losses: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted sum of all training physics losses."""
        if not training_losses:
            return torch.tensor(0.0)
        
        device = next(iter(training_losses.values())).device
        total = torch.tensor(0.0, device=device)

        for name, loss in training_losses.items():
            # Look up weight from Tier 1 or Tier 2
            weight = self.pcfg.tier1_weights.get(name, 0.0)
            if weight == 0.0:
                weight = self.pcfg.tier2_weights.get(name, 0.0)
            total = total + weight * loss

        return total

    # ══════════════════════════════════════════════════════════════════════
    # TIER 1: Hard Constraints (Thermodynamic Laws)
    # ══════════════════════════════════════════════════════════════════════

    def _tier1_temperature_ordering(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """2nd Law: return temp > supply temp for both loops."""
        losses = {}
        beta = self.pcfg.softplus_beta
        
        # Primary: T_prim_r > T_prim_s (heat absorbed → return is hotter)
        losses['temp_ordering_primary'] = F.softplus(
            phys['T_prim_s'] - phys['T_prim_r'] + 0.1, beta=beta
        ).mean()
        
        # Secondary: T_sec_r > T_sec_s (heat rejected → return is hotter)
        losses['temp_ordering_secondary'] = F.softplus(
            phys['T_sec_s'] - phys['T_sec_r'] + 0.1, beta=beta
        ).mean()
        
        return losses

    def _tier1_hx_feasibility(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Heat exchanger: secondary side must be hotter than primary side."""
        losses = {}
        beta = self.pcfg.softplus_beta
        
        # Hot end: T_sec_r > T_prim_r
        losses['hx_feasibility_hot'] = F.softplus(
            phys['T_prim_r'] - phys['T_sec_r'] + 0.1, beta=beta
        ).mean()
        
        # Cold end: T_sec_s > T_prim_s
        losses['hx_feasibility_cold'] = F.softplus(
            phys['T_prim_s'] - phys['T_sec_s'] + 0.1, beta=beta
        ).mean()
        
        return losses

    def _tier1_pressure_drops(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Fluid flows from high pressure (supply) to low pressure (return)."""
        losses = {}
        cfg = self.pcfg
        
        # Primary: p_s > p_r
        delta_p_prim = phys['p_prim_s'] - phys['p_prim_r']
        losses['pressure_drop_primary'] = F.relu(cfg.pressure_drop_min_psi - delta_p_prim).mean()
        
        # Secondary: p_s > p_r
        delta_p_sec = phys['p_sec_s'] - phys['p_sec_r']
        losses['pressure_drop_secondary'] = F.relu(cfg.pressure_drop_min_psi - delta_p_sec).mean()
        
        return losses

    def _tier1_pue_bounds(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """PUE ≥ 1.0 by definition."""
        losses = {}
        cfg = self.pcfg
        
        total_IT_kW = phys['Q_flow_total_kW'].sum(dim=1)
        total_pump_kW = phys['W_flow'].sum(dim=1)
        calculated_pue = (total_IT_kW + total_pump_kW) / (total_IT_kW + self.eps)
        
        pue_low = F.relu(cfg.pue_min - calculated_pue)
        pue_high = F.relu(calculated_pue - cfg.pue_max)
        losses['pue_bounds'] = (pue_low.mean() + pue_high.mean())
        
        return losses

    def _tier1_carnot_limit(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Cannot exceed Carnot efficiency."""
        losses = {}
        cfg = self.pcfg
        
        total_cooling_W = phys['Q_flow_total_kW'].sum(dim=1) * cfg.kw_to_w
        total_pump_W = phys['W_flow'].sum(dim=1) * cfg.kw_to_w
        actual_cop = total_cooling_W / (total_pump_W + self.eps)

        T_cold_K = phys['T_prim_s'].mean(dim=1) + 273.15
        T_hot_K = phys['T_sec_r'].mean(dim=1) + 273.15
        T_diff = torch.clamp(T_hot_K - T_cold_K, min=1.0)
        carnot_cop = T_cold_K / T_diff
        max_cop = cfg.cop_carnot_fraction * carnot_cop

        losses['carnot_limit'] = F.relu(actual_cop - max_cop).mean()
        return losses

    # ══════════════════════════════════════════════════════════════════════
    # TIER 2: Soft Constraints (with Tolerance)
    # ══════════════════════════════════════════════════════════════════════

    def _tier2_energy_conservation(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Energy conservation for primary and secondary loops."""
        losses = {}
        cfg = self.pcfg
        
        Q_load_W = phys['Q_flow_total_kW'] * cfg.kw_to_w

        # Secondary loop
        V_sec_m3s = phys['V_flow_sec'] * cfg.gpm_to_m3_s
        m_dot_sec = V_sec_m3s * cfg.water_density
        delta_T_sec = torch.abs(phys['T_sec_r'] - phys['T_sec_s'])
        Q_sec_W = m_dot_sec * cfg.water_specific_heat * delta_T_sec

        energy_error_sec = torch.abs(Q_load_W - Q_sec_W) / (Q_load_W + self.eps)
        losses['energy_conservation_secondary'] = F.relu(
            energy_error_sec - cfg.energy_balance_tolerance
        ).mean()

        # Primary loop (if available)
        if 'V_flow_prim' in phys:
            V_prim_m3s = phys['V_flow_prim'] * cfg.gpm_to_m3_s
            m_dot_prim = V_prim_m3s * cfg.water_density
            delta_T_prim = torch.abs(phys['T_prim_r'] - phys['T_prim_s'])
            Q_prim_W = m_dot_prim * cfg.water_specific_heat * delta_T_prim

            energy_error_prim = torch.abs(Q_load_W - Q_prim_W) / (Q_load_W + self.eps)
            losses['energy_conservation_primary'] = F.relu(
                energy_error_prim - cfg.energy_balance_tol_primary
            ).mean()

            # HX balance
            hx_error = torch.abs(Q_sec_W - Q_prim_W) / (Q_load_W + self.eps)
            losses['energy_balance_hx'] = F.relu(
                hx_error - cfg.hx_balance_tolerance
            ).mean()

        return losses

    def _tier2_approach_temps(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Minimum approach temperature for heat exchange."""
        losses = {}
        cfg = self.pcfg
        
        approach_hot = phys['T_sec_r'] - phys['T_prim_r']
        approach_cold = phys['T_sec_s'] - phys['T_prim_s']
        
        losses['approach_temp_hot'] = F.relu(cfg.min_approach_temp - approach_hot).mean()
        losses['approach_temp_cold'] = F.relu(cfg.min_approach_temp - approach_cold).mean()
        
        return losses

    def _tier2_wet_bulb(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Primary supply must be above wet-bulb temperature."""
        losses = {}
        cfg = self.pcfg
        
        T_ext_C = phys['T_ext_C']
        if T_ext_C.dim() == 1:
            T_ext_C = T_ext_C.unsqueeze(1).expand_as(phys['T_prim_s'])
        
        wet_bulb_margin = phys['T_prim_s'] - T_ext_C
        losses['wet_bulb_constraint'] = F.relu(cfg.wet_bulb_approach_min - wet_bulb_margin).mean()
        
        return losses

    def _tier2_bounds(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """Temperature and pressure within physical bounds."""
        losses = {}
        cfg = self.pcfg

        # Temperature bounds
        all_temps = torch.cat([
            phys['T_prim_s'], phys['T_prim_r'],
            phys['T_sec_s'], phys['T_sec_r']
        ], dim=1)
        
        temp_low = F.relu(cfg.temp_min_c - all_temps)
        temp_high = F.relu(all_temps - cfg.temp_max_c)
        losses['temperature_bounds'] = (temp_low.mean() + temp_high.mean())

        # Pressure bounds (if available)
        if 'p_prim_s' in phys:
            all_pressures = torch.cat([
                phys['p_prim_s'], phys['p_prim_r'],
                phys['p_sec_s'], phys['p_sec_r']
            ], dim=1)
            
            p_low = F.relu(cfg.pressure_min_psig - all_pressures)
            p_high = F.relu(all_pressures - cfg.pressure_max_psig)
            losses['pressure_bounds'] = (p_low.mean() + p_high.mean())

        return losses

    def _tier2_cop_bounds(self, phys: Dict) -> Dict[str, torch.Tensor]:
        """COP within reasonable bounds."""
        losses = {}
        cfg = self.pcfg
        
        total_cooling_W = phys['Q_flow_total_kW'].sum(dim=1) * cfg.kw_to_w
        total_pump_W = phys['W_flow'].sum(dim=1) * cfg.kw_to_w
        actual_cop = total_cooling_W / (total_pump_W + self.eps)
        
        cop_low = F.relu(cfg.cop_min - actual_cop)
        cop_high = F.relu(actual_cop - cfg.cop_max)
        losses['cop_bounds'] = (cop_low.mean() + cop_high.mean())
        
        return losses

    # ══════════════════════════════════════════════════════════════════════
    # TIER 3: Monitoring Only
    # ══════════════════════════════════════════════════════════════════════

    def _tier3_monitoring(self, phys: Dict) -> Dict[str, float]:
        """Losses computed for monitoring only — not used in training gradient."""
        monitoring = {}
        cfg = self.pcfg

        with torch.no_grad():
            # Pump power consistency (if data available)
            if all(k in phys for k in ['p_sec_s', 'p_sec_r', 'V_flow_sec', 'W_flow']):
                delta_p_sec = phys['p_sec_s'] - phys['p_sec_r']
                delta_p_pa = delta_p_sec * cfg.psi_to_pa
                V_sec_m3s = phys['V_flow_sec'] * cfg.gpm_to_m3_s
                hydraulic_power_W = delta_p_pa * V_sec_m3s
                expected_pump_kW = hydraulic_power_W / (cfg.pump_eff_typical * cfg.kw_to_w)
                pump_error = torch.abs(phys['W_flow'] - expected_pump_kW) / (
                    phys['W_flow'] + self.eps)
                monitoring['pump_power_consistency'] = pump_error.mean().item()

            # Load-temperature monotonicity
            if 'Q_flow_total_kW' in phys and 'T_sec_r' in phys:
                B = phys['Q_flow_total_kW'].shape[0]
                batch_size = min(B, 50)
                mono_losses = []
                for b in range(batch_size):
                    load_sorted_idx = torch.argsort(phys['Q_flow_total_kW'][b])
                    sorted_T = phys['T_sec_r'][b][load_sorted_idx]
                    mono_losses.append(F.relu(-torch.diff(sorted_T)).mean())
                if mono_losses:
                    monitoring['load_temperature_monotonicity'] = torch.stack(mono_losses).mean().item()

        return monitoring


class PhysicsInformedFederatedDeepMMNet(FederatedDeepMMNet):
    """
    Phase 6: Physics-Informed Federated DeepMMNet.
    
    Extends FederatedDeepMMNet with physics constraint losses for training.
    The model architecture is identical to Phase 5, but training includes
    additional physics-based loss terms.
    
    Parameters
    ----------
    n_inputs : int
        Number of input features
    n_dynamic : int
        Total number of dynamic output features
    column_info : Dict[str, Any]
        Column information with head group indices
    config : PhysicsInformedConfig
        Model configuration with physics settings
    physics_config : PhysicsConfig, optional
        Physics constraint configuration (created with defaults if not provided)
    """

    def __init__(
        self,
        n_inputs: int,
        n_dynamic: int,
        column_info: Dict[str, Any],
        config: PhysicsInformedConfig,
        physics_config: Optional[PhysicsConfig] = None,
    ):
        # Initialize base FederatedDeepMMNet
        super().__init__(n_inputs, n_dynamic, column_info, config)
        self.column_info = column_info
        
        # Store physics-specific config
        self.pi_config = config
        self.physics_config = physics_config or PhysicsConfig()
        
        # Create physics loss calculator
        self.physics_loss_calculator = PhysicsLossCalculator(self.physics_config)
        
        # PredictionToPhysics converter will be set externally after normalizer is available
        self.pred_to_physics: Optional[PredictionToPhysics] = None

    def set_physics_converter(
        self,
        normalizer: Any,  # FederatedNormalizer
    ) -> None:
        """
        Set up the prediction-to-physics converter.
        
        Must be called after the normalizer is fitted and before computing physics losses.
        
        Parameters
        ----------
        normalizer : FederatedNormalizer
            Fitted normalizer for converting predictions
        """
        # Multi-rate physics runs on the shared physics grid, so denormalize with
        # the physics_rate delta scales; single-rate keeps the default.
        rate = self.config.physics_rate if getattr(self.config, 'multi_rate', False) else None
        self.pred_to_physics = PredictionToPhysics(
            column_info=self._get_column_info(),
            normalizer=normalizer,
            config=self.config,
            rate=rate,
        )

    def _get_column_info(self) -> Dict[str, Any]:
        """Return original column_info used at construction."""
        return self.column_info

    def compute_physics_loss(
        self,
        pred_normalized: torch.Tensor,
        last_dynamic: torch.Tensor,
        raw_inputs_last: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute physics-based loss from model predictions.
        
        Parameters
        ----------
        pred_normalized : torch.Tensor
            Model output (normalized deltas), shape (B, K, n_dynamic)
        last_dynamic : torch.Tensor
            Raw absolute values at last history step, shape (B, n_dynamic)
        raw_inputs_last : torch.Tensor
            Raw input features at last history step, shape (B, n_inputs)
            
        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, float]]
            - total_physics_loss: Weighted sum of physics losses
            - training_losses: Individual training loss terms
            - monitoring_losses: Monitoring-only metrics
        """
        if self.pred_to_physics is None:
            # Return zero loss if converter not set up
            device = pred_normalized.device
            return (
                torch.tensor(0.0, device=device),
                {},
                {},
            )
        
        # Convert predictions to physical quantities
        phys = self.pred_to_physics(pred_normalized, last_dynamic, raw_inputs_last)
        
        # Compute physics losses
        training_losses, monitoring_losses = self.physics_loss_calculator.compute_training_losses(phys)
        
        # Compute weighted total
        total_physics_loss = self.physics_loss_calculator.compute_weighted_total(training_losses)
        
        return total_physics_loss, training_losses, monitoring_losses

    def forward_with_physics(
        self,
        u_hist: torch.Tensor,
        y_hist: torch.Tensor,
        last_dynamic: torch.Tensor,
        raw_inputs_last: torch.Tensor,
        K: Optional[int] = None,
        active_heads: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass with physics loss computation.
        
        Parameters
        ----------
        u_hist : torch.Tensor
            Input history, shape (B, H, n_inputs)
        y_hist : torch.Tensor
            Output history, shape (B, H, n_dynamic)
        last_dynamic : torch.Tensor
            Last known dynamic values, shape (B, n_dynamic)
        raw_inputs_last : torch.Tensor
            Raw inputs at last history step, shape (B, n_inputs)
        K : int, optional
            Number of prediction steps
        active_heads : List[str], optional
            Which heads to compute
            
        Returns
        -------
        Dict[str, Any]
            Model outputs plus physics loss information
        """
        # Standard forward pass
        result = self.forward(u_hist, y_hist, K, active_heads)
        
        # Compute physics loss if predictions available
        if 'predictions' in result and self.pred_to_physics is not None:
            physics_loss, training_losses, monitoring_losses = self.compute_physics_loss(
                result['predictions'],
                last_dynamic,
                raw_inputs_last,
            )
            result['physics_loss'] = physics_loss
            result['physics_training_losses'] = training_losses
            result['physics_monitoring_losses'] = monitoring_losses

        return result

    def forward_with_physics_multirate(
        self,
        branch_inputs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        last_dynamic_phys: torch.Tensor,
        raw_inputs_last: torch.Tensor,
        active_heads: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-rate forward + physics on the shared physics grid.

        Per-group predictions are returned on their own ``K`` grids (for the data
        loss); the physics constraints — which couple thermal+hydraulic+power at the
        same instant — are evaluated on the co-located ``predictions_phys`` grid
        (W5b), using ``last_dynamic_phys`` and ``raw_inputs_last`` sampled at
        ``physics_rate``.
        """
        result = self.forward_multirate(branch_inputs, active_heads, physics=True)
        if 'predictions_phys' in result and self.pred_to_physics is not None:
            physics_loss, training_losses, monitoring_losses = self.compute_physics_loss(
                result['predictions_phys'],
                last_dynamic_phys,
                raw_inputs_last,
            )
            result['physics_loss'] = physics_loss
            result['physics_training_losses'] = training_losses
            result['physics_monitoring_losses'] = monitoring_losses
        return result

    @classmethod
    def from_config(
        cls,
        config: PhysicsInformedConfig,
        column_info: Dict[str, Any],
        physics_config: Optional[PhysicsConfig] = None,
    ) -> 'PhysicsInformedFederatedDeepMMNet':
        """Create model from configuration."""
        n_inputs = len(column_info['input_cols'])
        n_dynamic = len(column_info['dynamic_cols'])
        return cls(n_inputs, n_dynamic, column_info, config, physics_config)


def federated_pi(
    config: PhysicsInformedConfig,
    column_info: Dict[str, Any],
    physics_config: Optional[PhysicsConfig] = None,
) -> PhysicsInformedFederatedDeepMMNet:
    """
    Factory function to create Phase 6 PhysicsInformedFederatedDeepMMNet model.
    
    Parameters
    ----------
    config : PhysicsInformedConfig
        Model configuration with physics settings
    column_info : Dict[str, Any]
        Column information with head group indices
    physics_config : PhysicsConfig, optional
        Physics constraint configuration
        
    Returns
    -------
    PhysicsInformedFederatedDeepMMNet
        Instantiated model
    """
    return PhysicsInformedFederatedDeepMMNet.from_config(config, column_info, physics_config)


__all__ = [
    'PhysicsConfig',
    'PredictionToPhysics',
    'PhysicsLossCalculator',
    'PhysicsInformedFederatedDeepMMNet',
    'federated_pi',
]