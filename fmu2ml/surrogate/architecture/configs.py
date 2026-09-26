"""
Unified configuration dataclasses for surrogate models.

Provides:
- SurrogateConfig: Base configuration with system-aware defaults
- LSTMConfig: Phase 1 baseline LSTM configuration
- DeepONetConfig: Phase 2 basic DeepONet configuration  
- HybridDeepONetConfig: Phase 3 hybrid DeepONet configuration
- DomainDeepONetConfig: Phase 4 domain-specific DeepONet configuration
- FederatedConfig: Phase 5 federated DeepMMNet configuration
- PhysicsInformedConfig: Phase 6 physics-informed configuration
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Tuple
from enum import Enum


class OutputType(Enum):
    """Classification of output variables."""
    TEMPORAL = "temporal"      # Dynamic variables with memory (T, V_flow)
    ALGEBRAIC = "algebraic"    # Near-instantaneous variables (p_drop, W_pump)


class SystemName(Enum):
    """Supported HPC system names."""
    SUMMIT = "summit"
    LASSEN = "lassen"
    MARCONI100 = "marconi100"


class DomainType(Enum):
    """Domain types for Phase 4."""
    TEMPERATURE = "temperature"
    FLOW = "flow"
    PRESSURE = "pressure"
    POWER = "power"


# Default system configurations (fallback if RAPS not available)
SYSTEM_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "summit": {
        "NUM_CDUS": 257,
        "RACKS_PER_CDU": 1,
        "NODES_PER_RACK": 18,
        "MIN_POWER": 13.8226,
        "MAX_POWER": 49.3232,
    },
    "lassen": {
        "NUM_CDUS": 44,
        "RACKS_PER_CDU": 1,
        "NODES_PER_RACK": 18,
        "MIN_POWER": 10.86,
        "MAX_POWER": 34.43,
    },
    "marconi100": {
        "NUM_CDUS": 49,
        "RACKS_PER_CDU": 1,
        "NODES_PER_RACK": 20,
        "MIN_POWER": 0,
        "MAX_POWER": 50,
    },
}

# Default column naming patterns
DEFAULT_INPUT_PATTERNS: Dict[str, str] = {
    "T_Air": "T_Air",
    "Q_flow": "Q_flow",
    "T_ext": "T_ext",
}

DEFAULT_OUTPUT_PATTERNS: Dict[str, str] = {
    "T_prim_s": "T_prim_s",
    "T_sec_s": "T_sec_s",
    "T_prim_r": "T_prim_r",
    "T_sec_r": "T_sec_r",
    "p_prim_s": "p_prim_s_psig",
    "p_prim_r": "p_prim_r_psig",
    "p_sec_s": "p_sec_s_psig",
    "p_sec_r": "p_sec_r_psig",
    "V_flow_prim": "V_flow_prim_GPM",
    "V_flow_sec": "V_flow_sec_GPM",
    "W_flow": "W_flow_CDUP_kW",
}

# Classification of outputs as temporal vs algebraic
OUTPUT_CLASSIFICATION: Dict[str, OutputType] = {
    "T_prim_s": OutputType.TEMPORAL,
    "T_sec_s": OutputType.TEMPORAL,
    "T_prim_r": OutputType.TEMPORAL,
    "T_sec_r": OutputType.TEMPORAL,
    "V_flow_prim": OutputType.TEMPORAL,
    "V_flow_sec": OutputType.ALGEBRAIC,
    "p_prim_s": OutputType.ALGEBRAIC,
    "p_prim_r": OutputType.ALGEBRAIC,
    "p_sec_s": OutputType.ALGEBRAIC,
    "p_sec_r": OutputType.ALGEBRAIC,
    "W_flow": OutputType.ALGEBRAIC,
}

# Domain-specific output groupings for Phase 4
DOMAIN_OUTPUTS: Dict[str, List[str]] = {
    "temperature": ["T_prim_s", "T_sec_s", "T_prim_r", "T_sec_r"],
    "flow": ["V_flow_prim", "V_flow_sec"],
    "pressure": ["p_prim_s", "p_prim_r", "p_sec_s", "p_sec_r"],
    "power": ["W_flow"],
}

# Federated head groupings for Phase 5/6
FEDERATED_HEAD_OUTPUTS: Dict[str, List[str]] = {
    "G_T": ["T_prim_s", "T_sec_s", "T_prim_r", "T_sec_r"], # Temperature head (standard)
    "G_V": ["V_flow_prim"],                            # Primary flow head (standard)
    "G_p": ["p_prim_s", "p_prim_r"],                   # Primary pressure head (standard)
    "G_Vs": ["V_flow_sec"],                            # Secondary flow head (skip)
    "G_ps": ["p_sec_s", "p_sec_r"],                    # Secondary pressure head (skip)
    "G_W": ["W_flow"],                                 # Power head (skip)
}

# Head types for Phase 5/6
HEAD_TYPES: Dict[str, str] = {
    "G_T": "standard",
    "G_V": "standard",
    "G_p": "standard",
    "G_Vs": "skip",
    "G_ps": "skip",
    "G_W": "skip",
}


def _get_system_num_cdus(system_name: str) -> int:
    """Get NUM_CDUS for a system, trying RAPS first then fallback."""
    try:
        from raps.config import ConfigManager
        config_manager = ConfigManager(system_name=system_name)
        raps_config = config_manager.get_config()
        return raps_config.get('NUM_CDUS', SYSTEM_DEFAULTS.get(system_name, {}).get('NUM_CDUS', 49))
    except (ImportError, FileNotFoundError, Exception):
        return SYSTEM_DEFAULTS.get(system_name, {}).get('NUM_CDUS', 49)


def zero_persistence_heads(out: Dict[str, Any], pred_keys: Dict[str, str],
                           heads: List[str]) -> Dict[str, Any]:
    """Force the excluded heads' predicted deltas to exactly zero, in place.

    `heads` is `config.persistence_heads`; `pred_keys` maps a group name to its
    key in `out` (each model knows its own layout). A zero delta reconstructs
    through Eq. (delta_recon) to exactly the last observed state, i.e. exact
    persistence -- which is the correct output for a channel the spectral
    budget has declared unpredictable, and is reproducible across architectures
    rather than depending on how a head happened to initialise.

    Implemented as `t * 0.0` rather than `torch.zeros_like(t)` DELIBERATELY.
    zeros_like detaches the head from the autograd graph, which makes its
    parameters unused; phases 1-4 run DDP with `find_unused_parameters=False`
    (see trainer._setup_distributed), so that would abort training under
    multi-GPU. Multiplying keeps the head in the graph with an exactly-zero
    gradient, costing one elementwise op per head per step.
    """
    for h in heads:
        key = pred_keys.get(h)
        if key is not None and key in out:
            out[key] = out[key] * 0.0
    return out


@dataclass
class SurrogateConfig:
    """
    Base configuration for all surrogate models.
    
    Attributes
    ----------
    system_name : str
        HPC system name ('summit', 'lassen', 'marconi100')
    num_cdus : int
        Number of CDUs in the system (auto-detected from system_name if not provided)
    cdu_ids : List[int]
        List of CDU indices (auto-generated from num_cdus if not provided)
    
    history_steps : int
        Number of historical time steps for input sequence
    prediction_steps : int
        Number of future time steps to predict
    subsample_factor : int
        Factor to subsample time series data
    
    train_ratio : float
        Fraction of data for training
    val_ratio : float
        Fraction of data for validation
    batch_size : int
        Training batch size
    
    time_col : str
        Name of timestamp column in data
    input_patterns : Dict[str, str]
        Column name patterns for inputs (with {} placeholder for CDU ID)
    output_patterns : Dict[str, str]
        Column name patterns for outputs (with {} placeholder for CDU ID)
    """
    
    # System configuration
    system_name: str = "marconi100"
    num_cdus: Optional[int] = None
    cdu_ids: Optional[List[int]] = None
    
    # Sequence configuration
    history_steps: int = 40
    prediction_steps: int = 20
    subsample_factor: int = 10
    
    # Data split configuration
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    # test_ratio is implicitly 1 - train_ratio - val_ratio

    # Split strategy: "contiguous" (legacy temporal prefix split) or
    # "chunk_shuffled" (regime-balanced chunk-shuffled split). With
    # chunk_shuffled, chunk_size="auto" derives the size from a constraint
    # solver + measured operational cycle (see data.dataset.suggest_chunk_size);
    # an explicit int overrides but is still validated against the same checks.
    split_strategy: str = "chunk_shuffled"
    chunk_size: Any = "auto"
    min_windows_per_chunk: int = 15   # statistical floor per chunk
    min_chunks_per_split: int = 3     # regime-diversity floor for val/test
    coverage_factor: float = 1.0      # chunk should span >= this many full cycles
    split_seed: int = 42

    # Explicit per-file hold-out (used by split_strategy="per_file" / the memmap
    # store): chunk ids (the <id> in chunk_<id>/) reserved for val/test; every
    # other chunk is train. None -> fall back to the seeded chunk shuffle.
    val_chunk_ids: Optional[List[int]] = None
    test_chunk_ids: Optional[List[int]] = None

    # Memmap-store split mode:
    #   "chunk"   -> whole-chunk hold-out (explicit val/test_chunk_ids, else a
    #                seeded chunk shuffle). Val/test are *separate* chunks, so a
    #                regime-heterogeneous chunk makes val look like a different
    #                system and early-stopping fires prematurely.
    #   "segment" -> carve chunks into contiguous train/val/test time-segments
    #                (by train_ratio/val_ratio) with a guard gap so no window's
    #                history/horizon crosses a segment boundary (no leakage).
    #                * test_chunk_ids is None: every chunk feeds all three splits,
    #                  so val/test match the train distribution. Best with few
    #                  chunks where a whole-chunk val looks like a different system.
    #                * test_chunk_ids set: those whole chunks are the held-out test
    #                  set (true generalization check) and the remaining chunks are
    #                  carved into train/val only. Recommended hybrid.
    store_split_mode: str = "chunk"

    # Training configuration
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    gradient_clip: float = 1.0

    # ── Per-head loss weighting ────────────────────────────────────────────────
    # Lives on the BASE config (not just FederatedConfig) because every phase
    # 1-6 trains through a per-head multi-rate loss and every one of them needs
    # to be able to set these. Defaults are all 1.0, i.e. exactly the previous
    # behaviour — nothing changes unless a run opts in.
    #
    # WHY THIS MATTERS. With uniform weights the raw Huber losses are wildly
    # unbalanced: on `systematic-720`, G_Vs contributes ~52% and G_ps ~47% of
    # the total, so the four heads that actually carry learnable signal
    # (G_T, G_V, G_p, G_W) share ~1.2% of the gradient between them. G_Vs is a
    # measured channel noise floor (its deltas are temporally white), so half
    # the optimisation budget is spent on something no model can fit. Every
    # architecture converges to the same loss in one epoch and then flatlines.
    #
    # Set a head's weight to 0.0 to drop it from the objective entirely; it is
    # still predicted and still scored in the report, just not optimised.
    head_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'G_T': 1.0,
        'G_V': 1.0,
        'G_p': 1.0,
        'G_Vs': 1.0,
        'G_ps': 1.0,
        'G_W': 1.0,
    })

    # When True the trainer divides each head's loss by that head's measured
    # PERSISTENCE-BASELINE loss before applying `head_loss_weights`. Targets are
    # consecutive deltas, so the persistence prediction is exactly zero delta
    # and the baseline is loss(0, target) — measurable directly from the data
    # with no model involved.
    #
    # The effect: the objective becomes "fraction of persistence error
    # remaining", summed over heads. A 1% gain on G_T then counts the same as a
    # 1% gain on G_ps, which is precisely what the reported skill score
    # measures. Without this, the loss optimises raw MSE magnitude and is
    # therefore dominated by whichever head happens to have the largest units.
    head_loss_normalize: bool = False
    head_loss_calib_batches: int = 50   # batches used to measure the baseline

    # ── history perturbation (train/deploy mismatch) ───────────────────────────
    # Training is teacher-forced: `y_hist` is always ground truth. Deployment is
    # closed-loop, so `y_hist` is the model's own (imperfect) output, and the
    # model has never seen a history it did not trust. Adding Gaussian noise to
    # the normalised output history during training simulates that degraded
    # input and is the cheap, self-contained half of scheduled sampling: it
    # needs no rollout unroll, no resampling between rates, and no normalizer in
    # the training loop.
    #
    # `history_noise_std` is in NORMALISED output units, so 0.01 is 1% of a
    # typical output scale. 0.0 (default) reproduces previous behaviour exactly.
    # Noise is applied to y_hist only, never to u_hist (exogenous inputs stay
    # known -- see the rollout protocol) and never during validation.
    history_noise_std: float = 0.0
    history_noise_warmup_epochs: int = 0   # ramp 0 -> std over this many epochs

    # Column configuration
    time_col: str = "time"
    input_patterns: Dict[str, str] = field(default_factory=lambda: DEFAULT_INPUT_PATTERNS.copy())
    output_patterns: Dict[str, str] = field(default_factory=lambda: DEFAULT_OUTPUT_PATTERNS.copy())
    output_names: Optional[List[str]] = None
    
    # Data source (optional)
    doi: Optional[str] = None
    data_name: Optional[str] = None
    
    @property
    def persistence_heads(self) -> List[str]:
        """Groups excluded from the objective, which must emit EXACT zero deltas.

        A zero loss weight removes a head from the objective but does NOT make
        its output harmless: the head still runs, receives no gradient, and
        therefore keeps emitting whatever its random initialisation produces.
        Those spurious increments accumulate through the cumsum reconstruction
        of Eq. (delta_recon) into unbounded drift. Measured on the 2026-08-02
        runs, phase 1's untrained `G_Vs` head reached R^2 = -1.46 teacher-forced
        and -2274 under closed-loop rollout, against -0.18 / -0.98 for the
        phases whose heads happened to initialise near zero.

        Models must therefore force these heads to zero structurally, which
        makes the prediction exactly persistence -- the honest output for a
        channel declared unpredictable. See `zero_persistence_heads`.
        """
        return [h for h, w in (self.head_loss_weights or {}).items() if w == 0.0]

    def __post_init__(self):
        """Validate and auto-calculate dimensions."""
        # Auto-detect num_cdus from system if not provided
        if self.num_cdus is None:
            self.num_cdus = _get_system_num_cdus(self.system_name)
        
        # Auto-generate CDU IDs if not provided
        if self.cdu_ids is None:
            self.cdu_ids = list(range(self.num_cdus))
        
        # Auto-generate output names if not provided
        if self.output_names is None:
            self.output_names = list(self.output_patterns.keys())
        
        # Validation
        assert self.num_cdus > 0, "num_cdus must be positive"
        assert self.history_steps > 0, "history_steps must be positive"
        assert self.prediction_steps > 0, "prediction_steps must be positive"
        assert self.subsample_factor >= 1, "subsample_factor must be >= 1"
        assert 0 < self.train_ratio < 1, "train_ratio must be in (0, 1)"
        assert 0 < self.val_ratio < 1, "val_ratio must be in (0, 1)"
        assert self.train_ratio + self.val_ratio < 1, "train_ratio + val_ratio must be < 1"
        assert self.split_strategy in ("contiguous", "chunk_shuffled", "per_file"), \
            "split_strategy must be 'contiguous', 'chunk_shuffled', or 'per_file'"
        assert self.store_split_mode in ("chunk", "segment"), \
            "store_split_mode must be 'chunk' or 'segment'"
        assert self.min_windows_per_chunk >= 1, "min_windows_per_chunk must be >= 1"
        assert self.min_chunks_per_split >= 1, "min_chunks_per_split must be >= 1"
        assert self.coverage_factor > 0, "coverage_factor must be positive"
        assert self.chunk_size == "auto" or (
            isinstance(self.chunk_size, int) and self.chunk_size > 0
        ), "chunk_size must be 'auto' or a positive int"
    
    def __getattr__(self, name: str) -> Any:
        """
        Auto-resolve UPPER_CASE attribute access to snake_case fields.

        This allows notebook-style access patterns like ``config.LEARNING_RATE``
        to transparently resolve to ``config.learning_rate``, maintaining
        backward compatibility with the canonical development notebooks.
        """
        # Only attempt conversion for ALL_UPPER names (with optional digits)
        if name.isupper() or (name.replace('_', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').isupper() and '_' in name):
            snake = name.lower()
            # Use object.__getattribute__ to avoid recursion
            try:
                return object.__getattribute__(self, snake)
            except AttributeError:
                pass
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    @property
    def test_ratio(self) -> float:
        """Compute test ratio from train and val ratios."""
        return 1.0 - self.train_ratio - self.val_ratio
    
    @property
    def n_inputs_per_cdu(self) -> int:
        """Number of input features per CDU (excluding global features)."""
        return len([k for k in self.input_patterns.keys() if k != 'T_ext'])
    
    @property
    def n_outputs_per_cdu(self) -> int:
        """Number of output features per CDU."""
        return len(self.output_patterns)
    
    @property
    def total_outputs(self) -> int:
        """Total number of output features across all CDUs."""
        return self.num_cdus * self.n_outputs_per_cdu
    
    def get_output_type(self, col_name: str) -> OutputType:
        """
        Classify an output column as temporal or algebraic.
        
        Parameters
        ----------
        col_name : str
            Column name to classify
            
        Returns
        -------
        OutputType
            Classification of the output type
        """
        for output_name, output_type in OUTPUT_CLASSIFICATION.items():
            if output_name.lower() in col_name.lower():
                return output_type
        # Default to temporal for unknown columns
        return OutputType.TEMPORAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'system_name': self.system_name,
            'num_cdus': self.num_cdus,
            'cdu_ids': self.cdu_ids,
            'history_steps': self.history_steps,
            'prediction_steps': self.prediction_steps,
            'subsample_factor': self.subsample_factor,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'batch_size': self.batch_size,
            'time_col': self.time_col,
            'input_patterns': self.input_patterns,
            'output_patterns': self.output_patterns,
            'output_names': self.output_names,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SurrogateConfig':
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_system(cls, system_name: str, **kwargs) -> 'SurrogateConfig':
        """
        Create config with system-specific defaults.
        
        Parameters
        ----------
        system_name : str
            HPC system name ('summit', 'lassen', 'marconi100')
        **kwargs
            Additional config overrides
            
        Returns
        -------
        SurrogateConfig
            Configuration with system defaults
        """
        return cls(system_name=system_name, **kwargs)


@dataclass
class MultiRateSpec:
    """
    Mixin: per-decoder-group multi-rate task specification.

    Shared by ALL phase configs so every phase can train and run inference on
    the multi-rate chunk store: per-group sampling rate / look-back / horizon
    plus head -> encoder-branch assignment. With the defaults (uniform rates,
    ``force_multi_rate=False``) the spec is inert and each phase keeps its
    legacy single-rate behavior, datasets and forward path.

    Must be mixed in BEFORE ``SurrogateConfig`` (``class C(MultiRateSpec,
    SurrogateConfig)``) so the ``__post_init__`` chain reaches both.
    """

    # Head -> output-type names (canonical 6-group layout shared by all phases
    # so cross-phase comparisons are on the identical task).
    head_outputs: Dict[str, List[str]] = field(
        default_factory=lambda: FEDERATED_HEAD_OUTPUTS.copy())

    # Per-group sampling rate / look-back / prediction horizon. Empty {} are
    # filled in __post_init__ from the scalar subsample_factor / history_steps /
    # prediction_steps, so a uniform config reproduces the single-rate task.
    head_subsample: Dict[str, int] = field(default_factory=dict)
    head_history_steps: Dict[str, int] = field(default_factory=dict)
    head_prediction_steps: Dict[str, int] = field(default_factory=dict)
    # Head -> encoder-branch id. Heads sharing a branch share one encoder and so
    # must share (rate, history). Empty {} -> all heads on one "shared" branch.
    encoder_groups: Dict[str, str] = field(default_factory=dict)
    # Temporal backbone for the branch encoders (multi-rate mode).
    encoder_type: Literal["lstm", "attention"] = "lstm"
    # Opt into the group-structured multi-rate path even with a uniform spec
    # (e.g. per-group heads all at the same rate, trained from the chunk store).
    force_multi_rate: bool = False

    # ── Shared physics grid (Phase 6) ─────────────────────────────────────
    # Physics constraints relate thermal+hydraulic+power at the same instant, so
    # under multi-rate they are evaluated on one shared reference grid queried
    # via the trunk. None -> use the finest group's (rate, K).
    physics_rate: Optional[int] = None
    physics_K: Optional[int] = None

    def __post_init__(self):
        # Continue the MRO chain (SurrogateConfig and beyond) before deriving
        # the multi-rate maps from the scalar fields it validates.
        sup = super()
        if hasattr(sup, "__post_init__"):
            sup.__post_init__()
        self._init_multi_rate()

    def _init_multi_rate(self) -> None:
        """Fill per-group maps from scalars and validate cross-rate consistency."""
        heads = list(self.head_outputs.keys())
        for h in heads:
            self.head_subsample.setdefault(h, self.subsample_factor)
            self.head_history_steps.setdefault(h, self.history_steps)
            self.head_prediction_steps.setdefault(h, self.prediction_steps)
            self.encoder_groups.setdefault(h, "shared")
            assert self.head_subsample[h] >= 1, f"head_subsample[{h}] must be >= 1"
            assert self.head_history_steps[h] >= 1, f"head_history_steps[{h}] must be >= 1"
            assert self.head_prediction_steps[h] >= 1, f"head_prediction_steps[{h}] must be >= 1"

        # Rates must share a common base (smallest divides all) so window origins
        # align across grids (the store and dataset rely on this).
        rates = sorted({self.head_subsample[h] for h in heads})
        base = rates[0]
        bad = [r for r in rates if r % base != 0]
        assert not bad, (
            f"head_subsample rates must be integer multiples of the smallest "
            f"({base}); offending: {bad}.")

        # Heads on the same encoder branch must share (rate, history).
        from collections import defaultdict
        by_branch = defaultdict(set)
        for h in heads:
            by_branch[self.encoder_groups[h]].add(
                (self.head_subsample[h], self.head_history_steps[h]))
        for b, combos in by_branch.items():
            assert len(combos) == 1, (
                f"encoder branch {b!r} has heads with differing (rate, history) "
                f"{sorted(combos)}; heads sharing a branch must share them.")

        # Physics grid defaults to the finest group's (rate, K).
        if self.physics_rate is None:
            self.physics_rate = base
        if self.physics_K is None:
            fine = [h for h in heads if self.head_subsample[h] == base]
            self.physics_K = max(self.head_prediction_steps[h] for h in fine)
        assert self.physics_rate % base == 0, (
            "physics_rate must be a multiple of the base rate.")

    @property
    def multi_rate(self) -> bool:
        """True iff the per-group spec is non-uniform (or explicitly forced)."""
        if self.force_multi_rate:
            return True
        heads = list(self.head_outputs.keys())
        return (
            len({self.head_subsample[h] for h in heads}) > 1
            or len({self.head_history_steps[h] for h in heads}) > 1
            or len({self.head_prediction_steps[h] for h in heads}) > 1
        )

    @property
    def store_rates(self) -> List[int]:
        """Distinct rates the memmap store must materialize (groups + physics)."""
        heads = list(self.head_outputs.keys())
        rates = {self.head_subsample[h] for h in heads}
        if self.physics_rate is not None:
            rates.add(self.physics_rate)
        return sorted(rates)

    @property
    def all_dynamic_outputs(self) -> List[str]:
        """Get all dynamic output variable names across all heads."""
        outputs = []
        for head_outputs in self.head_outputs.values():
            outputs.extend(head_outputs)
        return outputs


@dataclass
class LSTMConfig(MultiRateSpec, SurrogateConfig):
    """
    Configuration for Phase 1: Baseline LSTM model.
    
    Additional Attributes
    ---------------------
    lstm_hidden_size : int
        Hidden dimension of LSTM layers
    lstm_num_layers : int
        Number of LSTM layers
    attention_heads : int
        Number of attention heads for temporal attention
    dropout : float
        Dropout rate
    decoder_hidden_sizes : List[int]
        Hidden layer sizes for decoder MLP
    """
    
    # LSTM architecture
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.1
    
    # Attention
    attention_heads: int = 4
    attention_dropout: float = 0.1
    
    # Decoder
    decoder_hidden_sizes: List[int] = field(default_factory=lambda: [512, 256])
    decoder_dropout: float = 0.1
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    
    def __post_init__(self):
        super().__post_init__()
        assert self.lstm_hidden_size > 0, "lstm_hidden_size must be positive"
        assert self.lstm_num_layers > 0, "lstm_num_layers must be positive"


@dataclass
class DeepONetConfig(MultiRateSpec, SurrogateConfig):
    """
    Configuration for Phase 2: Basic DeepONet model.
    
    Additional Attributes
    ---------------------
    branch_lstm_hidden : int
        Hidden size for branch network LSTM
    branch_lstm_layers : int
        Number of LSTM layers in branch network
    trunk_n_fourier : int
        Number of Fourier features in trunk network
    trunk_hidden_sizes : List[int]
        Hidden layer sizes for trunk MLP
    basis_dim : int
        Dimension of basis functions (branch and trunk output dim)
    use_input_whitening : bool
        Whether to apply PCA whitening to inputs
    whitening_components : float
        Fraction of variance to retain in PCA (0-1)
    include_output_history : bool
        Whether to include output history in branch input
    output_history_steps : int
        Number of output history steps to include
    algebraic_primary_input : str
        Primary input column pattern for algebraic pathway (e.g., 'q_flow')
    """
    
    # Branch network (LSTM)
    branch_lstm_hidden: int = 256
    branch_lstm_layers: int = 2
    branch_lstm_dropout: float = 0.1
    branch_mlp_hidden: List[int] = field(default_factory=lambda: [256, 128])
    
    # Trunk network (Fourier features)
    trunk_n_fourier: int = 16
    trunk_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    
    # Basis dimension
    basis_dim: int = 64
    
    # Input preprocessing
    use_input_whitening: bool = True
    whitening_components: float = 0.99
    
    # Output history
    include_output_history: bool = True
    output_history_steps: int = 40  # Same as history_steps by default
    
    # Algebraic pathway
    algebraic_primary_input: str = "Q_flow"
    algebraic_mlp_hidden: List[int] = field(default_factory=lambda: [128, 64])
    
    # Skip connection
    use_skip_connection: bool = True
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    
    def __post_init__(self):
        super().__post_init__()
        if self.output_history_steps is None:
            self.output_history_steps = self.history_steps
        assert self.branch_lstm_hidden > 0, "branch_lstm_hidden must be positive"
        assert self.basis_dim > 0, "basis_dim must be positive"
        assert 0 < self.whitening_components <= 1, "whitening_components must be in (0, 1]"


@dataclass
class HybridDeepONetConfig(DeepONetConfig):
    """
    Configuration for Phase 3: Hybrid DeepONet with temporal + algebraic pathways.

    Additional Attributes
    ---------------------
    use_cdu_embedding : bool
        Whether to use CDU embeddings for per-CDU adaptation
    cdu_embedding_dim : int
        Dimension of CDU embeddings
    temporal_loss_type : str
        Loss function for temporal pathway ('huber', 'mse', 'mae')
    temporal_huber_delta : float
        Delta parameter for Huber loss
    variance_loss_weight : float
        Weight for variance matching loss
    algebraic_loss_weight : float
        Weight for algebraic pathway loss relative to temporal
    use_drift_correction : bool
        Whether to apply drift correction for temporal outputs
    algebraic_output_types : List[str]
        Output type name substrings that are classified as ALGEBRAIC.
        Defaults to Phase 3 original: only secondary pressures.
    """

    # CDU embeddings
    use_cdu_embedding: bool = True
    cdu_embedding_dim: int = 16

    # Hybrid loss configuration
    temporal_loss_type: Literal['huber', 'mse', 'mae'] = 'huber'
    temporal_huber_delta: float = 1.0
    variance_loss_weight: float = 0.1
    algebraic_loss_weight: float = 1.0
    alpha_reg_weight: float = 0.01

    # Drift correction
    use_drift_correction: bool = True

    # Enhanced decoder
    temporal_decoder_hidden: List[int] = field(default_factory=lambda: [256, 128])
    algebraic_decoder_hidden: List[int] = field(default_factory=lambda: [128, 64])

    # Phase 3 algebraic output classification: only secondary pressures are algebraic
    algebraic_output_types: List[str] = field(default_factory=lambda: ['p_sec_s', 'p_sec_r'])

    def __post_init__(self):
        super().__post_init__()
        if self.use_cdu_embedding:
            assert self.cdu_embedding_dim > 0, "cdu_embedding_dim must be positive"

    def get_output_type(self, col_name: str) -> 'OutputType':
        """Override: only columns matching algebraic_output_types are ALGEBRAIC."""
        col_lower = col_name.lower()
        for alg_type in self.algebraic_output_types:
            if alg_type.lower() in col_lower:
                return OutputType.ALGEBRAIC
        return OutputType.TEMPORAL


@dataclass
class DomainDeepONetConfig(MultiRateSpec, SurrogateConfig):
    """
    Configuration for Phase 4: Domain-specific DeepONet model.
    
    Each domain (temperature, flow, pressure, power) has its own expert DeepONet.
    
    Additional Attributes
    ---------------------
    domain : str
        Domain type ('temperature', 'flow', 'pressure', 'power')
    domain_outputs : List[str]
        Output variable names for this domain
    branch_lstm_hidden : int
        Hidden size for branch network LSTM
    branch_lstm_layers : int
        Number of LSTM layers in branch network
    trunk_n_fourier : int
        Number of Fourier features in trunk network
    basis_dim : int
        Dimension of basis functions
    include_output_history : bool
        Whether to include output history in branch input
    """
    
    # Domain specification
    domain: str = "temperature"
    domain_outputs: Optional[List[str]] = None
    
    # Branch network (LSTM)
    branch_lstm_hidden: int = 256
    branch_lstm_layers: int = 2
    branch_lstm_dropout: float = 0.1
    branch_mlp_hidden: List[int] = field(default_factory=lambda: [256, 128])
    
    # Trunk network (Fourier features)
    trunk_n_fourier: int = 16
    trunk_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    
    # Basis dimension
    basis_dim: int = 64
    
    # Output history
    include_output_history: bool = True
    output_history_steps: int = 40
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    
    def __post_init__(self):
        super().__post_init__()
        # Set domain outputs if not provided
        if self.domain_outputs is None:
            self.domain_outputs = DOMAIN_OUTPUTS.get(self.domain, [])
        assert self.domain in DOMAIN_OUTPUTS, f"Unknown domain: {self.domain}"
        assert self.branch_lstm_hidden > 0, "branch_lstm_hidden must be positive"
        assert self.basis_dim > 0, "basis_dim must be positive"
    
    @classmethod
    def for_domain(cls, domain: str, **kwargs) -> 'DomainDeepONetConfig':
        """
        Create config for a specific domain.
        
        Parameters
        ----------
        domain : str
            Domain type ('temperature', 'flow', 'pressure', 'power')
        **kwargs
            Additional config overrides
            
        Returns
        -------
        DomainDeepONetConfig
            Configuration for the specified domain
        """
        return cls(domain=domain, domain_outputs=DOMAIN_OUTPUTS[domain], **kwargs)


@dataclass
class FederatedConfig(MultiRateSpec, SurrogateConfig):
    """
    Configuration for Phase 5: Federated DeepMMNet model.
    
    Shared encoder with 6 decoder heads for different output groups.
    
    Additional Attributes
    ---------------------
    u_branch_lstm_hidden : int
        Hidden size for u-branch (input) LSTM
    y_branch_lstm_hidden : int
        Hidden size for y-branch (output history) LSTM
    trunk_n_fourier : int
        Number of Fourier features in trunk network
    basis_dim : int
        Dimension of basis functions (shared across heads)
    head_hidden_sizes : Dict[str, List[int]]
        Hidden layer sizes for each decoder head
    head_outputs : Dict[str, List[str]]
        Output variable names for each decoder head
    """
    
    # U-branch (input history encoder)
    u_branch_lstm_hidden: int = 256
    u_branch_lstm_layers: int = 2
    u_branch_lstm_dropout: float = 0.1
    
    # Y-branch (output history encoder)
    y_branch_lstm_hidden: int = 256
    y_branch_lstm_layers: int = 2
    y_branch_lstm_dropout: float = 0.1
    
    # Trunk network (Fourier features)
    trunk_n_fourier: int = 16
    trunk_hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    
    # Basis dimension (shared across all heads)
    basis_dim: int = 64
    
    # Decoder heads
    head_hidden_sizes: Dict[str, List[int]] = field(default_factory=lambda: {
        'G_T': [256, 128],
        'G_V': [128, 64],
        'G_p': [128, 64],
        'G_Vs': [128, 64],
        'G_ps': [128, 64],
        'G_W': [128, 64],
    })
    
    # Head output groupings come from MultiRateSpec (head_outputs).

    # Head types (standard vs skip)
    head_types: Dict[str, str] = field(default_factory=lambda: HEAD_TYPES.copy())
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 15
    phase1_epochs: int = 50
    phase2_epochs: int = 100
    phase2_head_lr_scale: float = 0.1

    # Loss configuration
    loss_type: Literal['huber', 'mse', 'mae'] = 'huber'
    huber_delta: float = 0.5
    
    # `head_loss_weights` / `head_loss_normalize` are inherited from
    # SurrogateConfig — they used to be declared here, which meant phases 1-4
    # could not set them even though the trainer reads them on every path.

    # Multi-rate spec (head_subsample / head_history_steps / head_prediction_
    # steps / encoder_groups / encoder_type / physics_rate / physics_K) and the
    # multi_rate / store_rates / all_dynamic_outputs properties come from
    # MultiRateSpec.

    def __post_init__(self):
        super().__post_init__()
        assert self.u_branch_lstm_hidden > 0, "u_branch_lstm_hidden must be positive"
        assert self.y_branch_lstm_hidden > 0, "y_branch_lstm_hidden must be positive"
        assert self.basis_dim > 0, "basis_dim must be positive"

    @property
    def standard_heads(self) -> List[str]:
        """Get list of standard (non-skip) head names."""
        return [h for h, t in self.head_types.items() if t == 'standard']
    
    @property
    def skip_heads(self) -> List[str]:
        """Get list of skip head names."""
        return [h for h, t in self.head_types.items() if t == 'skip']


@dataclass
class PhysicsInformedConfig(FederatedConfig):
    """
    Configuration for Phase 6: Physics-Informed Federated DeepMMNet.
    
    Extends FederatedConfig with physics constraint settings.
    
    Additional Attributes
    ---------------------
    use_physics_loss : bool
        Whether to include physics constraint losses
    physics_loss_weight : float
        Overall weight for physics loss term
    tier1_weight : float
        Weight for Tier 1 (hard) constraints
    tier2_weight : float  
        Weight for Tier 2 (soft) constraints
    tier3_weight : float
        Weight for Tier 3 (monitor-only) constraints
    physics_warmup_epochs : int
        Number of epochs before enabling physics loss
    """
    
    # Physics loss configuration
    use_physics_loss: bool = True
    physics_loss_weight: float = 0.1
    physics_weight_max: float = 0.1
    
    # Tier weights (Tier 1: hard, Tier 2: soft, Tier 3: monitor)
    tier1_weight: float = 1.0
    tier2_weight: float = 0.5
    tier3_weight: float = 0.0  # Monitor only, don't backprop
    
    # Physics loss warmup
    physics_warmup_epochs: int = 10
    physics_ramp_epochs: int = 30

    # Curriculum training schedule
    phase1_epochs: int = 50
    phase2_epochs: int = 50
    phase3_epochs: int = 50
    phase2_head_lr_scale: float = 0.1
    phase3_head_lr_scale: float = 0.1
    
    # Physical constants
    water_density: float = 997.0  # kg/m³
    water_specific_heat: float = 4186.0  # J/(kg·K)
    gpm_to_m3_s: float = 6.30902e-5
    
    def __post_init__(self):
        super().__post_init__()
        assert self.physics_loss_weight >= 0, "physics_loss_weight must be non-negative"
        assert self.physics_warmup_epochs >= 0, "physics_warmup_epochs must be non-negative"


# Convenience aliases
Phase1Config = LSTMConfig
Phase2Config = DeepONetConfig
Phase3Config = HybridDeepONetConfig
Phase4Config = DomainDeepONetConfig
Phase5Config = FederatedConfig
Phase6Config = PhysicsInformedConfig


def create_domain_configs(base_config: Optional[SurrogateConfig] = None) -> Dict[str, DomainDeepONetConfig]:
    """
    Create configurations for all domains.
    
    Parameters
    ----------
    base_config : SurrogateConfig, optional
        Base configuration to inherit system settings from
        
    Returns
    -------
    Dict[str, DomainDeepONetConfig]
        Dictionary mapping domain names to their configs
    """
    configs = {}
    for domain in DOMAIN_OUTPUTS.keys():
        if base_config is not None:
            configs[domain] = DomainDeepONetConfig.for_domain(
                domain,
                system_name=base_config.system_name,
                num_cdus=base_config.num_cdus,
                cdu_ids=base_config.cdu_ids,
                history_steps=base_config.history_steps,
                prediction_steps=base_config.prediction_steps,
                subsample_factor=base_config.subsample_factor,
                train_ratio=base_config.train_ratio,
                val_ratio=base_config.val_ratio,
                batch_size=base_config.batch_size,
                input_patterns=base_config.input_patterns,
                output_patterns=base_config.output_patterns,
            )
        else:
            configs[domain] = DomainDeepONetConfig.for_domain(domain)
    return configs


__all__ = [
    'OutputType',
    'SystemName',
    'DomainType',
    'SurrogateConfig',
    'LSTMConfig',
    'DeepONetConfig',
    'HybridDeepONetConfig',
    'DomainDeepONetConfig',
    'FederatedConfig',
    'PhysicsInformedConfig',
    'Phase1Config',
    'Phase2Config',
    'Phase3Config',
    'Phase4Config',
    'Phase5Config',
    'Phase6Config',
    'create_domain_configs',
    'SYSTEM_DEFAULTS',
    'DEFAULT_INPUT_PATTERNS',
    'DEFAULT_OUTPUT_PATTERNS',
    'OUTPUT_CLASSIFICATION',
    'DOMAIN_OUTPUTS',
    'FEDERATED_HEAD_OUTPUTS',
    'HEAD_TYPES',
]