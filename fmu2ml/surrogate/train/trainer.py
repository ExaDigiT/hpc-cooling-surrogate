"""
fmu2ml.surrogate.train.trainer - Unified trainer for surrogate models.

Provides a phase-aware trainer that handles all six surrogate model phases
with support for:
- Automatic phase detection and data format handling
- SLURM/NCCL-aware distributed training (DDP)
- Automatic Mixed Precision (AMP)
- Gradient clipping and learning rate scheduling
- Weights & Biases logging
- Checkpoint save/load with normalizer stats
"""

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from .callbacks import Callback, CallbackList, EarlyStopping, ModelCheckpoint
from .loss import (
    SurrogateLoss,
    PhysicsConstraintLoss,
    PhysicsConstraintConfig,
    create_loss,
)

__all__ = [
    "SurrogateTrainer",
    "TrainingHistory",
]


# ───────────────────────────────────────────────────────────────────────
# Training History
# ───────────────────────────────────────────────────────────────────────


@dataclass
class TrainingHistory:
    """Container for training history and metrics."""

    train_losses: Dict[str, List[float]] = field(default_factory=dict)
    val_losses: Dict[str, List[float]] = field(default_factory=dict)
    learning_rates: List[float] = field(default_factory=list)
    epochs_completed: int = 0
    best_val_loss: float = float("inf")
    best_epoch: int = 0
    training_time: float = 0.0
    phase_histories: Dict[str, "TrainingHistory"] = field(default_factory=dict)

    def add_epoch(
        self,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        lr: float,
    ) -> None:
        """Add an epoch's results to history."""
        for key, value in train_losses.items():
            if f"train_{key}" not in self.train_losses:
                self.train_losses[f"train_{key}"] = []
            self.train_losses[f"train_{key}"].append(value)

        for key, value in val_losses.items():
            if f"val_{key}" not in self.val_losses:
                self.val_losses[f"val_{key}"] = []
            self.val_losses[f"val_{key}"].append(value)

        self.learning_rates.append(lr)
        self.epochs_completed += 1

        # Track best
        total_val = val_losses.get("total", val_losses.get("combined_total", float("inf")))
        if total_val < self.best_val_loss:
            self.best_val_loss = total_val
            self.best_epoch = self.epochs_completed

    def absorb(self, other: "TrainingHistory") -> None:
        """Append another history's epochs onto this one.

        The multi-phase protocols (phase 5's two-phase, phase 6's three-phase
        curriculum) each build a LOCAL TrainingHistory per sub-phase and stored
        it only under `phase_histories`, never touching the top-level object.
        The run therefore reported `epochs_completed = 0` and
        `best_val_loss = inf`, and `training_curves.png` could not be produced
        at all -- while training itself was fine, since each sub-phase runs its
        own EarlyStopping and calls `load_best`. Replaying the sub-phase epochs
        here makes the top-level history mean what phases 1-4's means: the
        concatenated training run.
        """
        for i in range(other.epochs_completed):
            tr = {k[len("train_"):]: v[i]
                  for k, v in other.train_losses.items() if i < len(v)}
            va = {k[len("val_"):]: v[i]
                  for k, v in other.val_losses.items() if i < len(v)}
            lr = (other.learning_rates[i]
                  if i < len(other.learning_rates) else 0.0)
            self.add_epoch(tr, va, lr)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "learning_rates": self.learning_rates,
            "epochs_completed": self.epochs_completed,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "training_time": self.training_time,
        }


# ───────────────────────────────────────────────────────────────────────
# Surrogate Trainer
# ───────────────────────────────────────────────────────────────────────


class SurrogateTrainer:
    """
    Phase-aware trainer for surrogate models.

    Supports all six phases with automatic detection and appropriate
    data handling, loss functions, and training protocols.

    Features:
    - Auto-detection of phase based on model type
    - Phase-specific training protocols (e.g., two-phase training for federated)
    - Distributed training support (DDP)
    - Automatic Mixed Precision (AMP)
    - Gradient clipping and cosine LR scheduling
    - W&B logging with offline fallback
    - Checkpoint management
    """

    # Phase detection mapping
    PHASE_MODEL_TYPES = {
        "BaselineLSTM": "lstm",
        "MultiRateLSTM": "lstm",
        "BasicDeepONet": "deeponet",
        "MultiRateDeepONet": "deeponet",
        "HybridDeepONet": "hybrid_deeponet",
        "MultiRateHybridDeepONet": "hybrid_deeponet",
        "DomainDeepONet": "domain_deeponet",
        "MultiRateDomainDeepONet": "domain_deeponet",
        "FederatedDeepMMNet": "federated",
        "PhysicsInformedFederatedDeepMMNet": "federated_pi",
    }

    # Head mappings for federated phases
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
        model: nn.Module,
        config: Any,
        system: str = "summit",
        device: Optional[torch.device] = None,
        use_amp: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "fmu2ml-surrogate",
        wandb_run_name: Optional[str] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        pred_to_physics: Optional[Any] = None,
        physics_loss_calc: Optional[PhysicsConstraintLoss] = None,
    ):
        """
        Initialize SurrogateTrainer.

        Args:
            model: Surrogate model to train
            config: Training configuration (phase-specific config dataclass)
            system: HPC system name ('summit', 'lassen', 'marconi100')
            device: Target device (auto-detected if None)
            use_amp: Whether to use Automatic Mixed Precision
            use_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
            wandb_run_name: W&B run name
            checkpoint_dir: Directory for checkpoints
            pred_to_physics: PredictionToPhysics converter (Phase 6)
            physics_loss_calc: Physics loss calculator (Phase 6)
        """
        self.model = model
        self.config = config
        self.system = system
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_wandb = use_wandb and HAS_WANDB

        # Phase detection
        self.phase = self._detect_phase(model)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model = self.model.to(self.device)

        # AMP setup
        self.scaler = GradScaler() if self.use_amp else None

        # Loss function
        physics_config = (
            PhysicsConstraintConfig() if self.phase == "federated_pi" else None
        )
        self.criterion = create_loss(self.phase, config, physics_config)

        # Physics components (Phase 6)
        self.pred_to_physics = pred_to_physics
        self.physics_loss_calc = physics_loss_calc

        # Optimizer and scheduler (initialized in fit())
        self.optimizer = None
        self.scheduler = None

        # Per-head persistence-baseline loss scales, measured once at the start
        # of fit() when config.head_loss_normalize is set. Empty means "no
        # normalisation", which is the default and reproduces old behaviour.
        self._head_scales: Dict[str, float] = {}

        # Checkpoint directory
        if checkpoint_dir is not None:
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None

        # W&B setup
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=vars(config) if hasattr(config, "__dict__") else config,
            )

        # Training state
        self.current_epoch = 0
        self.history = TrainingHistory()

        # Distributed training state
        self.is_distributed = False
        self.local_rank = 0
        self.world_size = 1

    def _detect_phase(self, model: nn.Module) -> str:
        """Detect model phase from class name."""
        class_name = model.__class__.__name__
        return self.PHASE_MODEL_TYPES.get(class_name, "lstm")

    def _cfg(self, *keys: str, default: Any = None) -> Any:
        """Read the first matching config attribute from multiple key styles."""
        for key in keys:
            if hasattr(self.config, key):
                return getattr(self.config, key)
        return default

    def _log_epoch(
        self,
        epoch: int,
        max_epochs: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        lr: float,
        tr_s: float,
        va_s: float,
        train_loader: DataLoader,
        start_time: float,
    ) -> None:
        """Print a two-line per-epoch report (rank 0 only).

        tqdm's set_postfix only updates a live terminal line, so under `tee` or
        an sbatch log there is no per-epoch record at all — the run looks silent
        until early stopping fires. These are plain prints for that reason.
        """
        if self.local_rank != 0:
            return

        ep_s = tr_s + va_s
        n_steps = len(train_loader)
        bs = getattr(train_loader, "batch_size", None) or 0
        # Global throughput: every rank ran n_steps of its own shard.
        sps = (n_steps * bs * max(self.world_size, 1) / tr_s) if tr_s > 0 else 0.0
        gpu = (torch.cuda.max_memory_allocated() / 2 ** 30
               if torch.cuda.is_available() else 0.0)

        tr_tot = train_losses.get("total", float("nan"))
        va_tot = val_losses.get("total", float("nan"))
        # val:train ratio — ~1 means the model is not separating train from val
        # (underfitting; more capacity may help), >>1 means overfitting.
        vt = (va_tot / tr_tot) if tr_tot else float("nan")
        eta = ep_s * (max_epochs - (epoch + 1)) / 60.0

        heads = " ".join(f"{k}={v:.4f}"
                         for k, v in sorted(val_losses.items()) if k != "total")
        print(f"ep {epoch + 1:3d}/{max_epochs}  train {tr_tot:.5f}  "
              f"val {va_tot:.5f}  best@{self.history.best_epoch}  {heads}",
              flush=True)
        print(f"        time {ep_s:6.1f}s (tr {tr_s:5.1f} va {va_s:5.1f})  "
              f"{sps:8,.0f} samp/s  {n_steps:4d} steps  lr {lr:.2e}  "
              f"gpu {gpu:5.2f}GiB  v/t {vt:5.2f}  "
              f"elapsed {(time.time() - start_time) / 60:5.1f}m  "
              f"ETA {eta:5.0f}m", flush=True)

    @property
    def _core(self) -> nn.Module:
        """The underlying model, seen through the DistributedDataParallel wrap.

        DDP does not forward plain-attribute lookups to the module it wraps, so
        ``self.model.multi_rate`` is False and ``self.model.branches`` raises
        once distributed training is on — even though the wrapped model has
        both. Read model attributes through this property.

        For ATTRIBUTE ACCESS ONLY. Never call the forward pass through it: DDP
        registers its gradient-sync hooks on ``__call__``, so invoking
        ``self._core.forward_multirate(...)`` would compute correct-looking
        losses while skipping the all-reduce entirely, and each rank would
        silently train its own diverging copy of the model.
        """
        return getattr(self.model, "module", self.model)

    def _setup_distributed(self) -> None:
        """Setup distributed training if SLURM environment is detected."""
        if "SLURM_PROCID" in os.environ:
            self.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
            self.world_size = int(os.environ.get("SLURM_NTASKS", 1))

            if self.world_size > 1:
                import torch.distributed as dist

                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                self.is_distributed = True

                # find_unused_parameters costs an extra autograd-graph traversal
                # every iteration, so enable it only where it is actually
                # needed: the federated phases train head subsets via
                # `active_heads` (round-robin), leaving parameters out of the
                # graph. Phases 1-4 compute every head every batch, and turning
                # it on there just buys the overhead plus a warning.
                # Override with FMU2ML_DDP_FIND_UNUSED=1/0.
                _needs_unused = self.phase in ("federated", "federated_pi")
                _env = os.environ.get("FMU2ML_DDP_FIND_UNUSED")
                if _env is not None:
                    _needs_unused = _env not in ("0", "false", "False", "")

                self.model = nn.parallel.DistributedDataParallel(
                    self.model,
                    device_ids=[self.local_rank],
                    broadcast_buffers=False,
                    find_unused_parameters=_needs_unused,
                )
                print(f"Distributed training: rank {self.local_rank}/{self.world_size}")

    # ── per-head loss weighting ────────────────────────────────────────────────

    def _raw_head_loss(
        self, pred: torch.Tensor, target: torch.Tensor,
    ) -> torch.Tensor:
        """The unweighted per-head loss, matching whichever path is training.

        Phases 1-4 go through `_forward_batch_multirate`, which calls
        `huber_loss` directly; phases 5-6 go through the federated paths, which
        call `self.criterion._loss`. Calibration has to use the SAME function
        the training path uses, or the measured baseline is in the wrong units
        and the normalisation silently rescales the objective.
        """
        if self.phase in ("federated", "federated_pi"):
            return self.criterion._loss(pred, target)
        delta = self._cfg("huber_delta", "HUBER_DELTA", default=0.5)
        return nn.functional.huber_loss(
            pred.float(), target.float(), delta=delta)

    def _history_noise_std(self) -> float:
        """Current std of the training-time `y_hist` perturbation (0 if off).

        Ramped linearly over `history_noise_warmup_epochs` so the model first
        learns the clean mapping and only then is asked to tolerate a degraded
        history; starting at full noise slows early convergence for no gain.
        """
        std = float(self._cfg("history_noise_std", "HISTORY_NOISE_STD", default=0.0))
        if std <= 0.0 or not self.model.training:
            return 0.0
        warm = int(self._cfg("history_noise_warmup_epochs",
                             "HISTORY_NOISE_WARMUP_EPOCHS", default=0))
        if warm > 0:
            frac = min(1.0, (getattr(self, "current_epoch", 0) + 1) / warm)
            std *= frac
        return std

    def _branch_inputs(
        self, batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Assemble `{branch: (u_hist, y_hist)}`, perturbing y_hist in training.

        Only `y_hist` is perturbed: `u_hist` carries the exogenous inputs (job
        schedule, weather), which stay known at deployment, so corrupting them
        would model a different problem than the one being fixed.
        """
        dev = self.device
        std = self._history_noise_std()
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for b in self._core.branches:
            u = batch[f"u_hist__{b}"].to(dev, non_blocking=True)
            y = batch[f"y_hist__{b}"].to(dev, non_blocking=True)
            if std > 0.0:
                y = y + torch.randn_like(y) * std
            out[b] = (u, y)
        return out

    def _head_weight(self, head: str) -> float:
        """Effective loss weight for `head`: configured weight / baseline scale.

        With `head_loss_normalize` off this is just the configured weight, so
        the default path is bit-identical to the previous behaviour.
        """
        w = (getattr(self.config, "head_loss_weights", {}) or {}).get(head, 1.0)
        return w / self._head_scales.get(head, 1.0)

    @torch.no_grad()
    def _calibrate_head_scales(self, loader: DataLoader) -> None:
        """Measure each head's persistence-baseline loss, once, before training.

        Targets are consecutive deltas, so the persistence forecast ("nothing
        changes") is exactly a zero delta. The baseline is therefore
        ``loss(0, target)`` — a property of the data alone, with no model in the
        loop, which is why it can be measured up front and then frozen.

        Under DDP every rank MUST end up with identical scales or the ranks
        optimise subtly different objectives and the all-reduced gradient is
        meaningless; the values are all-reduced to the mean to guarantee that.
        """
        if not getattr(self.config, "head_loss_normalize", False):
            return

        max_batches = int(self._cfg(
            "head_loss_calib_batches", "HEAD_LOSS_CALIB_BATCHES", default=50))
        sums: Dict[str, float] = {h: 0.0 for h in self.ALL_HEAD_NAMES}
        counts: Dict[str, int] = {h: 0 for h in self.ALL_HEAD_NAMES}

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            for h in self.ALL_HEAD_NAMES:
                key = self.HEAD_TARGET_MAP[h]
                if key not in batch:
                    continue
                target = batch[key].to(self.device, non_blocking=True).float()
                base = self._raw_head_loss(torch.zeros_like(target), target)
                sums[h] += float(base.item())
                counts[h] += 1

        scales: Dict[str, float] = {}
        for h in self.ALL_HEAD_NAMES:
            if counts[h] == 0:
                continue
            s = sums[h] / counts[h]
            # A head whose baseline is ~0 is already perfectly predicted by
            # persistence; dividing by it would blow the weight up to infinity
            # and let numerical noise dominate the objective.
            scales[h] = s if s > 1e-12 else 1.0

        if self.is_distributed and scales:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                keys = sorted(scales)
                t = torch.tensor([scales[k] for k in keys],
                                 device=self.device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t /= dist.get_world_size()
                scales = {k: float(v) for k, v in zip(keys, t.tolist())}

        self._head_scales = scales

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("PER-HEAD LOSS NORMALISATION (persistence baseline)")
            print(f"{'='*70}")
            print(f"  {'head':6s} {'baseline':>12s} {'cfg_w':>8s} {'effective':>12s}")
            cfg_w = getattr(self.config, "head_loss_weights", {}) or {}
            for h in self.ALL_HEAD_NAMES:
                if h not in scales:
                    continue
                print(f"  {h:6s} {scales[h]:12.6e} {cfg_w.get(h, 1.0):8.2f} "
                      f"{self._head_weight(h):12.4e}")
            print(flush=True)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: Optional[int] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingHistory:
        """
        Train the model.

        Automatically selects the appropriate training protocol based on phase:
        - Phase 1-4: Standard single-phase training
        - Phase 5: Two-phase training (round-robin + fine-tuning)
        - Phase 6: Three-phase curriculum (data-only + physics ramp + full physics)

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            max_epochs: Maximum epochs (overrides config)
            callbacks: List of callbacks

        Returns:
            TrainingHistory with all metrics
        """
        # Setup distributed if available
        self._setup_distributed()

        # Measure the per-head persistence baselines before any training, so
        # every phase's objective is on one scale. No-op unless the config sets
        # head_loss_normalize. Must follow _setup_distributed(): the scales are
        # all-reduced so all ranks share one objective.
        self._calibrate_head_scales(train_loader)

        # Get max epochs from config or argument
        if max_epochs is None:
            max_epochs = getattr(self.config, "max_epochs", getattr(self.config, "MAX_EPOCHS", 100))

        # Route to appropriate training method
        if self.phase == "federated":
            return self._fit_federated(train_loader, val_loader, callbacks)
        elif self.phase == "federated_pi":
            return self._fit_physics_informed(train_loader, val_loader, callbacks)
        else:
            return self._fit_standard(train_loader, val_loader, max_epochs, callbacks)

    def _fit_standard(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingHistory:
        """Standard training loop for Phase 1-4."""
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=getattr(self.config, "learning_rate", getattr(self.config, "LEARNING_RATE", 1e-3)),
            weight_decay=getattr(self.config, "weight_decay", getattr(self.config, "WEIGHT_DECAY", 1e-5)),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        # Setup callbacks
        callback_list = CallbackList(callbacks or [])

        # Add default callbacks if none provided
        if not callbacks:
            callback_list.append(
                EarlyStopping(
                    patience=getattr(self.config, "patience", getattr(self.config, "PATIENCE", 20)),
                    verbose=self.local_rank == 0,
                )
            )
            if self.checkpoint_dir:
                callback_list.append(
                    ModelCheckpoint(
                        dirpath=self.checkpoint_dir,
                        save_best_only=True,
                        verbose=self.local_rank == 0,
                    )
                )

        callback_list.on_train_begin(self)

        # Training loop
        start_time = time.time()

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print(f"TRAINING — Phase: {self.phase.upper()}")
            print(f"{'='*70}")

        epoch_iter = range(max_epochs)
        # Only draw the bar on a real terminal: redirected to a file it emits a
        # carriage-return smear and its postfix metrics are lost entirely.
        if HAS_TQDM and self.local_rank == 0 and sys.stdout.isatty():
            epoch_iter = tqdm(epoch_iter, desc="Training")

        if self.local_rank == 0:
            print(f"\ncolumns: ep | train/val total loss | best epoch | per-head val\n"
                  f"         time line: wall (train/val) | throughput | steps | lr | "
                  f"peak GPU | v/t ratio | elapsed | ETA", flush=True)

        for epoch in epoch_iter:
            self.current_epoch = epoch
            callback_list.on_epoch_begin(epoch, self)

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()   # per-epoch peak, not run

            # Train and validate
            _t = time.time()
            train_losses = self._train_epoch(train_loader)
            tr_s = time.time() - _t
            _t = time.time()
            val_losses = self._validate_epoch(val_loader)
            va_s = time.time() - _t

            self.scheduler.step()

            # Record history
            lr = self.optimizer.param_groups[0]["lr"]
            self.history.add_epoch(train_losses, val_losses, lr)

            # Update progress bar
            if HAS_TQDM and self.local_rank == 0 and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    {
                        "train": f"{train_losses.get('total', 0):.4f}",
                        "val": f"{val_losses.get('total', 0):.4f}",
                        "lr": f"{lr:.2e}",
                    }
                )

            self._log_epoch(epoch, max_epochs, train_losses, val_losses, lr,
                            tr_s, va_s, train_loader, start_time)

            # Log to W&B
            if self.use_wandb and self.local_rank == 0:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": train_losses.get("total", 0),
                        "val_loss": val_losses.get("total", 0),
                        "learning_rate": lr,
                        **{f"train_{k}": v for k, v in train_losses.items()},
                        **{f"val_{k}": v for k, v in val_losses.items()},
                    }
                )

            # Callbacks
            if callback_list.on_epoch_end(epoch, train_losses, val_losses, self):
                if self.local_rank == 0:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        self.history.training_time = time.time() - start_time
        callback_list.on_train_end(self)

        if self.local_rank == 0:
            t = self.history.training_time
            print(f"\nTraining complete.")
            print(f"  Epochs trained:       {self.history.epochs_completed}")
            print(f"  Best validation loss: {self.history.best_val_loss:.6f}")
            print(f"  Training time:        {t:.1f}s ({t/60:.1f} min)")

        return self.history

    def _fit_federated(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingHistory:
        """
        Two-phase training for Federated DeepMMNet (Phase 5).

        Phase 1: Round-robin head training
        Phase 2: Joint fine-tuning with differential learning rates
        """
        start_time = time.time()

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("TRAINING — Phase 5: Federated DeepMMNet")
            print(f"{'='*70}")

        # Phase 1: Round-Robin
        phase1_history = self._run_federated_phase1(train_loader, val_loader)
        self.history.phase_histories["phase1"] = phase1_history
        self.history.absorb(phase1_history)
        # after phase 1 (early stopping has already restored phase-1 best into the model):
        best1_val = min(phase1_history.val_losses.get("val_total", [float("inf")]))
        best1_state = copy.deepcopy(self.model.state_dict())


        # Phase 2: Joint Fine-Tuning
        phase2_history = self._run_federated_phase2(train_loader, val_loader)
        self.history.phase_histories["phase2"] = phase2_history
        self.history.absorb(phase2_history)

        # after phase 2:
        best2_val = min(phase2_history.val_losses.get("val_total", [float("inf")]))
        if best2_val > best1_val:
            self.model.load_state_dict(best1_state)
            if self.local_rank == 0:
                print(f"Phase 2 ({best2_val:.6f}) worse than Phase 1 ({best1_val:.6f}) — "
                    f"restored Phase 1 weights.")
                
        self.history.training_time = time.time() - start_time

        if self.local_rank == 0:
            print(f"\nTotal training time: {self.history.training_time:.1f}s")

        return self.history

    def _run_federated_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Phase 1: Round-robin training for all 6 heads."""
        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("PHASE 1: Round-Robin Training (6 Decoder Heads)")
            print(f"{'='*70}")

        phase1_epochs = self._cfg("phase1_epochs", "PHASE1_EPOCHS", default=50)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._cfg("learning_rate", "LEARNING_RATE", default=1e-3),
            weight_decay=self._cfg("weight_decay", "WEIGHT_DECAY", default=1e-5),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2)

        early_stopping = EarlyStopping(
            patience=self._cfg("patience", "PATIENCE", default=20),
            verbose=self.local_rank == 0,
        )

        history = TrainingHistory()

        epoch_iter = range(phase1_epochs)
        if HAS_TQDM and self.local_rank == 0:
            epoch_iter = tqdm(epoch_iter, desc="Phase 1")

        for epoch in epoch_iter:
            train_losses = self._train_epoch_federated(train_loader)
            val_losses = self._validate_epoch_federated(val_loader)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            history.add_epoch(train_losses, val_losses, lr)

            if HAS_TQDM and self.local_rank == 0:
                epoch_iter.set_postfix(
                    {
                        "trn": f"{train_losses.get('total', 0):.4f}",
                        "val": f"{val_losses.get('total', 0):.4f}",
                        "T": f"{val_losses.get('G_T', 0):.4f}",
                    }
                )

            if early_stopping(val_losses.get("total", 0), self.model):
                if self.local_rank == 0:
                    print(f"\nPhase 1 early stopping at epoch {epoch + 1}")
                break

        early_stopping.load_best(self.model)

        if self.local_rank == 0:
            print(f"Phase 1 best validation loss: {early_stopping.best_value:.6f}")

        return history

    def _run_federated_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Phase 2: Joint fine-tuning with differential learning rates."""
        lr = self._cfg("learning_rate", "LEARNING_RATE", default=1e-3)
        head_lr_scale = self._cfg("phase2_head_lr_scale", "PHASE2_HEAD_LR_SCALE", default=0.1)

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("PHASE 2: Joint Fine-Tuning (6 Heads)")
            print(f"  Encoder LR: {lr}")
            print(f"  Head LR:    {lr * head_lr_scale}")
            print(f"{'='*70}")

        phase2_epochs = self._cfg("phase2_epochs", "PHASE2_EPOCHS", default=100)

        # Differential learning rates
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self._core.get_encoder_params(), "lr": lr},
                {"params": self._core.get_all_head_params(), "lr": lr * head_lr_scale},
            ],
            weight_decay=self._cfg("weight_decay", "WEIGHT_DECAY", default=1e-5),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        early_stopping = EarlyStopping(
            patience=self._cfg("patience", "PATIENCE", default=20),
            verbose=self.local_rank == 0,
        )

        history = TrainingHistory()

        epoch_iter = range(phase2_epochs)
        if HAS_TQDM and self.local_rank == 0:
            epoch_iter = tqdm(epoch_iter, desc="Phase 2")

        for epoch in epoch_iter:
            train_losses = self._train_epoch_federated(train_loader)
            val_losses = self._validate_epoch_federated(val_loader)
            self.scheduler.step()

            lr_enc = self.optimizer.param_groups[0]["lr"]
            history.add_epoch(train_losses, val_losses, lr_enc)

            if HAS_TQDM and self.local_rank == 0:
                epoch_iter.set_postfix(
                    {
                        "trn": f"{train_losses.get('total', 0):.4f}",
                        "val": f"{val_losses.get('total', 0):.4f}",
                        "T": f"{val_losses.get('G_T', 0):.4f}",
                    }
                )

            if early_stopping(val_losses.get("total", 0), self.model):
                if self.local_rank == 0:
                    print(f"\nPhase 2 early stopping at epoch {epoch + 1}")
                break

        early_stopping.load_best(self.model)

        if self.local_rank == 0:
            print(f"Phase 2 best validation loss: {early_stopping.best_value:.6f}")

        return history

    def _fit_physics_informed(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: Optional[List[Callback]] = None,
    ) -> TrainingHistory:
        """
        Three-phase curriculum training for Physics-Informed model (Phase 6).

        Phase 1: Data-only training (λ_physics = 0)
        Phase 2: Physics ramp (λ_physics: 0 → max)
        Phase 3: Full physics fine-tuning
        """
        start_time = time.time()

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("TRAINING — Phase 6: Physics-Informed Federated DeepMMNet")
            print(f"{'='*70}")

        # Phase 1: Data-Only
        phase1_history = self._run_physics_phase1(train_loader, val_loader)
        self.history.phase_histories["phase1"] = phase1_history
        self.history.absorb(phase1_history)

        # Phase 2: Physics Ramp
        phase2_history = self._run_physics_phase2(train_loader, val_loader)
        self.history.phase_histories["phase2"] = phase2_history
        self.history.absorb(phase2_history)

        # Phase 3: Full Physics
        phase3_history = self._run_physics_phase3(train_loader, val_loader)
        self.history.phase_histories["phase3"] = phase3_history
        self.history.absorb(phase3_history)

        self.history.training_time = time.time() - start_time

        if self.local_rank == 0:
            print(f"\nTotal training time: {self.history.training_time:.1f}s")

        return self.history

    def _run_physics_phase1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Phase 1: Data-only training."""
        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("PHASE 1: Data-Only Round-Robin Training (λ_physics = 0.0)")
            print(f"{'='*70}")

        phase1_epochs = self._cfg("phase1_epochs", "PHASE1_EPOCHS", default=50)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self._cfg("learning_rate", "LEARNING_RATE", default=1e-3),
            weight_decay=self._cfg("weight_decay", "WEIGHT_DECAY", default=1e-5),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2)

        early_stopping = EarlyStopping(
            patience=self._cfg("patience", "PATIENCE", default=20),
            verbose=self.local_rank == 0,
        )

        history = TrainingHistory()

        epoch_iter = range(phase1_epochs)
        if HAS_TQDM and self.local_rank == 0:
            epoch_iter = tqdm(epoch_iter, desc="Phase 1")

        for epoch in epoch_iter:
            train_losses = self._train_epoch_physics(train_loader, physics_weight=0.0)
            val_losses = self._validate_epoch_physics(val_loader, physics_weight=0.0)
            self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            history.add_epoch(train_losses, val_losses, lr)

            if HAS_TQDM and self.local_rank == 0:
                epoch_iter.set_postfix(
                    {
                        "trn": f"{train_losses.get('data_total', 0):.4f}",
                        "val": f"{val_losses.get('data_total', 0):.4f}",
                    }
                )

            if early_stopping(val_losses.get("data_total", 0), self.model):
                if self.local_rank == 0:
                    print(f"\nPhase 1 early stopping at epoch {epoch + 1}")
                break

        early_stopping.load_best(self.model)
        return history

    def _run_physics_phase2(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Phase 2: Introduce physics gradually."""
        physics_max = self._cfg("physics_weight_max", "PHYSICS_WEIGHT_MAX", default=0.1)
        ramp_epochs = self._cfg("physics_ramp_epochs", "PHYSICS_RAMP_EPOCHS", default=30)
        lr = self._cfg("learning_rate", "LEARNING_RATE", default=1e-3)
        head_lr_scale = self._cfg("phase2_head_lr_scale", "PHASE2_HEAD_LR_SCALE", default=0.1)

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("PHASE 2: Introduce Physics Losses (Curriculum Ramp)")
            print(f"  λ_physics: 0.0 → {physics_max} over {ramp_epochs} epochs")
            print(f"{'='*70}")

        phase2_epochs = self._cfg("phase2_epochs", "PHASE2_EPOCHS", default=50)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self._core.get_encoder_params(), "lr": lr},
                {"params": self._core.get_all_head_params(), "lr": lr * head_lr_scale},
            ],
            weight_decay=self._cfg("weight_decay", "WEIGHT_DECAY", default=1e-5),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        early_stopping = EarlyStopping(
            patience=self._cfg("patience", "PATIENCE", default=20),
            verbose=self.local_rank == 0,
        )

        history = TrainingHistory()

        epoch_iter = range(phase2_epochs)
        if HAS_TQDM and self.local_rank == 0:
            epoch_iter = tqdm(epoch_iter, desc="Phase 2")

        for epoch in epoch_iter:
            # Compute ramped physics weight
            ramp = min(epoch / max(ramp_epochs, 1), 1.0)
            physics_weight = physics_max * ramp

            train_losses = self._train_epoch_physics(train_loader, physics_weight)
            val_losses = self._validate_epoch_physics(val_loader, physics_weight)
            self.scheduler.step()

            lr_enc = self.optimizer.param_groups[0]["lr"]
            history.add_epoch(train_losses, val_losses, lr_enc)

            if HAS_TQDM and self.local_rank == 0:
                epoch_iter.set_postfix(
                    {
                        "data": f"{val_losses.get('data_total', 0):.4f}",
                        "phys": f"{val_losses.get('physics_total', 0):.4f}",
                        "λ": f"{physics_weight:.4f}",
                    }
                )

            if early_stopping(val_losses.get("combined_total", 0), self.model):
                if self.local_rank == 0:
                    print(f"\nPhase 2 early stopping at epoch {epoch + 1}")
                break

        early_stopping.load_best(self.model)
        return history

    def _run_physics_phase3(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> TrainingHistory:
        """Phase 3: Joint fine-tuning with full physics."""
        physics_weight = self._cfg("physics_weight_max", "PHYSICS_WEIGHT_MAX", default=0.1)
        lr = self._cfg("learning_rate", "LEARNING_RATE", default=1e-3)
        head_lr_scale = self._cfg("phase3_head_lr_scale", "PHASE3_HEAD_LR_SCALE", default=0.1)

        if self.local_rank == 0:
            print(f"\n{'='*70}")
            print("PHASE 3: Joint Fine-Tuning with Full Physics Weight")
            print(f"  λ_physics: {physics_weight} (constant)")
            print(f"{'='*70}")

        phase3_epochs = self._cfg("phase3_epochs", "PHASE3_EPOCHS", default=50)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": self._core.get_encoder_params(), "lr": lr},
                {"params": self._core.get_all_head_params(), "lr": lr * head_lr_scale},
            ],
            weight_decay=self._cfg("weight_decay", "WEIGHT_DECAY", default=1e-5),
        )
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2)

        early_stopping = EarlyStopping(
            patience=self._cfg("patience", "PATIENCE", default=20),
            verbose=self.local_rank == 0,
        )

        history = TrainingHistory()

        epoch_iter = range(phase3_epochs)
        if HAS_TQDM and self.local_rank == 0:
            epoch_iter = tqdm(epoch_iter, desc="Phase 3")

        for epoch in epoch_iter:
            train_losses = self._train_epoch_physics(train_loader, physics_weight)
            val_losses = self._validate_epoch_physics(val_loader, physics_weight)
            self.scheduler.step()

            lr_enc = self.optimizer.param_groups[0]["lr"]
            history.add_epoch(train_losses, val_losses, lr_enc)

            if HAS_TQDM and self.local_rank == 0:
                epoch_iter.set_postfix(
                    {
                        "data": f"{val_losses.get('data_total', 0):.4f}",
                        "phys": f"{val_losses.get('physics_total', 0):.4f}",
                        "comb": f"{val_losses.get('combined_total', 0):.4f}",
                    }
                )

            if early_stopping(val_losses.get("combined_total", 0), self.model):
                if self.local_rank == 0:
                    print(f"\nPhase 3 early stopping at epoch {epoch + 1}")
                break

        early_stopping.load_best(self.model)
        return history

    # =========================================================================
    # Training Epoch Methods
    # =========================================================================

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Single training epoch for Phase 1-4."""
        self.model.train()
        total_losses: Dict[str, float] = {}
        n_batches = 0
        gradient_clip = self._cfg("gradient_clip", "GRADIENT_CLIP", default=1.0)

        for batch in loader:
            self.optimizer.zero_grad()

            # Forward pass (phase-specific)
            with autocast(enabled=self.use_amp):
                loss, loss_dict = self._forward_batch(batch)

            if torch.isnan(loss):
                continue

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] = total_losses.get(key, 0.0) + value
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def _validate_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Validation epoch for Phase 1-4."""
        self.model.eval()
        total_losses: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            with autocast(enabled=self.use_amp):
                loss, loss_dict = self._forward_batch(batch)

            for key, value in loss_dict.items():
                total_losses[key] = total_losses.get(key, 0.0) + value
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in total_losses.items()}

    def _forward_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Forward pass for a single batch (Phase 1-4)."""
        # Multi-rate mode is phase-agnostic: any phase 1-4 model exposing the
        # forward_multirate contract trains on per-group targets.
        if getattr(self._core, "multi_rate", False) and hasattr(
                self._core, "forward_multirate"):
            return self._forward_batch_multirate(batch)

        if self.phase == "lstm":
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            pred = self.model(x)
            return self.criterion(pred, y)

        elif self.phase in ("deeponet", "hybrid_deeponet"):
            x_temporal = batch["x_temporal"].to(self.device, non_blocking=True)
            x_algebraic = batch["x_algebraic"].to(self.device, non_blocking=True)
            y_temporal = batch["y_temporal"].to(self.device, non_blocking=True)
            y_algebraic = batch["y_algebraic"].to(self.device, non_blocking=True)

            if self.phase == "hybrid_deeponet":
                # Additional inputs for hybrid model
                last_temporal = batch["last_temporal_normalized"].to(self.device, non_blocking=True)
                temporal_mean = batch["temporal_mean"].to(self.device, non_blocking=True)
                temporal_cdu_idx = batch["temporal_cdu_indices"][0].to(self.device, non_blocking=True)
                algebraic_cdu_idx = batch["algebraic_cdu_indices"][0].to(self.device, non_blocking=True)

                output = self.model(
                    x_temporal, x_algebraic,
                    last_temporal, temporal_mean,
                    temporal_cdu_idx, algebraic_cdu_idx,
                )

                if isinstance(output, tuple):
                    if len(output) == 3:
                        pred_temporal, pred_algebraic, alpha = output
                    elif len(output) == 2:
                        pred_temporal, pred_algebraic = output
                        alpha = None
                    else:
                        raise ValueError(
                            f"Unexpected hybrid output tuple length: {len(output)}"
                        )
                elif isinstance(output, dict):
                    pred_temporal = output["temporal"]
                    pred_algebraic = output["algebraic"]
                    alpha = output.get("alpha")
                else:
                    raise ValueError(
                        f"Unexpected hybrid output type: {type(output)}"
                    )

                return self.criterion(pred_temporal, y_temporal, pred_algebraic, y_algebraic, alpha)
            else:
                pred_temporal, pred_algebraic = self.model(x_temporal, x_algebraic)
                return self.criterion(pred_temporal, y_temporal, pred_algebraic, y_algebraic)

        elif self.phase == "domain_deeponet":
            x = batch["x"].to(self.device, non_blocking=True)
            y = batch["y"].to(self.device, non_blocking=True)
            result = self.model(x)
            predictions = result["predictions"]
            return self.criterion(predictions, y)

        else:
            raise ValueError(f"Unknown phase for _forward_batch: {self.phase}")

    def _forward_batch_multirate(
        self, batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Phase-agnostic multi-rate forward + per-group loss (Phase 1-4).

        Consumes the chunk-store batch (per-branch ``u_hist__{b}``/``y_hist__{b}``,
        per-group ``y_delta_{suf}`` targets) exactly like the federated path, so
        every phase trains on the identical multi-rate task.
        """
        dev = self.device
        branch_inputs = self._branch_inputs(batch)
        # Called through self.model (the DDP wrapper when distributed) so the
        # backward pass all-reduces gradients; the model's forward() dispatches
        # a dict argument to forward_multirate.
        output = self.model(branch_inputs)

        delta = self._cfg("huber_delta", "HUBER_DELTA", default=0.5)
        total = torch.tensor(0.0, device=dev)
        raw_total = 0.0
        loss_dict: Dict[str, float] = {}
        for h in self.ALL_HEAD_NAMES:
            target_key = self.HEAD_TARGET_MAP[h]
            if target_key not in batch:
                continue
            target = batch[target_key].to(dev, non_blocking=True)
            pred = output[self.HEAD_PRED_MAP[h]]
            head_loss = nn.functional.huber_loss(
                pred.float(), target.float(), delta=delta)
            total = total + self._head_weight(h) * head_loss
            loss_dict[h] = head_loss.item()      # RAW per-head, for reporting
            raw_total += loss_dict[h]
        # "total" is the quantity actually optimised AND the one EarlyStopping
        # monitors, so weighting it is what makes early stopping track the
        # channels that can still improve. "raw_total" preserves the old
        # unweighted sum so runs stay comparable across the change.
        loss_dict["total"] = float(total.item())
        loss_dict["raw_total"] = raw_total
        return total, loss_dict

    def _federated_forward(self, batch):
        """
        Run the federated model for one batch, handling single- and multi-rate.

        Returns ``(output, phys_pred, phys_last)`` where ``output`` holds the
        per-head predictions (``pred_T``..), and ``phys_pred``/``phys_last`` are the
        prediction tensor + raw last-state used for the physics loss (Phase 6):
        ``predictions``/``last_dynamic`` (single-rate) or the co-located
        ``predictions_phys``/``last_dynamic_phys`` (multi-rate physics grid).
        """
        dev = self.device
        if getattr(self._core, "multi_rate", False):
            branch_inputs = self._branch_inputs(batch)
            want_phys = self.pred_to_physics is not None
            # Through self.model, not self._core: see the _core docstring.
            output = self.model(branch_inputs, physics=want_phys)
            phys_pred = output.get("predictions_phys")
            phys_last = (batch["last_dynamic_phys"].to(dev, non_blocking=True)
                         if "last_dynamic_phys" in batch else None)
        else:
            u_hist = batch["u_hist"].to(dev, non_blocking=True)
            y_hist = batch["y_hist"].to(dev, non_blocking=True)
            output = self.model(u_hist, y_hist)
            phys_pred = output.get("predictions")
            phys_last = (batch["last_dynamic"].to(dev, non_blocking=True)
                         if "last_dynamic" in batch else None)
        return output, phys_pred, phys_last

    def _train_epoch_federated(self, loader: DataLoader) -> Dict[str, float]:
        """Training epoch for Federated model (Phase 5)."""
        self.model.train()
        losses = {h: 0.0 for h in self.ALL_HEAD_NAMES}
        losses["total"] = 0.0
        losses["raw_total"] = 0.0
        n_batches = 0
        gradient_clip = self._cfg("gradient_clip", "GRADIENT_CLIP", default=1.0)

        for batch in loader:
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output, _, _ = self._federated_forward(batch)

                batch_loss = torch.tensor(0.0, device=self.device)
                raw_batch = 0.0

                for head_name in self.ALL_HEAD_NAMES:
                    target = batch[self.HEAD_TARGET_MAP[head_name]].to(self.device, non_blocking=True)
                    pred = output[self.HEAD_PRED_MAP[head_name]]
                    head_loss = self.criterion._loss(pred, target)
                    batch_loss = batch_loss + self._head_weight(head_name) * head_loss  # weighted total (tensor)
                    losses[head_name] += head_loss.item()                               # RAW logging (float)
                    raw_batch += head_loss.item()

            if torch.isnan(batch_loss):
                continue

            if self.use_amp:
                self.scaler.scale(batch_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()

            losses["total"] += batch_loss.item()
            losses["raw_total"] += raw_batch
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in losses.items()}

    @torch.no_grad()
    def _validate_epoch_federated(self, loader: DataLoader) -> Dict[str, float]:
        """Validation epoch for Federated model (Phase 5)."""
        self.model.eval()
        losses = {h: 0.0 for h in self.ALL_HEAD_NAMES}
        losses["total"] = 0.0
        losses["raw_total"] = 0.0
        n_batches = 0

        for batch in loader:
            with autocast(enabled=self.use_amp):
                output, _, _ = self._federated_forward(batch)

            batch_loss = 0.0
            raw_batch = 0.0
            for head_name in self.ALL_HEAD_NAMES:
                target = batch[self.HEAD_TARGET_MAP[head_name]].to(self.device, non_blocking=True)
                pred = output[self.HEAD_PRED_MAP[head_name]]
                head_loss = self.criterion._loss(pred, target).item()             # float
                losses[head_name] += head_loss                                    # RAW logging
                batch_loss += self._head_weight(head_name) * head_loss            # weighted total
                raw_batch += head_loss

            losses["total"] += batch_loss
            losses["raw_total"] += raw_batch
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in losses.items()}

    def _train_epoch_physics(
        self,
        loader: DataLoader,
        physics_weight: float,
    ) -> Dict[str, float]:
        """Training epoch with physics constraints (Phase 6)."""
        self.model.train()
        losses = {h: 0.0 for h in self.ALL_HEAD_NAMES}
        losses.update({"data_total": 0.0, "physics_total": 0.0,
                       "combined_total": 0.0, "raw_total": 0.0})
        physics_accumulator: Dict[str, float] = {}
        n_batches = 0
        gradient_clip = self._cfg("gradient_clip", "GRADIENT_CLIP", default=1.0)

        for batch in loader:
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                output, phys_pred, phys_last = self._federated_forward(batch)

                # Data loss. Applies the same per-head weighting as every other
                # path — this branch used to sum the heads unweighted, so a
                # phase-6 run's head_loss_weights were silently ignored.
                data_loss = torch.tensor(0.0, device=self.device)
                raw_batch = 0.0
                for head_name in self.ALL_HEAD_NAMES:
                    target = batch[self.HEAD_TARGET_MAP[head_name]].to(self.device, non_blocking=True)
                    pred = output[self.HEAD_PRED_MAP[head_name]]
                    head_loss = self.criterion._loss(pred, target)
                    data_loss = data_loss + self._head_weight(head_name) * head_loss
                    losses[head_name] += head_loss.item()
                    raw_batch += head_loss.item()

                # Physics loss (single-rate: predictions/last_dynamic; multi-rate:
                # co-located predictions_phys/last_dynamic_phys on the physics grid)
                physics_loss = torch.tensor(0.0, device=self.device)
                if (
                    physics_weight > 0.0
                    and self.pred_to_physics is not None
                    and self.physics_loss_calc is not None
                    and phys_pred is not None
                ):
                    raw_inputs = batch["raw_inputs_last"].to(self.device, non_blocking=True)

                    phys = self.pred_to_physics.convert(
                        phys_pred, phys_last, raw_inputs, self.device
                    )
                    training_losses, _ = self.physics_loss_calc.compute_training_losses(phys)
                    physics_loss = self.physics_loss_calc.compute_weighted_total(training_losses)

                    for name, val in training_losses.items():
                        physics_accumulator[name] = physics_accumulator.get(name, 0.0) + val.item()

                total_loss = data_loss + physics_weight * physics_loss

            if torch.isnan(total_loss):
                continue

            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()

            losses["data_total"] += data_loss.item()
            losses["physics_total"] += physics_loss.item()
            losses["combined_total"] += total_loss.item()
            losses["raw_total"] += raw_batch
            n_batches += 1

        result = {k: v / max(n_batches, 1) for k, v in losses.items()}
        for name, value in physics_accumulator.items():
            result[f"physics_{name}"] = value / max(n_batches, 1)
        return result

    @torch.no_grad()
    def _validate_epoch_physics(
        self,
        loader: DataLoader,
        physics_weight: float,
    ) -> Dict[str, float]:
        """Validation epoch with physics constraints (Phase 6)."""
        self.model.eval()
        losses = {h: 0.0 for h in self.ALL_HEAD_NAMES}
        losses.update({"data_total": 0.0, "physics_total": 0.0,
                       "combined_total": 0.0, "raw_total": 0.0})
        physics_accumulator: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            with autocast(enabled=self.use_amp):
                output, phys_pred, phys_last = self._federated_forward(batch)

            # Data loss (weighted, mirroring _train_epoch_physics)
            data_loss = 0.0
            raw_batch = 0.0
            for head_name in self.ALL_HEAD_NAMES:
                target = batch[self.HEAD_TARGET_MAP[head_name]].to(self.device, non_blocking=True)
                pred = output[self.HEAD_PRED_MAP[head_name]]
                head_loss = self.criterion._loss(pred, target).item()
                losses[head_name] += head_loss
                data_loss += self._head_weight(head_name) * head_loss
                raw_batch += head_loss

            # Physics loss (single- or multi-rate; see _federated_forward)
            physics_loss = 0.0
            if (
                physics_weight > 0.0
                and self.pred_to_physics is not None
                and self.physics_loss_calc is not None
                and phys_pred is not None
            ):
                raw_inputs = batch["raw_inputs_last"].to(self.device, non_blocking=True)

                phys = self.pred_to_physics.convert(
                    phys_pred, phys_last, raw_inputs, self.device
                )
                training_losses, _ = self.physics_loss_calc.compute_training_losses(phys)
                physics_loss = self.physics_loss_calc.compute_weighted_total(training_losses).item()

                for name, val in training_losses.items():
                    physics_accumulator[name] = physics_accumulator.get(name, 0.0) + val.item()

            losses["data_total"] += data_loss
            losses["physics_total"] += physics_loss
            losses["combined_total"] += data_loss + physics_weight * physics_loss
            losses["raw_total"] += raw_batch
            n_batches += 1

        result = {k: v / max(n_batches, 1) for k, v in losses.items()}
        for name, value in physics_accumulator.items():
            result[f"physics_{name}"] = value / max(n_batches, 1)
        return result

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def save_checkpoint(
        self,
        filepath: Union[str, Path],
        include_optimizer: bool = True,
        normalizer: Optional[Any] = None,
    ) -> None:
        """
        Save a training checkpoint.

        Args:
            filepath: Path to save checkpoint
            include_optimizer: Whether to include optimizer/scheduler state
            normalizer: Optional normalizer to save with checkpoint
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "phase": self.phase,
            "history": self.history.to_dict(),
        }

        if include_optimizer and self.optimizer is not None:
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()

        if include_optimizer and self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if normalizer is not None:
            checkpoint["normalizer"] = normalizer.get_stats()

        if self.config is not None:
            checkpoint["config"] = (
                vars(self.config) if hasattr(self.config, "__dict__") else self.config
            )

        torch.save(checkpoint, filepath)

        if self.local_rank == 0:
            print(f"Checkpoint saved: {filepath}")

    def load_checkpoint(
        self,
        filepath: Union[str, Path],
        load_optimizer: bool = True,
    ) -> Dict[str, Any]:
        """
        Load a training checkpoint.

        Args:
            filepath: Path to checkpoint
            load_optimizer: Whether to load optimizer/scheduler state

        Returns:
            Checkpoint dictionary with any extra data
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.current_epoch = checkpoint.get("epoch", 0)

        if load_optimizer and "optimizer_state_dict" in checkpoint:
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if load_optimizer and "scheduler_state_dict" in checkpoint:
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.local_rank == 0:
            print(f"Checkpoint loaded: {filepath}")

        return checkpoint