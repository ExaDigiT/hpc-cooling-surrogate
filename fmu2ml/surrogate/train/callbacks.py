"""
fmu2ml.surrogate.train.callbacks - Training callbacks for surrogate models.

Provides callback utilities for training:
- EarlyStopping: Stop training when validation loss stops improving
- ModelCheckpoint: Save model checkpoints during training
- CallbackList: Manage multiple callbacks
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "CallbackList",
]


# ───────────────────────────────────────────────────────────────────────
# Base Callback Interface
# ───────────────────────────────────────────────────────────────────────


class Callback(ABC):
    """Abstract base class for training callbacks."""

    def on_train_begin(self, trainer: Any) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: Any) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """
        Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            train_losses: Training losses for this epoch
            val_losses: Validation losses for this epoch
            trainer: Reference to the trainer

        Returns:
            True to stop training, False to continue
        """
        return False

    def on_batch_begin(self, batch: int, trainer: Any) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, loss: float, trainer: Any) -> None:
        """Called at the end of each batch."""
        pass


# ───────────────────────────────────────────────────────────────────────
# Early Stopping Callback
# ───────────────────────────────────────────────────────────────────────


class EarlyStopping(Callback):
    """
    Early stopping with best model checkpoint.

    Monitors a metric (default: validation loss) and stops training when
    it stops improving for a specified number of epochs (patience).
    Automatically saves the best model state.
    """

    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 1e-6,
        monitor: str = "val_loss",
        mode: str = "min",
        restore_best: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize EarlyStopping.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            monitor: Metric to monitor (key in val_losses dict, or 'val_loss' for total)
            mode: 'min' for loss (lower is better), 'max' for metrics (higher is better)
            restore_best: Whether to restore best model weights at the end
            verbose: Whether to print messages
        """
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose

        # State
        self.counter = 0
        self.best_value = None
        self.should_stop = False
        self.best_state = None
        self.best_epoch = 0

        # Set comparison function based on mode
        if mode == "min":
            self._is_better = lambda new, best: new < best - min_delta
            self.best_value = float("inf")
        else:
            self._is_better = lambda new, best: new > best + min_delta
            self.best_value = float("-inf")

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop (legacy interface).

        Args:
            val_loss: Current validation loss
            model: Model to checkpoint

        Returns:
            True if training should stop
        """
        if self._is_better(val_loss, self.best_value):
            self.best_value = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def on_epoch_end(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Check if training should stop based on validation metrics."""
        # Get monitored value
        if self.monitor == "val_loss":
            current = val_losses.get("total", val_losses.get("val_loss", 0.0))
        else:
            current = val_losses.get(self.monitor, 0.0)

        # Check if improved
        if self._is_better(current, self.best_value):
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0

            # Save best state
            if hasattr(trainer, "model"):
                self.best_state = {
                    k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
                }

            if self.verbose:
                print(f"  EarlyStopping: New best {self.monitor}={current:.6f}")
        else:
            self.counter += 1
            if self.verbose and self.counter > 0:
                print(
                    f"  EarlyStopping: No improvement for {self.counter}/{self.patience} epochs"
                )

        self.should_stop = self.counter >= self.patience
        return self.should_stop

    def on_train_end(self, trainer: Any) -> None:
        """Restore best model weights if enabled."""
        if self.restore_best and self.best_state is not None:
            if hasattr(trainer, "model"):
                trainer.model.load_state_dict(self.best_state)
                if self.verbose:
                    print(
                        f"  EarlyStopping: Restored best weights from epoch {self.best_epoch}"
                    )

    def load_best(self, model: nn.Module) -> None:
        """
        Load best model state into model (legacy interface).

        Args:
            model: Model to load weights into
        """
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    def reset(self) -> None:
        """Reset early stopping state for new training phase."""
        self.counter = 0
        self.should_stop = False
        self.best_state = None
        self.best_epoch = 0

        if self.mode == "min":
            self.best_value = float("inf")
        else:
            self.best_value = float("-inf")


# ───────────────────────────────────────────────────────────────────────
# Model Checkpoint Callback
# ───────────────────────────────────────────────────────────────────────


class ModelCheckpoint(Callback):
    """
    Save model checkpoints during training.

    Can save based on best validation metric, at regular intervals,
    or at the end of training.
    """

    def __init__(
        self,
        dirpath: Union[str, Path],
        filename: str = "model_{epoch:03d}_{val_loss:.4f}.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        save_last: bool = True,
        save_weights_only: bool = False,
        every_n_epochs: Optional[int] = None,
        verbose: bool = True,
    ):
        """
        Initialize ModelCheckpoint.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename template (supports {epoch}, {val_loss}, etc.)
            monitor: Metric to monitor for best model
            mode: 'min' or 'max'
            save_best_only: Only save when monitored metric improves
            save_last: Always save at the last epoch
            save_weights_only: Save only model weights (not optimizer, scheduler)
            every_n_epochs: Save every N epochs (overrides save_best_only)
            verbose: Whether to print messages
        """
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.save_weights_only = save_weights_only
        self.every_n_epochs = every_n_epochs
        self.verbose = verbose

        # State
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_path = None

        # Set comparison function
        if mode == "min":
            self._is_better = lambda new, best: new < best
        else:
            self._is_better = lambda new, best: new > best

    def on_train_begin(self, trainer: Any) -> None:
        """Create checkpoint directory."""
        self.dirpath.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """Save checkpoint if conditions are met."""
        # Get monitored value
        if self.monitor == "val_loss":
            current = val_losses.get("total", val_losses.get("val_loss", 0.0))
        else:
            current = val_losses.get(self.monitor, 0.0)

        should_save = False

        # Check periodic saving
        if self.every_n_epochs is not None and (epoch + 1) % self.every_n_epochs == 0:
            should_save = True

        # Check if best
        if self.save_best_only and self._is_better(current, self.best_value):
            self.best_value = current
            should_save = True

        if should_save:
            self._save_checkpoint(epoch, current, trainer)

        return False  # Never stop training

    def on_train_end(self, trainer: Any) -> None:
        """Save final checkpoint if save_last is True."""
        if self.save_last:
            self._save_checkpoint(
                epoch=getattr(trainer, "current_epoch", 0),
                val_loss=self.best_value,
                trainer=trainer,
                is_last=True,
            )

    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        trainer: Any,
        is_last: bool = False,
    ) -> None:
        """Save a checkpoint."""
        if is_last:
            filename = "last.pt"
        else:
            filename = self.filename.format(epoch=epoch, val_loss=val_loss)

        filepath = self.dirpath / filename

        # Build checkpoint dict
        checkpoint = {"epoch": epoch, "val_loss": val_loss}

        if hasattr(trainer, "model"):
            if self.save_weights_only:
                checkpoint["model_state_dict"] = trainer.model.state_dict()
            else:
                checkpoint["model_state_dict"] = trainer.model.state_dict()

                if hasattr(trainer, "optimizer") and trainer.optimizer is not None:
                    checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

                if hasattr(trainer, "scheduler") and trainer.scheduler is not None:
                    checkpoint["scheduler_state_dict"] = trainer.scheduler.state_dict()

        if hasattr(trainer, "config"):
            checkpoint["config"] = trainer.config

        torch.save(checkpoint, filepath)

        if not is_last:
            self.best_path = filepath

        if self.verbose:
            print(f"  Checkpoint saved: {filepath}")


# ───────────────────────────────────────────────────────────────────────
# Callback List Manager
# ───────────────────────────────────────────────────────────────────────


class CallbackList:
    """
    Manage multiple callbacks.

    Provides a unified interface to call all callbacks at each event.
    """

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize CallbackList.

        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback."""
        self.callbacks.append(callback)

    def on_train_begin(self, trainer: Any) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer: Any) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer: Any) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(
        self,
        epoch: int,
        train_losses: Dict[str, float],
        val_losses: Dict[str, float],
        trainer: Any,
    ) -> bool:
        """
        Call on_epoch_end for all callbacks.

        Returns:
            True if any callback requests to stop training
        """
        should_stop = False
        for callback in self.callbacks:
            if callback.on_epoch_end(epoch, train_losses, val_losses, trainer):
                should_stop = True
        return should_stop

    def on_batch_begin(self, batch: int, trainer: Any) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, trainer)

    def on_batch_end(self, batch: int, loss: float, trainer: Any) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, loss, trainer)

    def __iter__(self):
        return iter(self.callbacks)

    def __len__(self):
        return len(self.callbacks)