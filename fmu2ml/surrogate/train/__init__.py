"""
fmu2ml.surrogate.train - Training infrastructure for surrogate models.

Provides phase-aware training utilities including loss functions, callbacks,
and a unified trainer that handles all six surrogate model phases.
"""

from .loss import (
    BaseLoss,
    MSELoss,
    HuberLoss,
    HybridLoss,
    DomainLoss,
    HeadLoss,
    PhysicsConstraintLoss,
    SurrogateLoss,
    create_loss,
)
from .callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    CallbackList,
)
from .trainer import (
    SurrogateTrainer,
    TrainingHistory,
)

__all__ = [
    # Loss functions
    "BaseLoss",
    "MSELoss",
    "HuberLoss",
    "HybridLoss",
    "DomainLoss",
    "HeadLoss",
    "PhysicsConstraintLoss",
    "SurrogateLoss",
    "create_loss",
    # Callbacks
    "EarlyStopping",
    "ModelCheckpoint",
    "CallbackList",
    # Trainer
    "SurrogateTrainer",
    "TrainingHistory",
]