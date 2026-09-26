"""
Architecture module for surrogate models.

Provides model implementations and configuration dataclasses
for all 6 phases of surrogate model development.
"""

from .configs import (
    OutputType,
    SystemName,
    DomainType,
    SurrogateConfig,
    MultiRateSpec,
    LSTMConfig,
    DeepONetConfig,
    HybridDeepONetConfig,
    DomainDeepONetConfig,
    FederatedConfig,
    PhysicsInformedConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    Phase5Config,
    Phase6Config,
    create_domain_configs,
    SYSTEM_DEFAULTS,
    DEFAULT_INPUT_PATTERNS,
    DEFAULT_OUTPUT_PATTERNS,
    OUTPUT_CLASSIFICATION,
    DOMAIN_OUTPUTS,
    FEDERATED_HEAD_OUTPUTS,
    HEAD_TYPES,
)

# Shared multi-rate plumbing (phases 1-4 multi-rate variants)
from .multirate import (
    HEAD_META,
    head_layout,
    BranchEncoderBank,
    OperatorHead,
    MultiRateOperatorModel,
)

# Phase 1: Baseline LSTM
from .lstm import (
    BaselineLSTM,
    MultiRateLSTM,
    TemporalAttention,
    lstm,
)

# Phase 2: Basic DeepONet
from .deeponet import (
    FourierFeatures,
    BranchNetwork,
    TrunkNetwork,
    TemporalDeepONet,
    AlgebraicPathway,
    BasicDeepONet,
    MultiRateDeepONet,
    deeponet,
)

# Phase 3: Hybrid DeepONet with fixes
from .hybrid_deeponet import (
    CDUEmbeddingLayer,
    AlgebraicPathwayWithCDU,
    TemporalDeepONetWithFixes,
    HybridDeepONet,
    MultiRateHybridDeepONet,
    hybrid_deeponet,
)

# Phase 4: Domain-specific DeepONet
from .domain_deeponet import (
    DomainBranchNetwork,
    DomainTrunkNetwork,
    DomainDeepONet,
    MultiRateDomainDeepONet,
    domain_deeponet,
    create_domain_models,
)

# Phase 5: Federated DeepMMNet
from .federated import (
    BranchNetwork as FederatedBranchNetwork,
    TBranchNetwork,
    FourierTrunkNetwork,
    DecoderHead,
    SkipDecoderHead,
    FederatedDeepMMNet,
    federated,
)

# Phase 6: Physics-Informed Federated DeepMMNet
from .federated_pi import (
    PhysicsConfig,
    PredictionToPhysics,
    PhysicsLossCalculator,
    PhysicsInformedFederatedDeepMMNet,
    federated_pi,
)

# PI3NN: OOD-aware prediction intervals (wraps the federated mean model)
from .pi3nn import (
    PI3NNBoundHead,
    FederatedPI3NN,
    federated_pi3nn,
)

# Custom architecture building and registry
from .builder import build, CustomArchitectureBuilder
from .registry import SurrogateRegistry, list_architectures


def get_architecture(name: str):
    """Get architecture factory function by name."""
    return SurrogateRegistry().get(name)


__all__ = [
    'HEAD_META',
    'head_layout',
    'BranchEncoderBank',
    'OperatorHead',
    'MultiRateOperatorModel',
    # Enums
    'OutputType',
    'SystemName',
    'DomainType',
    
    # Config classes
    'SurrogateConfig',
    'MultiRateSpec',
    'LSTMConfig',
    'DeepONetConfig',
    'HybridDeepONetConfig',
    'DomainDeepONetConfig',
    'FederatedConfig',
    'PhysicsInformedConfig',
    
    # Phase aliases
    'Phase1Config',
    'Phase2Config',
    'Phase3Config',
    'Phase4Config',
    'Phase5Config',
    'Phase6Config',
    
    # Factory functions
    'create_domain_configs',
    'list_architectures',
    'get_architecture',
    
    # Config constants
    'SYSTEM_DEFAULTS',
    'DEFAULT_INPUT_PATTERNS',
    'DEFAULT_OUTPUT_PATTERNS',
    'OUTPUT_CLASSIFICATION',
    'DOMAIN_OUTPUTS',
    'FEDERATED_HEAD_OUTPUTS',
    'HEAD_TYPES',
    
    # Phase 1: LSTM
    'BaselineLSTM',
    'MultiRateLSTM',
    'TemporalAttention',
    'lstm',
    
    # Phase 2: DeepONet
    'FourierFeatures',
    'BranchNetwork',
    'TrunkNetwork',
    'TemporalDeepONet',
    'AlgebraicPathway',
    'BasicDeepONet',
    'MultiRateDeepONet',
    'deeponet',
    
    # Phase 3: Hybrid DeepONet
    'CDUEmbeddingLayer',
    'AlgebraicPathwayWithCDU',
    'TemporalDeepONetWithFixes',
    'HybridDeepONet',
    'MultiRateHybridDeepONet',
    'hybrid_deeponet',
    
    # Phase 4: Domain DeepONet
    'DomainBranchNetwork',
    'DomainTrunkNetwork',
    'DomainDeepONet',
    'MultiRateDomainDeepONet',
    'domain_deeponet',
    'create_domain_models',
    
    # Phase 5: Federated
    'FederatedBranchNetwork',
    'TBranchNetwork',
    'FourierTrunkNetwork',
    'DecoderHead',
    'SkipDecoderHead',
    'FederatedDeepMMNet',
    'federated',
    
    # Phase 6: Physics-Informed
    'PhysicsConfig',
    'PredictionToPhysics',
    'PhysicsLossCalculator',
    'PhysicsInformedFederatedDeepMMNet',
    'federated_pi',

    # PI3NN prediction intervals
    'PI3NNBoundHead',
    'FederatedPI3NN',
    'federated_pi3nn',

    # Custom architecture building
    'build',
    'CustomArchitectureBuilder',
    'SurrogateRegistry',
]