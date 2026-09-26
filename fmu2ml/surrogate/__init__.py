"""
Surrogate Module: fmu2ml.surrogate

A comprehensive sub-package for surrogate model development across all six phases:
- Phase 1: Baseline LSTM with temporal attention
- Phase 2: Basic DeepONet (LSTM branch + Fourier trunk)
- Phase 3: Hybrid DeepONet (temporal + algebraic pathways)
- Phase 4: Domain-specific DeepONet (per-domain experts)
- Phase 5: Federated DeepMMNet (shared encoder + 6 decoder heads)
- Phase 6: Physics-Informed Federated DeepMMNet

Example usage:
    import fmu2ml.surrogate as surrogate

    # Use built-in architectures
    model = surrogate.architecture.lstm(config)
    model = surrogate.architecture.federated(config, column_info)

    # Train
    trainer = surrogate.train.SurrogateTrainer(model, config)
    trainer.fit(train_loader, val_loader)

    # Evaluate
    results = surrogate.evaluate.evaluate(model, test_loader, normalizer, column_info, config)
    surrogate.evaluate.print_summary(results)

    # Benchmark
    bench_results = surrogate.benchmark.benchmark_model(model, input_fn, phase="federated")
    surrogate.benchmark.SpeedupBenchmarker().print_results(bench_results)

    # Visualize
    plotter = surrogate.visualize.SurrogatePlotter(output_dir='./plots')
    plotter.plot_evaluation_suite(results.metrics_df, predictions, targets)
"""

__version__ = "0.1.0"
__author__ = "HPCI Lab"

# ───────────────────────────────────────────────────────────────────────
# Submodule imports (lazy loading pattern for faster imports)
# ───────────────────────────────────────────────────────────────────────

from . import architecture
from . import data
from . import train
from . import evaluate
from . import benchmark
from . import visualize


# ───────────────────────────────────────────────────────────────────────
# Convenience re-exports from architecture
# ───────────────────────────────────────────────────────────────────────

from .architecture import (
    # Factory functions
    lstm,
    deeponet,
    hybrid_deeponet,
    domain_deeponet,
    federated,
    federated_pi,
    federated_pi3nn,
    FederatedPI3NN,
    # Utilities
    list_architectures,
    get_architecture,
    create_domain_configs,
    # Config classes
    SurrogateConfig,
    LSTMConfig,
    DeepONetConfig,
    HybridDeepONetConfig,
    DomainDeepONetConfig,
    FederatedConfig,
    PhysicsInformedConfig,
    # Phase aliases
    Phase1Config,
    Phase2Config,
    Phase3Config,
    Phase4Config,
    Phase5Config,
    Phase6Config,
)

# ───────────────────────────────────────────────────────────────────────
# Convenience re-exports from data
# ───────────────────────────────────────────────────────────────────────

from .data import (
    # Column info
    build_column_info,
    build_federated_column_info,
    get_system_column_info,
    # Normalizers
    SurrogateNormalizer,
    ZScoreNormalizer,
    DeltaNormalizer,
    FederatedNormalizer,
    # Datasets
    LSTMDataset,
    HybridDataset,
    DomainDataset,
    FederatedDataset,
    # Dataloader factories
    create_surrogate_dataloaders,
    # Per-file (per-chunk) loading
    load_chunk_dataframes,
    concat_chunk_dataframes,
    # Out-of-core memmap store (recommended for full-data Phase 5/6)
    build_chunk_store,
    build_store_for_config,
    create_dataloaders_federated_store,
)

# ───────────────────────────────────────────────────────────────────────
# Convenience re-exports from train
# ───────────────────────────────────────────────────────────────────────

from .train import (
    # Trainer
    SurrogateTrainer,
    TrainingHistory,
    # Loss functions
    BaseLoss,
    HybridLoss,
    DomainLoss,
    HeadLoss,
    PhysicsConstraintLoss,
    SurrogateLoss,
    create_loss,
    # Callbacks
    EarlyStopping,
    ModelCheckpoint,
)

# ───────────────────────────────────────────────────────────────────────
# Convenience re-exports from evaluate
# ───────────────────────────────────────────────────────────────────────

from .evaluate import (
    # Core classes
    SurrogateEvaluator,
    EvalResults,
    UQResults,
    # Convenience functions
    evaluate,
    compute_metrics,
    collect_predictions,
    print_summary,
    # Uncertainty quantification (MC dropout)
    evaluate_mc_dropout,
    enable_mc_dropout,
    mc_dropout_mode,
    print_uncertainty_summary,
    # Uncertainty quantification (PI3NN)
    train_pi3nn_bounds,
    calibrate_pi3nn,
    evaluate_pi3nn,
)

# ───────────────────────────────────────────────────────────────────────
# Convenience re-exports from benchmark
# ───────────────────────────────────────────────────────────────────────

from .benchmark import (
    # Core classes
    SpeedupBenchmarker,
    BenchmarkConfig,
    BenchmarkResults,
    # Convenience functions
    time_surrogate,
    time_fmu,
    compute_speedup,
    benchmark_model,
)

# ───────────────────────────────────────────────────────────────────────
# Convenience re-exports from visualize
# ───────────────────────────────────────────────────────────────────────

from .visualize import (
    # Main class
    SurrogatePlotter,
    # Training
    plot_training_curves,
    # Evaluation
    plot_r2_distribution,
    plot_r2_by_type,
    plot_variance_ratio,
    plot_r2_heatmap,
    # Predictions
    plot_predictions,
    plot_prediction_vs_target,
    plot_error_distribution,
    # Physics
    plot_physics_evolution,
    # Benchmark
    plot_speedup,
    plot_latency_heatmap,
    plot_speedup_summary,
    # Comparison
    plot_phase_comparison,
    # Uncertainty quantification
    plot_prediction_with_uncertainty,
    plot_reliability_diagram,
    plot_uncertainty_vs_error,
    plot_uncertainty_by_group,
    plot_timeseries_with_uncertainty,
    plot_group_uncertainty_timeseries,
    plot_cdu_uncertainty_timeseries,
    plot_cdu_io_uncertainty,
    plot_cdu_output_uncertainty,
    plot_worst_calibrated,
    plot_uncertainty_dashboard,
    plot_uq_comparison,
)


# ───────────────────────────────────────────────────────────────────────
# Module-level functions
# ───────────────────────────────────────────────────────────────────────

def get_version() -> str:
    """Return package version."""
    return __version__


def list_phases() -> list:
    """List all supported surrogate model phases."""
    return [
        "Phase 1: Baseline LSTM",
        "Phase 2: Basic DeepONet",
        "Phase 3: Hybrid DeepONet",
        "Phase 4: Domain-specific DeepONet",
        "Phase 5: Federated DeepMMNet",
        "Phase 6: Physics-Informed Federated DeepMMNet",
    ]


def quick_start_guide() -> str:
    """Return quick start guide string."""
    return """
    fmu2ml.surrogate Quick Start Guide
    ===================================
    
    1. Import the module:
       >>> import fmu2ml.surrogate as surrogate
    
    2. Create a configuration:
       >>> config = surrogate.FederatedConfig(num_cdus=257, system_name='summit')
    
    3. Build column info from your data:
       >>> column_info = surrogate.build_federated_column_info(df, config)
    
    4. Create a model:
       >>> model = surrogate.federated(config, column_info)
    
    5. Create dataloaders:
       >>> loaders = surrogate.create_surrogate_dataloaders(
       ...     df, column_info, config, phase='federated'
       ... )
    
    6. Train the model:
       >>> trainer = surrogate.SurrogateTrainer(model, config)
       >>> history = trainer.fit(loaders['train'], loaders['val'])
    
    7. Evaluate:
       >>> results = surrogate.evaluate(
       ...     model, loaders['test'], normalizer, column_info, config
       ... )
       >>> surrogate.print_summary(results)
    
    8. Visualize:
       >>> plotter = surrogate.SurrogatePlotter(output_dir='./results')
       >>> plotter.plot_evaluation_suite(results.metrics_df)
    
    9. Benchmark:
       >>> bench = surrogate.benchmark_model(model, input_fn, phase='federated')
       >>> print(f"Max speedup: {bench.max_throughput_speedup:.1f}x")
    
    For more details, see the documentation for each submodule:
    - surrogate.architecture: Model architectures
    - surrogate.data: Datasets and normalization
    - surrogate.train: Training infrastructure
    - surrogate.evaluate: Evaluation metrics
    - surrogate.benchmark: Speedup benchmarking
    - surrogate.visualize: Plotting utilities
    """


# ───────────────────────────────────────────────────────────────────────
# __all__ export list
# ───────────────────────────────────────────────────────────────────────

__all__ = [
    # Version
    '__version__',
    'get_version',
    'list_phases',
    'quick_start_guide',
    
    # Submodules
    'architecture',
    'data',
    'train',
    'evaluate',
    'benchmark',
    'visualize',
    
    # Architecture factory functions
    'lstm',
    'deeponet',
    'hybrid_deeponet',
    'domain_deeponet',
    'federated',
    'federated_pi',
    'federated_pi3nn',
    'FederatedPI3NN',
    'list_architectures',
    'get_architecture',
    'create_domain_configs',
    
    # Config classes
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
    
    # Data
    'build_column_info',
    'build_federated_column_info',
    'get_system_column_info',
    'SurrogateNormalizer',
    'ZScoreNormalizer',
    'DeltaNormalizer',
    'FederatedNormalizer',
    'LSTMDataset',
    'HybridDataset',
    'DomainDataset',
    'FederatedDataset',
    'create_surrogate_dataloaders',
    'load_chunk_dataframes',
    'concat_chunk_dataframes',
    'build_chunk_store',
    'build_store_for_config',
    'create_dataloaders_federated_store',

    # Training
    'SurrogateTrainer',
    'TrainingHistory',
    'BaseLoss',
    'HybridLoss',
    'DomainLoss',
    'HeadLoss',
    'PhysicsConstraintLoss',
    'SurrogateLoss',
    'create_loss',
    'EarlyStopping',
    'ModelCheckpoint',
    
    # Evaluation
    'SurrogateEvaluator',
    'EvalResults',
    'UQResults',
    'evaluate',
    'compute_metrics',
    'collect_predictions',
    'print_summary',
    # Uncertainty quantification
    'evaluate_mc_dropout',
    'enable_mc_dropout',
    'mc_dropout_mode',
    'print_uncertainty_summary',
    'train_pi3nn_bounds',
    'calibrate_pi3nn',
    'evaluate_pi3nn',
    
    # Benchmark
    'SpeedupBenchmarker',
    'BenchmarkConfig',
    'BenchmarkResults',
    'time_surrogate',
    'time_fmu',
    'compute_speedup',
    'benchmark_model',
    
    # Visualization
    'SurrogatePlotter',
    'plot_training_curves',
    'plot_r2_distribution',
    'plot_r2_by_type',
    'plot_variance_ratio',
    'plot_r2_heatmap',
    'plot_predictions',
    'plot_prediction_vs_target',
    'plot_error_distribution',
    'plot_physics_evolution',
    'plot_speedup',
    'plot_latency_heatmap',
    'plot_speedup_summary',
    'plot_phase_comparison',
    'plot_prediction_with_uncertainty',
    'plot_reliability_diagram',
    'plot_uncertainty_vs_error',
    'plot_uncertainty_by_group',
    'plot_timeseries_with_uncertainty',
    'plot_group_uncertainty_timeseries',
    'plot_cdu_uncertainty_timeseries',
    'plot_cdu_io_uncertainty',
    'plot_cdu_output_uncertainty',
    'plot_worst_calibrated',
    'plot_uncertainty_dashboard',
    'plot_uq_comparison',
]