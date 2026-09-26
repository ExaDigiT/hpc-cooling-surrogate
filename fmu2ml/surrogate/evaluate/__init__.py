"""
Evaluation utilities for surrogate models.

Provides prediction collection, metrics computation, and result summarization
for all phases of surrogate model development.
"""

from .evaluator import (
    # Result containers
    EvalResults,
    MetricsSummary,
    # Core classes
    PredictionCollector,
    MetricsComputer,
    SurrogateEvaluator,
    # Convenience functions
    evaluate,
    compute_metrics,
    collect_predictions,
    print_summary,
    get_r2_distribution,
)

from .uncertainty import (
    # Result container
    UQResults,
    # MC-dropout state helpers
    enable_mc_dropout,
    set_dropout_p,
    restore_dropout_p,
    mc_dropout_mode,
    # Sampling + metrics
    collect_mc_predictions,
    gaussian_interval_metrics,
    fit_recalibration_scale,
    compute_uncertainty_columns,
    # Orchestration
    evaluate_mc_dropout,
    print_uncertainty_summary,
)

from .pi3nn import (
    train_pi3nn_bounds,
    calibrate_pi3nn,
    evaluate_pi3nn,
)


__all__ = [
    # Result containers
    'EvalResults',
    'MetricsSummary',
    'UQResults',
    # Core classes
    'PredictionCollector',
    'MetricsComputer',
    'SurrogateEvaluator',
    # Convenience functions
    'evaluate',
    'compute_metrics',
    'collect_predictions',
    'print_summary',
    'get_r2_distribution',
    # Uncertainty quantification (MC dropout)
    'enable_mc_dropout',
    'set_dropout_p',
    'restore_dropout_p',
    'mc_dropout_mode',
    'collect_mc_predictions',
    'gaussian_interval_metrics',
    'fit_recalibration_scale',
    'compute_uncertainty_columns',
    'evaluate_mc_dropout',
    'print_uncertainty_summary',
    # PI3NN prediction intervals
    'train_pi3nn_bounds',
    'calibrate_pi3nn',
    'evaluate_pi3nn',
]