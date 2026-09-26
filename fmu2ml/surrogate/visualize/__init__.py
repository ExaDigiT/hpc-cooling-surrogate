"""
Visualization utilities for surrogate models.

Provides plotting functions for training curves, evaluation metrics,
benchmark results, and prediction analysis.
"""

from .plotter import (
    # Main class
    SurrogatePlotter,
    # Training visualization
    plot_training_curves,
    plot_learning_rate,
    plot_loss_by_head,
    # Evaluation visualization
    plot_r2_distribution,
    plot_r2_by_type,
    plot_r2_by_group,
    plot_variance_ratio,
    plot_r2_heatmap,
    plot_rmse_by_type,
    plot_model_vs_persistence,
    plot_correlation_by_type,
    plot_loop_dichotomy,
    # Prediction visualization
    plot_predictions,
    plot_error_distribution,
    plot_prediction_vs_target,
    plot_timeseries_chained,
    plot_timeseries_nearconst,
    plot_delta_quality,
    # Benchmark visualization
    plot_speedup,
    plot_latency_heatmap,
    plot_speedup_summary,
    # Physics visualization (Phase 6)
    plot_physics_evolution,
    # Comparison visualization
    plot_phase_comparison,
    # Uncertainty quantification (MC dropout / PI3NN)
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
    coverage_curve,
    calibration_error,
    # Utilities
    save_figure,
    set_style,
)


__all__ = [
    # Main class
    'SurrogatePlotter',
    # Training visualization
    'plot_training_curves',
    'plot_learning_rate',
    'plot_loss_by_head',
    # Evaluation visualization
    'plot_r2_distribution',
    'plot_r2_by_type',
    'plot_r2_by_group',
    'plot_variance_ratio',
    'plot_r2_heatmap',
    'plot_rmse_by_type',
    'plot_model_vs_persistence',
    'plot_correlation_by_type',
    'plot_loop_dichotomy',
    # Prediction visualization
    'plot_predictions',
    'plot_error_distribution',
    'plot_prediction_vs_target',
    'plot_timeseries_chained',
    'plot_timeseries_nearconst',
    'plot_delta_quality',
    # Benchmark visualization
    'plot_speedup',
    'plot_latency_heatmap',
    'plot_speedup_summary',
    # Physics visualization
    'plot_physics_evolution',
    # Comparison visualization
    'plot_phase_comparison',
    # Uncertainty quantification
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
    'coverage_curve',
    'calibration_error',
    # Utilities
    'save_figure',
    'set_style',
]