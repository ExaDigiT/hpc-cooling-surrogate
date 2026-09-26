"""
Benchmarking utilities for surrogate models.

Provides timing infrastructure and speedup computation
for comparing surrogate models against FMU simulators.
"""

from .speedup import (
    # Configuration
    BenchmarkConfig,
    # Result containers
    TimingResult,
    SpeedupResult,
    BenchmarkResults,
    # Core classes
    FMUTimer,
    SurrogateTimer,
    SpeedupBenchmarker,
    # Convenience functions
    time_surrogate,
    time_fmu,
    compute_speedup,
    benchmark_model,
    benchmark_all_phases,
)


__all__ = [
    # Configuration
    'BenchmarkConfig',
    # Result containers
    'TimingResult',
    'SpeedupResult',
    'BenchmarkResults',
    # Core classes
    'FMUTimer',
    'SurrogateTimer',
    'SpeedupBenchmarker',
    # Convenience functions
    'time_surrogate',
    'time_fmu',
    'compute_speedup',
    'benchmark_model',
    'benchmark_all_phases',
]