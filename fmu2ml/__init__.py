"""
FMU2ML: Physics-informed Machine Learning for Datacenter Cooling Systems

A modular package for datacenter cooling simulation and machine learning across
different HPC systems (Summit, Marconi100, Frontier, Fugaku, Lassen).

Features:
- System-agnostic configuration with RAPS integration
- Data generation (power scenarios, temperature modeling)
- FMU simulation integration
- Surrogate modeling via the `fmu2ml.surrogate` sub-package
"""


__author__ = "HPCI Lab"

# Configuration
from fmu2ml.config import SystemConfig, get_system_config

# Data
from fmu2ml.data.generators import PowerGenerator, TemperatureGenerator, ScenarioGenerator
from fmu2ml.data.processors import NormalizationHandler, create_data_loaders

# Evaluation
from fmu2ml.evaluation import calculate_metrics, MetricsCalculator, PhysicsValidator

# Utils
from fmu2ml.utils import save_results, load_results, setup_logger

from fmu2ml import surrogate

__all__ = [
    # Config
    'SystemConfig',
    'get_system_config',

    # Data
    'PowerGenerator',
    'TemperatureGenerator',
    'ScenarioGenerator',
    'NormalizationHandler',
    'create_data_loaders',

    # Evaluation
    'calculate_metrics',
    'MetricsCalculator',
    'PhysicsValidator',

    # Utils
    'save_results',
    'load_results',
    'setup_logger',

    # Surrogate models
    'surrogate',
]
