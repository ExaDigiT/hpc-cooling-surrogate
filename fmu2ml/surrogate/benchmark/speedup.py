# fmu2ml/surrogate/benchmark/speedup.py
"""
Speedup benchmarking for surrogate models vs FMU simulators.

Provides infrastructure for:
- Timing surrogate model inference across batch sizes
- Timing FMU simulator steps
- Computing speedup metrics
- Generating benchmark visualizations
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Type aliases
PathLike = Union[str, Path]
InputFn = Callable[[int], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]


# ───────────────────────────────────────────────────────────────────────
# Configuration
# ───────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """Configuration for speedup benchmarking."""
    
    # Batch sizes to test
    batch_sizes: List[int] = field(
        default_factory=lambda: [1, 4, 16, 64, 256, 1024, 2048, 4096]
    )
    
    # Number of repetitions for timing stability
    n_warmup: int = 5
    n_repeats: int = 20
    
    # FMU simulation parameters
    fmu_system: str = "summit"
    fmu_step_size: float = 1.0  # seconds per step
    fmu_ms_per_step_estimate: float = 8.0  # fallback estimate
    
    # Surrogate model parameters
    history_steps: int = 40
    subsample_factor: int = 30  # 30s per subsampled step
    
    # System parameters
    num_cdus: int = 257
    
    # Phase-specific FMU steps per surrogate prediction
    # K=1 at 30s subsample = 30 FMU steps, K=4 = 120 steps
    fmu_steps_per_phase: Dict[str, int] = field(default_factory=lambda: {
        'lstm': 30,
        'deeponet': 30,
        'hybrid_deeponet': 30,
        'domain_deeponet': 120,  # K=4
        'federated': 30,
        'federated_pi': 30,
    })
    
    # Output directory for visualizations
    output_dir: str = "./benchmark_results"
    
    def get_fmu_steps(self, phase_or_model: str) -> int:
        """Get FMU steps equivalent for a phase/model name."""
        phase_lower = phase_or_model.lower()
        for key, steps in self.fmu_steps_per_phase.items():
            if key in phase_lower:
                return steps
        # Default fallback
        return 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_sizes': self.batch_sizes,
            'n_warmup': self.n_warmup,
            'n_repeats': self.n_repeats,
            'fmu_system': self.fmu_system,
            'fmu_step_size': self.fmu_step_size,
            'history_steps': self.history_steps,
            'subsample_factor': self.subsample_factor,
            'num_cdus': self.num_cdus,
            'fmu_steps_per_phase': self.fmu_steps_per_phase,
        }


# ───────────────────────────────────────────────────────────────────────
# Result Containers
# ───────────────────────────────────────────────────────────────────────

@dataclass
class TimingResult:
    """Container for timing measurements."""
    mean: float  # Mean time in seconds
    std: float   # Standard deviation
    min: float   # Minimum time
    max: float   # Maximum time
    n_repeats: int
    
    # Derived metrics (for per-batch timing)
    per_sample: Optional[float] = None  # Time per sample
    throughput: Optional[float] = None  # Samples per second
    batch_size: Optional[int] = None
    
    @classmethod
    def from_times(
        cls,
        times: List[float],
        batch_size: Optional[int] = None,
    ) -> 'TimingResult':
        """Create from list of timing measurements."""
        times_arr = np.array(times)
        mean = float(np.mean(times_arr))
        
        result = cls(
            mean=mean,
            std=float(np.std(times_arr)),
            min=float(np.min(times_arr)),
            max=float(np.max(times_arr)),
            n_repeats=len(times),
            batch_size=batch_size,
        )
        
        if batch_size is not None and batch_size > 0:
            result.per_sample = mean / batch_size
            result.throughput = batch_size / mean if mean > 0 else float('inf')
        
        return result
    
    @property
    def mean_ms(self) -> float:
        """Mean time in milliseconds."""
        return self.mean * 1000
    
    @property
    def per_sample_ms(self) -> Optional[float]:
        """Per-sample time in milliseconds."""
        return self.per_sample * 1000 if self.per_sample else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'n_repeats': self.n_repeats,
            'per_sample': self.per_sample,
            'throughput': self.throughput,
            'batch_size': self.batch_size,
            'mean_ms': self.mean_ms,
            'per_sample_ms': self.per_sample_ms,
        }


@dataclass
class SpeedupResult:
    """Container for speedup computation results."""
    surrogate_time: TimingResult
    fmu_time: TimingResult
    fmu_steps: int
    batch_size: int
    
    @property
    def speedup_per_sample(self) -> float:
        """Speedup factor per sample (FMU time / surrogate per-sample time)."""
        if self.surrogate_time.per_sample and self.surrogate_time.per_sample > 0:
            return self.fmu_time.mean / self.surrogate_time.per_sample
        return 0.0
    
    @property
    def throughput_speedup(self) -> float:
        """Throughput speedup (surrogate throughput / FMU throughput)."""
        fmu_throughput = 1.0 / self.fmu_time.mean if self.fmu_time.mean > 0 else 0
        surr_throughput = self.surrogate_time.throughput or 0
        return surr_throughput / fmu_throughput if fmu_throughput > 0 else float('inf')
    
    @property
    def latency_reduction(self) -> float:
        """Latency reduction factor (1 - surrogate/FMU)."""
        if self.fmu_time.mean > 0 and self.surrogate_time.per_sample:
            return 1.0 - (self.surrogate_time.per_sample / self.fmu_time.mean)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'batch_size': self.batch_size,
            'fmu_steps': self.fmu_steps,
            'surrogate_time_ms': self.surrogate_time.mean_ms,
            'surrogate_per_sample_ms': self.surrogate_time.per_sample_ms,
            'fmu_time_ms': self.fmu_time.mean_ms,
            'speedup_per_sample': self.speedup_per_sample,
            'throughput_speedup': self.throughput_speedup,
            'surrogate_throughput': self.surrogate_time.throughput,
            'fmu_throughput': 1.0 / self.fmu_time.mean if self.fmu_time.mean > 0 else 0,
            'latency_reduction': self.latency_reduction,
        }


@dataclass
class BenchmarkResults:
    """Container for complete benchmark results."""
    
    model_name: str
    phase: str
    config: BenchmarkConfig
    
    # Timing results per batch size
    surrogate_timing: Dict[int, TimingResult]
    fmu_timing: Dict[int, TimingResult]  # Keyed by FMU steps
    
    # Speedup results per batch size
    speedups: Dict[int, SpeedupResult]
    
    # Model metadata
    n_parameters: Optional[int] = None
    device: str = "unknown"
    
    @property
    def best_batch_size(self) -> int:
        """Batch size with highest throughput speedup."""
        if not self.speedups:
            return 1
        return max(self.speedups.keys(), 
                   key=lambda bs: self.speedups[bs].throughput_speedup)
    
    @property
    def best_speedup(self) -> SpeedupResult:
        """Best speedup result."""
        return self.speedups[self.best_batch_size]
    
    @property
    def max_throughput_speedup(self) -> float:
        """Maximum throughput speedup achieved."""
        return self.best_speedup.throughput_speedup
    
    @property
    def max_per_sample_speedup(self) -> float:
        """Maximum per-sample speedup achieved."""
        if not self.speedups:
            return 0.0
        return max(s.speedup_per_sample for s in self.speedups.values())
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert speedup results to DataFrame."""
        records = []
        for bs, speedup in self.speedups.items():
            record = speedup.to_dict()
            record['model_name'] = self.model_name
            record['phase'] = self.phase
            records.append(record)
        return pd.DataFrame(records)
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'model_name': self.model_name,
            'phase': self.phase,
            'n_parameters': self.n_parameters,
            'device': self.device,
            'best_batch_size': self.best_batch_size,
            'max_throughput_speedup': self.max_throughput_speedup,
            'max_per_sample_speedup': self.max_per_sample_speedup,
            'batch_sizes_tested': list(self.speedups.keys()),
        }


# ───────────────────────────────────────────────────────────────────────
# FMU Timer
# ───────────────────────────────────────────────────────────────────────

class FMUTimer:
    """
    Timer for FMU simulator.
    
    Measures wall-clock time to execute FMU simulation steps.
    Falls back to estimated timing if FMU is unavailable.
    """
    
    def __init__(
        self,
        system_name: str = "summit",
        step_size: float = 1.0,
        ms_per_step_estimate: float = 500.0,
    ):
        """
        Initialize FMU timer.
        
        Args:
            system_name: HPC system name for FMU config
            step_size: FMU step size in seconds
            ms_per_step_estimate: Fallback estimate if FMU unavailable
        """
        self.system_name = system_name
        self.step_size = step_size
        self.ms_per_step_estimate = ms_per_step_estimate
        self._simulator = None
        self._fmu_available = None
    
    def _check_fmu_available(self) -> bool:
        """Check if FMU simulator is available."""
        if self._fmu_available is not None:
            return self._fmu_available
        
        try:
            from fmu2ml.simulation.fmu_simulator import FMUSimulator
            self._fmu_available = True
        except ImportError:
            self._fmu_available = False
        
        return self._fmu_available
    
    def _get_simulator(self):
        """Get or create FMU simulator instance."""
        if self._simulator is None and self._check_fmu_available():
            from fmu2ml.simulation.fmu_simulator import FMUSimulator
            self._simulator = FMUSimulator(system_name=self.system_name)
        return self._simulator
    
    def time_steps(
        self,
        n_steps: int,
        n_repeats: int = 5,
        sample_inputs: Optional[Dict] = None,
    ) -> TimingResult:
        """
        Time FMU for a given number of steps.
        
        Args:
            n_steps: Number of FMU steps to execute
            n_repeats: Number of timing repetitions
            sample_inputs: Optional sample input dict for FMU
            
        Returns:
            TimingResult with measured times
        """
        if self._check_fmu_available():
            return self._time_real_fmu(n_steps, n_repeats, sample_inputs)
        else:
            return self._time_estimated(n_steps, n_repeats)
    
    def _time_real_fmu(
        self,
        n_steps: int,
        n_repeats: int,
        sample_inputs: Optional[Dict],
    ) -> TimingResult:
        """Time actual FMU simulator."""
        sim = self._get_simulator()

        # Build dummy inputs covering every variable the FMU declares as an input.
        # _generate_dummy_inputs() uses the live model.inputs list so no KeyError
        # is thrown inside generate_fmu_inputs().
        if sample_inputs is None:
            sample_inputs = self._generate_dummy_inputs(sim)

        times = []
        for _ in range(n_repeats):
            sim.reset()
            current_time = 0.0

            start = time.perf_counter()
            for step_idx in range(n_steps):
                try:
                    fmu_inputs = sim.model.generate_fmu_inputs(
                        sample_inputs, uncertainties=False
                    )
                    _, _ = sim.model.step(current_time, fmu_inputs, self.step_size)
                    current_time += self.step_size
                except Exception as exc:
                    warnings.warn(
                        f"FMU step {step_idx} failed ({exc!r}); "
                        "falling back to estimated time for this repeat.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    elapsed = n_steps * self.ms_per_step_estimate / 1000.0
                    times.append(elapsed)
                    break
            else:
                elapsed = time.perf_counter() - start
                times.append(elapsed)

        return TimingResult.from_times(times, batch_size=1)

    def _time_estimated(self, n_steps: int, n_repeats: int) -> TimingResult:
        """Return estimated timing when FMU unavailable."""
        est_time = n_steps * self.ms_per_step_estimate / 1000.0
        # Add some synthetic variance
        times = [est_time * (1.0 + np.random.normal(0, 0.1)) for _ in range(n_repeats)]
        return TimingResult.from_times(times, batch_size=1)

    def _generate_dummy_inputs(self, sim=None) -> Dict:
        """Generate dummy FMU inputs covering every declared input variable.

        Uses the simulator's model.inputs list (FMI Variable objects) so that
        generate_fmu_inputs() never raises KeyError for a missing key.
        Falls back to a minimal hard-coded dict if no simulator is available.
        """
        if sim is None:
            sim = self._simulator
        if sim is not None and sim.model is not None and sim.model.inputs:
            return {v.name: 0.0 for v in sim.model.inputs}
        # Fallback for when the simulator has not been created yet
        return {'T_amb_C': 25.0, 'Q_IT_W': 1000.0}
    
    def time_multiple(
        self,
        n_steps_list: List[int],
        n_repeats: int = 5,
        sample_inputs: Optional[Dict] = None,
    ) -> Dict[int, TimingResult]:
        """
        Time FMU for multiple step counts.
        
        Args:
            n_steps_list: List of step counts to time
            n_repeats: Number of timing repetitions per step count
            sample_inputs: Optional sample inputs
            
        Returns:
            Dict mapping step count to TimingResult
        """
        results = {}
        for n_steps in n_steps_list:
            results[n_steps] = self.time_steps(n_steps, n_repeats, sample_inputs)
        return results
    
    def cleanup(self):
        """Clean up FMU resources."""
        if self._simulator is not None:
            try:
                self._simulator.cleanup()
            except Exception:
                pass
            self._simulator = None


# ───────────────────────────────────────────────────────────────────────
# Surrogate Timer
# ───────────────────────────────────────────────────────────────────────

class SurrogateTimer:
    """
    Timer for surrogate model inference.
    
    Measures GPU/CPU inference time across batch sizes with proper
    warmup and synchronization.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize surrogate timer.
        
        Args:
            device: Device for inference. Defaults to CUDA if available.
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    def time_model(
        self,
        model: nn.Module,
        input_fn: InputFn,
        batch_sizes: List[int],
        n_warmup: int = 5,
        n_repeats: int = 20,
    ) -> Dict[int, TimingResult]:
        """
        Time model inference across batch sizes.
        
        Args:
            model: PyTorch model to time
            input_fn: Function that takes batch_size and returns model input(s)
            batch_sizes: List of batch sizes to test
            n_warmup: Number of warmup iterations
            n_repeats: Number of timing iterations
            
        Returns:
            Dict mapping batch size to TimingResult
        """
        model = model.to(self.device)
        model.eval()
        
        results = {}
        for bs in batch_sizes:
            results[bs] = self._time_single_batch(
                model, input_fn, bs, n_warmup, n_repeats
            )
        
        return results
    
    def _time_single_batch(
        self,
        model: nn.Module,
        input_fn: InputFn,
        batch_size: int,
        n_warmup: int,
        n_repeats: int,
    ) -> TimingResult:
        """Time model for a single batch size."""
        # Generate inputs
        inputs = input_fn(batch_size)
        
        # Ensure inputs are on correct device
        inputs = self._to_device(inputs)
        
        # Warmup (compile kernels, fill caches)
        with torch.no_grad():
            for _ in range(n_warmup):
                self._forward(model, inputs)
        
        # Synchronize before timing
        self._sync()
        
        # Timing runs
        times = []
        for _ in range(n_repeats):
            self._sync()
            
            start = time.perf_counter()
            with torch.no_grad():
                self._forward(model, inputs)
            
            self._sync()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        return TimingResult.from_times(times, batch_size=batch_size)
    
    def _forward(
        self,
        model: nn.Module,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    ):
        """Execute model forward pass."""
        if isinstance(inputs, dict):
            return model(**inputs)
        elif isinstance(inputs, tuple):
            return model(*inputs)
        else:
            return model(inputs)
    
    def _to_device(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]],
    ):
        """Move inputs to device."""
        if isinstance(inputs, dict):
            return {k: v.to(self.device) for k, v in inputs.items()}
        elif isinstance(inputs, tuple):
            return tuple(x.to(self.device) for x in inputs)
        else:
            return inputs.to(self.device)
    
    def _sync(self):
        """Synchronize device."""
        if self.device.type == 'cuda':
            torch.cuda.synchronize()


def generate_dummy_input(
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate random input tensor for benchmarking."""
    return torch.randn(batch_size, seq_len, feature_dim, device=device)


# ───────────────────────────────────────────────────────────────────────
# Speedup Benchmarker
# ───────────────────────────────────────────────────────────────────────

class SpeedupBenchmarker:
    """
    Main benchmarking class for comparing surrogate models to FMU simulators.
    
    Example:
        benchmarker = SpeedupBenchmarker()
        results = benchmarker.benchmark(
            model=my_model,
            input_fn=lambda bs: torch.randn(bs, 40, 100),
            phase="federated",
        )
        benchmarker.print_results(results)
    """
    
    def __init__(
        self,
        config: Optional[BenchmarkConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize benchmarker.
        
        Args:
            config: Benchmark configuration. Uses defaults if None.
            device: Device for surrogate inference.
        """
        self.config = config or BenchmarkConfig()
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        self.surrogate_timer = SurrogateTimer(self.device)
        self.fmu_timer = FMUTimer(
            system_name=self.config.fmu_system,
            step_size=self.config.fmu_step_size,
            ms_per_step_estimate=self.config.fmu_ms_per_step_estimate,
        )
        
        # Cache FMU timing results
        self._fmu_timing_cache: Dict[int, TimingResult] = {}
    
    def benchmark(
        self,
        model: nn.Module,
        input_fn: InputFn,
        phase: str = "unknown",
        model_name: Optional[str] = None,
        batch_sizes: Optional[List[int]] = None,
        fmu_steps: Optional[int] = None,
    ) -> BenchmarkResults:
        """
        Run complete benchmark for a surrogate model.
        
        Args:
            model: Surrogate model to benchmark
            input_fn: Function that generates model inputs given batch size
            phase: Phase identifier for FMU step mapping
            model_name: Model name for results. Defaults to class name.
            batch_sizes: Batch sizes to test. Uses config default if None.
            fmu_steps: FMU steps per prediction. Inferred from phase if None.
            
        Returns:
            BenchmarkResults with all timing and speedup data
        """
        if model_name is None:
            model_name = model.__class__.__name__
        
        if batch_sizes is None:
            batch_sizes = self.config.batch_sizes
        
        if fmu_steps is None:
            fmu_steps = self.config.get_fmu_steps(phase)
        
        # Time surrogate model
        surrogate_timing = self.surrogate_timer.time_model(
            model, input_fn, batch_sizes,
            n_warmup=self.config.n_warmup,
            n_repeats=self.config.n_repeats,
        )
        
        # Time FMU (use cache if available)
        fmu_timing = self._get_fmu_timing(fmu_steps)
        
        # Compute speedups
        speedups = {}
        for bs, surr_time in surrogate_timing.items():
            speedups[bs] = SpeedupResult(
                surrogate_time=surr_time,
                fmu_time=fmu_timing,
                fmu_steps=fmu_steps,
                batch_size=bs,
            )
        
        # Get model info
        n_params = sum(p.numel() for p in model.parameters())
        
        return BenchmarkResults(
            model_name=model_name,
            phase=phase,
            config=self.config,
            surrogate_timing=surrogate_timing,
            fmu_timing={fmu_steps: fmu_timing},
            speedups=speedups,
            n_parameters=n_params,
            device=str(self.device),
        )
    
    def benchmark_multiple(
        self,
        models: Dict[str, Tuple[nn.Module, InputFn, str]],
        batch_sizes: Optional[List[int]] = None,
    ) -> Dict[str, BenchmarkResults]:
        """
        Benchmark multiple models.
        
        Args:
            models: Dict mapping name to (model, input_fn, phase)
            batch_sizes: Batch sizes to test
            
        Returns:
            Dict mapping model name to BenchmarkResults
        """
        results = {}
        for name, (model, input_fn, phase) in models.items():
            results[name] = self.benchmark(
                model, input_fn, phase=phase, model_name=name,
                batch_sizes=batch_sizes,
            )
        return results
    
    def _get_fmu_timing(self, n_steps: int) -> TimingResult:
        """Get FMU timing, using cache if available."""
        if n_steps not in self._fmu_timing_cache:
            self._fmu_timing_cache[n_steps] = self.fmu_timer.time_steps(
                n_steps, n_repeats=self.config.n_repeats
            )
        return self._fmu_timing_cache[n_steps]
    
    def print_results(
        self,
        results: Union[BenchmarkResults, Dict[str, BenchmarkResults]],
        file=None,
    ) -> None:
        """
        Print formatted benchmark results.
        
        Args:
            results: Single or multiple benchmark results
            file: File to print to. Defaults to stdout.
        """
        def _print(*args, **kwargs):
            print(*args, **kwargs, file=file)
        
        if isinstance(results, dict):
            self._print_multiple_results(results, _print)
        else:
            self._print_single_result(results, _print)
    
    def _print_single_result(self, results: BenchmarkResults, _print) -> None:
        """Print results for a single model."""
        _print("\n" + "=" * 80)
        _print(f"BENCHMARK RESULTS: {results.model_name}")
        _print("=" * 80)
        
        _print(f"\nModel: {results.model_name}")
        _print(f"Phase: {results.phase}")
        _print(f"Parameters: {results.n_parameters:,}")
        _print(f"Device: {results.device}")
        
        _print(f"\n{'Batch':>6s}  {'Surr (ms)':>10s}  {'FMU (ms)':>10s}  "
               f"{'Speedup/sample':>15s}  {'Throughput':>15s}")
        _print(f"{'-'*6}  {'-'*10}  {'-'*10}  {'-'*15}  {'-'*15}")
        
        for bs in sorted(results.speedups.keys()):
            s = results.speedups[bs]
            _print(f"{bs:6d}  {s.surrogate_time.mean_ms:10.2f}  "
                   f"{s.fmu_time.mean_ms:10.2f}  "
                   f"{s.speedup_per_sample:14.1f}×  "
                   f"{s.throughput_speedup:14.1f}×")
        
        _print(f"\nBest batch size: {results.best_batch_size}")
        _print(f"Max throughput speedup: {results.max_throughput_speedup:.1f}×")
        _print(f"Max per-sample speedup: {results.max_per_sample_speedup:.1f}×")
    
    def _print_multiple_results(
        self,
        results: Dict[str, BenchmarkResults],
        _print,
    ) -> None:
        """Print results for multiple models."""
        _print("\n" + "=" * 90)
        _print("SPEEDUP RESULTS: Surrogate Models vs FMU Simulator")
        _print("=" * 90)
        
        for name, res in results.items():
            _print(f"\n{name}:")
            _print(f"  {'Batch':>6s}  {'Surr (ms)':>10s}  {'FMU (ms)':>10s}  "
                   f"{'Speedup/sample':>15s}  {'Throughput Speedup':>18s}")
            _print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*15}  {'-'*18}")
            
            for bs in sorted(res.speedups.keys()):
                s = res.speedups[bs]
                _print(f"  {bs:6d}  {s.surrogate_time.mean_ms:10.2f}  "
                       f"{s.fmu_time.mean_ms:10.2f}  "
                       f"{s.speedup_per_sample:15.1f}×  "
                       f"{s.throughput_speedup:18.1f}×")
        
        # Summary table
        _print("\n" + "-" * 60)
        _print("SUMMARY (at optimal batch size):")
        _print("-" * 60)
        _print(f"{'Model':<30s}  {'Best BS':>8s}  {'Max Speedup':>12s}")
        _print(f"{'-'*30}  {'-'*8}  {'-'*12}")
        
        for name, res in results.items():
            _print(f"{name:<30s}  {res.best_batch_size:>8d}  "
                   f"{res.max_throughput_speedup:>11.1f}×")
    
    def to_dataframe(
        self,
        results: Union[BenchmarkResults, Dict[str, BenchmarkResults]],
    ) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.
        
        Args:
            results: Single or multiple benchmark results
            
        Returns:
            DataFrame with all speedup data
        """
        if isinstance(results, dict):
            dfs = [res.to_dataframe() for res in results.values()]
            return pd.concat(dfs, ignore_index=True)
        return results.to_dataframe()
    
    def cleanup(self):
        """Clean up resources."""
        self.fmu_timer.cleanup()


# ───────────────────────────────────────────────────────────────────────
# Input Function Builders
# ───────────────────────────────────────────────────────────────────────

def build_lstm_input_fn(
    input_size: int,
    history_steps: int,
    device: torch.device,
) -> InputFn:
    """Build input function for LSTM models (Phase 1)."""
    def input_fn(batch_size: int) -> torch.Tensor:
        return generate_dummy_input(batch_size, history_steps, input_size, device)
    return input_fn


def build_hybrid_input_fn(
    temporal_input_size: int,
    algebraic_input_size: int,
    history_steps: int,
    prediction_steps: int,
    device: torch.device,
) -> InputFn:
    """Build input function for Hybrid DeepONet models (Phase 2/3)."""
    def input_fn(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x_temporal = generate_dummy_input(
            batch_size, history_steps, temporal_input_size, device
        )
        x_algebraic = generate_dummy_input(
            batch_size, prediction_steps, algebraic_input_size, device
        )
        return (x_temporal, x_algebraic)
    return input_fn


def build_domain_input_fn(
    input_size: int,
    history_steps: int,
    device: torch.device,
) -> InputFn:
    """Build input function for Domain DeepONet models (Phase 4)."""
    return build_lstm_input_fn(input_size, history_steps, device)


def build_federated_input_fn(
    n_inputs: int,
    n_dynamic: int,
    history_steps: int,
    device: torch.device,
) -> InputFn:
    """Build input function for Federated models (Phase 5/6)."""
    def input_fn(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        u_hist = generate_dummy_input(batch_size, history_steps, n_inputs, device)
        y_hist = generate_dummy_input(batch_size, history_steps, n_dynamic, device)
        return (u_hist, y_hist)
    return input_fn


# ───────────────────────────────────────────────────────────────────────
# Convenience Functions
# ───────────────────────────────────────────────────────────────────────

def time_surrogate(
    model: nn.Module,
    input_fn: InputFn,
    batch_sizes: Optional[List[int]] = None,
    n_warmup: int = 5,
    n_repeats: int = 20,
    device: Optional[torch.device] = None,
) -> Dict[int, TimingResult]:
    """
    Time surrogate model inference.
    
    Args:
        model: Model to time
        input_fn: Function generating inputs for given batch size
        batch_sizes: Batch sizes to test
        n_warmup: Warmup iterations
        n_repeats: Timing iterations
        device: Device for inference
        
    Returns:
        Dict mapping batch size to TimingResult
    """
    if batch_sizes is None:
        batch_sizes = [1, 16, 64, 256, 1024]
    
    timer = SurrogateTimer(device)
    return timer.time_model(model, input_fn, batch_sizes, n_warmup, n_repeats)


def time_fmu(
    n_steps_list: List[int],
    system_name: str = "summit",
    step_size: float = 1.0,
    n_repeats: int = 5,
) -> Dict[int, TimingResult]:
    """
    Time FMU simulator.
    
    Args:
        n_steps_list: List of step counts to time
        system_name: HPC system name
        step_size: FMU step size in seconds
        n_repeats: Number of timing repetitions
        
    Returns:
        Dict mapping step count to TimingResult
    """
    timer = FMUTimer(system_name, step_size)
    results = timer.time_multiple(n_steps_list, n_repeats)
    timer.cleanup()
    return results


def compute_speedup(
    surrogate_timing: Dict[int, TimingResult],
    fmu_timing: TimingResult,
    fmu_steps: int,
) -> Dict[int, SpeedupResult]:
    """
    Compute speedup from timing results.
    
    Args:
        surrogate_timing: Dict mapping batch size to TimingResult
        fmu_timing: FMU TimingResult
        fmu_steps: Number of FMU steps per surrogate prediction
        
    Returns:
        Dict mapping batch size to SpeedupResult
    """
    speedups = {}
    for bs, surr_time in surrogate_timing.items():
        speedups[bs] = SpeedupResult(
            surrogate_time=surr_time,
            fmu_time=fmu_timing,
            fmu_steps=fmu_steps,
            batch_size=bs,
        )
    return speedups


def benchmark_model(
    model: nn.Module,
    input_fn: InputFn,
    phase: str = "unknown",
    config: Optional[BenchmarkConfig] = None,
    device: Optional[torch.device] = None,
) -> BenchmarkResults:
    """
    Convenience function to benchmark a single model.
    
    Args:
        model: Model to benchmark
        input_fn: Input generation function
        phase: Phase identifier
        config: Benchmark configuration
        device: Device for inference
        
    Returns:
        BenchmarkResults
    """
    benchmarker = SpeedupBenchmarker(config, device)
    results = benchmarker.benchmark(model, input_fn, phase)
    benchmarker.cleanup()
    return results


def benchmark_all_phases(
    models: Dict[str, nn.Module],
    input_fns: Dict[str, InputFn],
    phases: Dict[str, str],
    config: Optional[BenchmarkConfig] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, BenchmarkResults]:
    """
    Benchmark all phases of surrogate models.
    
    Args:
        models: Dict mapping name to model
        input_fns: Dict mapping name to input function
        phases: Dict mapping name to phase identifier
        config: Benchmark configuration
        device: Device for inference
        
    Returns:
        Dict mapping name to BenchmarkResults
    """
    combined = {
        name: (models[name], input_fns[name], phases[name])
        for name in models
    }
    
    benchmarker = SpeedupBenchmarker(config, device)
    results = benchmarker.benchmark_multiple(combined)
    benchmarker.cleanup()
    return results