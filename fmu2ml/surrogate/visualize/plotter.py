"""
Visualization utilities for surrogate models.

Provides comprehensive plotting for:
- Training curves (loss, learning rate, per-head losses)
- Evaluation metrics (R² distribution, variance ratio, per-type performance)
- Prediction analysis (time-series, error distributions)
- Benchmark results (speedup, latency heatmaps)
- Physics constraint evolution (Phase 6)

All public function and method names are preserved for backward compatibility.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

# Type aliases
PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, List]


# ───────────────────────────────────────────────────────────────────────
# Style and Configuration
# ───────────────────────────────────────────────────────────────────────

# Default color palettes
CATEGORY_COLORS = {
    'A (Primary loop)': 'steelblue',
    'B (Secondary temp)': 'coral',
    'C (Primary flow)': 'green',
    'D (Secondary pressure)': 'mediumpurple',
    'E (Constant)': 'goldenrod',
}

# Canonical category order 
CATEGORY_ORDER = [
    'A (Primary loop)',
    'B (Secondary temp)',
    'C (Primary flow)',
    'D (Secondary pressure)',
    'E (Constant)',
]

DOMAIN_COLORS = {
    'temperature': 'tab:red',
    'flow': 'tab:blue',
    'pressure': 'tab:green',
    'power': 'tab:orange',
}

GROUP_COLORS = {
    'G_T': 'steelblue',
    'G_V': 'coral',
    'G_p': 'green',
    'G_Vs': 'mediumpurple',
    'G_ps': 'goldenrod',
    'G_W': 'teal',
}

PHASE_COLORS = {
    'Phase 1: LSTM': '#1f77b4',
    'Phase 2: DeepONet': '#ff7f0e',
    'Phase 3: DeepONet+Fixes': '#2ca02c',
    'Phase 4: Domain-Specific': '#d62728',
    'Phase 5: Federated': '#9467bd',
    'Phase 6: Physics-Informed': '#8c564b',
}

GROUP_LABELS = {
    'G_T': 'G_T (Temperatures)',
    'G_V': 'G_V (Prim Flow)',
    'G_p': 'G_p (Prim Pressure)',
    'G_Vs': 'G_Vs (Sec Flow)',
    'G_ps': 'G_ps (Sec Pressure)',
    'G_W': 'G_W (Pump Power)',
}

PHASE_TITLE_MAP = {
    "1": "Phase 1: Baseline LSTM",
    "2": "Phase 2: Basic DeepONet",
    "3": "Phase 3: Hybrid DeepONet (with Fixes)",
    "4": "Phase 4: Domain-Specific DeepONet",
    "5": "Phase 5: Federated DeepM&Mnet",
    "6": "Phase 6: Physics-Informed Federated",
}

# Phase 6 tier organization
TIER1_NAMES = [
    'temp_ordering_primary', 'temp_ordering_secondary',
    'hx_feasibility_hot', 'hx_feasibility_cold',
    'pressure_drop_primary', 'pressure_drop_secondary',
    'pue_bounds', 'carnot_limit',
]
TIER2_NAMES = [
    'energy_conservation_secondary', 'energy_conservation_primary',
    'energy_balance_hx', 'approach_temp_hot', 'approach_temp_cold',
    'wet_bulb_constraint', 'temperature_bounds', 'pressure_bounds',
    'cop_bounds',
]
TIER3_NAMES = [
    'pump_power_consistency',
    'load_temperature_monotonicity',
    'external_temp_sensitivity',
]


def set_style(style: str = 'default') -> None:
    """Set matplotlib style for surrogate visualizations."""
    if style == 'paper':
        plt.rcParams.update({
            'font.size': 10,
            'axes.labelsize': 11,
            'axes.titlesize': 12,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'figure.dpi': 150,
        })
    elif style == 'presentation':
        plt.rcParams.update({
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'legend.fontsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.dpi': 100,
        })
    else:
        plt.rcParams.update(plt.rcParamsDefault)


def save_figure(
    fig: plt.Figure,
    path: PathLike,
    dpi: int = 150,
    bbox_inches: str = 'tight',
    **kwargs,
) -> None:
    """
    Save figure to file, creating directories if needed.
    
    Args:
        fig: Matplotlib figure
        path: Output path
        dpi: Resolution
        bbox_inches: Bounding box setting
        **kwargs: Additional savefig arguments
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)


# ───────────────────────────────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────────────────────────────

def _find_key(d: Dict, candidates: List[str]) -> Optional[str]:
    """Find first matching key from candidates."""
    for key in candidates:
        if key in d:
            return key
    return None


def _get_throughput_speedup(speedup_data) -> float:
    """Extract throughput speedup from various formats."""
    if isinstance(speedup_data, dict):
        return speedup_data.get('throughput_speedup', 0)
    return getattr(speedup_data, 'throughput_speedup', 0)


def _get_speedup_per_sample(speedup_data) -> float:
    """Extract per-sample speedup from various formats."""
    if isinstance(speedup_data, dict):
        return speedup_data.get('speedup_per_sample', speedup_data.get('speedup', 0))
    return getattr(speedup_data, 'speedup_per_sample', 0)


def _phase_color(phase_name: str) -> str:
    """Resolve a deterministic color for a phase name."""
    if phase_name in PHASE_COLORS:
        return PHASE_COLORS[phase_name]
    # Deterministic fallback via hash → tab10 cycle
    palette = list(matplotlib.colors.TABLEAU_COLORS.values())
    h = int(hashlib.md5(phase_name.encode('utf-8')).hexdigest(), 16)
    return palette[h % len(palette)]


def _match_type_columns(
    dynamic_cols: List[str],
    output_type: str,
    output_patterns: Dict[str, str],
    cdu_ids: List[int],
) -> List[int]:
    """
    Robustly map an output_type to indices in dynamic_cols using pattern.format().

    Falls back to substring match only when the pattern is missing.
    """
    indices: List[int] = []
    pattern = output_patterns.get(output_type, "") if output_patterns else ""
    if pattern and "{}" in pattern:
        for cdu_id in cdu_ids:
            col = pattern.format(cdu_id)
            if col in dynamic_cols:
                indices.append(dynamic_cols.index(col))
        return indices
    # Fallback: exact suffix match (avoid prefix collisions)
    for i, col in enumerate(dynamic_cols):
        if col.endswith(output_type) or col == output_type:
            indices.append(i)
    return indices


def _resolve_cdu_ids(
    cdu_ids: Optional[List[int]],
    config: Optional[Any],
    n_cdus: int,
) -> List[int]:
    """Resolve CDU IDs from explicit list, config, or sensible default."""
    if cdu_ids is not None:
        return list(cdu_ids[:n_cdus])
    if config is not None and hasattr(config, 'CDU_IDS'):
        return list(config.CDU_IDS[:n_cdus])
    return list(range(1, n_cdus + 1))


def _is_multi_domain_history(history: Dict) -> bool:
    """Detect whether `history` contains multiple per-domain histories."""
    if not history:
        return False
    sample_val = next(iter(history.values()))
    return isinstance(sample_val, dict)


def _get_r2_col(metrics_df: pd.DataFrame) -> str:
    return 'R²' if 'R²' in metrics_df.columns else 'R2'


def _get_persistence_col(metrics_df: pd.DataFrame) -> str:
    return 'Persistence_R²' if 'Persistence_R²' in metrics_df.columns else 'Persistence_R2'


def _resolve_type_col(metrics_df: pd.DataFrame, type_col: str) -> str:
    if type_col in metrics_df.columns:
        return type_col
    if 'Output_Type' in metrics_df.columns:
        return 'Output_Type'
    return 'Type'


# ───────────────────────────────────────────────────────────────────────
# Training Visualization
# ───────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List[float]],
    phase: str = "1",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Plot training and validation loss curves.
    
    Args:
        history: Training history dict with keys like 'train_loss', 'val_loss', 'lr'
        phase: Phase identifier for layout selection
        title: Optional figure title
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Determine layout based on phase
    if phase in ["5", "6"]:
        return _plot_training_curves_federated(history, phase, title, figsize, save_path)
    if phase in ["2", "3"]:
        return _plot_training_curves_hybrid(history, phase, title, figsize, save_path)
    if phase == "4":
        return _plot_training_curves_domain(history, title, figsize, save_path)
    return _plot_training_curves_simple(history, phase, title, figsize, save_path)


def _plot_training_curves_simple(
    history: Dict[str, List[float]],
    phase: str,
    title: Optional[str],
    figsize: Tuple[int, int],
    save_path: Optional[PathLike],
) -> plt.Figure:
    """Phase 1 layout: 1×2 (loss, lr) with notebook-style suptitle + summary text."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    train_key = _find_key(history, ['train_loss', 'train_total'])
    val_key = _find_key(history, ['val_loss', 'val_total'])

    if train_key:
        axes[0].plot(history[train_key], label='Train', linewidth=1.5, alpha=0.8)
    if val_key:
        axes[0].plot(history[val_key], label='Validation', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (Huber)')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    if 'lr' in history:
        axes[1].plot(history['lr'], linewidth=1.5, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule (Cosine Annealing)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # Notebook-style suptitle with epoch count + training time
    n_epochs = history.get('n_epochs_trained',
                           len(history.get(train_key, [])) if train_key else 0)
    train_time = history.get('train_time', None)
    if isinstance(n_epochs, list):
        n_epochs = len(n_epochs)
    phase_label = PHASE_TITLE_MAP.get(phase, f'Phase {phase}')

    if title:
        plt.suptitle(title, fontsize=13)
    else:
        if train_time is not None:
            plt.suptitle(
                f'{phase_label} — Training Details ({n_epochs} epochs, {train_time:.0f}s)',
                fontsize=13,
            )
        else:
            plt.suptitle(f'{phase_label} — Training Details ({n_epochs} epochs)',
                         fontsize=13)

    # Final summary text block (mirrors notebook's printed summary)
    if train_key and val_key and len(history[train_key]) and len(history[val_key]):
        final_train = history[train_key][-1]
        final_val = history[val_key][-1]
        best_val = min(history[val_key])
        ratio = final_val / (final_train + 1e-10)
        summary = (f'Final train: {final_train:.4f}  |  Final val: {final_val:.4f}  |  '
                   f'Best val: {best_val:.4f}  |  Val/Train: {ratio:.2f}')
        fig.text(0.5, -0.02, summary, ha='center', fontsize=9, color='dimgray')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _plot_training_curves_hybrid(
    history: Dict[str, List[float]],
    phase: str,
    title: Optional[str],
    figsize: Tuple[int, int],
    save_path: Optional[PathLike],
) -> plt.Figure:
    """Phase 2/3 layout: 1×3 (total, per-pathway, lr)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train_key = _find_key(history, ['train_total', 'train_loss'])
    val_key = _find_key(history, ['val_total', 'val_loss'])

    if train_key:
        axes[0].plot(history[train_key], label='Train', linewidth=1.5, alpha=0.8)
    if val_key:
        axes[0].plot(history[val_key], label='Validation', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    has_pathway = ('val_temporal' in history) or ('val_algebraic' in history)
    if has_pathway:
        if 'val_temporal' in history:
            axes[1].plot(history['val_temporal'], label='Temporal',
                         linewidth=1.5, alpha=0.8)
        if 'val_algebraic' in history:
            axes[1].plot(history['val_algebraic'], label='Algebraic',
                         linewidth=1.5, alpha=0.8)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Loss')
        axes[1].set_title('Loss by Pathway')
        axes[1].legend()
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    if 'lr' in history:
        axes[2].plot(history['lr'], linewidth=1.5, color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)

    phase_label = PHASE_TITLE_MAP.get(phase, f'Phase {phase}')
    plt.suptitle(title if title else f'{phase_label} — Training Details', fontsize=13)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _plot_training_curves_domain(
    history: Dict,
    title: Optional[str],
    figsize: Tuple[int, int],
    save_path: Optional[PathLike],
) -> plt.Figure:
    """Phase 4 layout: handles both single-history and multi-domain dict inputs."""
    if _is_multi_domain_history(history):
        histories = history
    else:
        histories = {'all': history}

    domain_names = list(histories.keys())
    n_domains = len(domain_names)
    fig, axes = plt.subplots(2, max(n_domains, 1),
                             figsize=(max(6 * n_domains, 8), 8), squeeze=False)

    for idx, domain_name in enumerate(domain_names):
        h = histories[domain_name]

        ax = axes[0, idx]
        train_key = _find_key(h, ['train_total', 'train_loss'])
        val_key = _find_key(h, ['val_total', 'val_loss'])
        if train_key:
            ax.plot(h[train_key], label='Train', linewidth=1.5, alpha=0.8)
        if val_key:
            ax.plot(h[val_key], label='Val', linewidth=1.5, alpha=0.8)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title(f'{str(domain_name).capitalize()} — Total Loss')
        ax.legend(fontsize=8); ax.set_yscale('log'); ax.grid(True, alpha=0.3)

        ax = axes[1, idx]
        if 'lr' in h:
            ax.plot(h['lr'], linewidth=1.5, color='green')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
        ax.set_title(f'{str(domain_name).capitalize()} — LR Schedule')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)

    plt.suptitle(title if title else 'Phase 4: Domain-Specific DeepONet Training Curves',
                 fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _plot_training_curves_federated(
    history: Dict[str, List[float]],
    phase: str,
    title: Optional[str],
    figsize: Tuple[int, int],
    save_path: Optional[PathLike],
) -> plt.Figure:
    """Phase 5/6 layout: 2×3 grid; physics panels conditional; stage shading."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    train_key = _find_key(history, ['train_total', 'train_loss', 'train_data'])
    val_key = _find_key(history, ['val_total', 'val_loss', 'val_data'])

    # Stage boundaries
    if phase == "6":
        stage_lengths = history.get('phase_epochs', (50, 50, 50))
        stage_colors = ['#e3f2fd', '#fff3e0', '#fce4ec']
        stage_labels = ['Phase 1: data-only', 'Phase 2: ramp physics',
                        'Phase 3: full physics']
    else:
        stage_lengths = history.get('phase_epochs', (50, 50))
        stage_colors = ['#e3f2fd', '#fff3e0']
        stage_labels = ['Stage 1: encoder warm-up', 'Stage 2: joint fine-tune']

    def _shade_stages(ax):
        cum = 0
        for length, color, label in zip(stage_lengths, stage_colors, stage_labels):
            ax.axvspan(cum, cum + length, color=color, alpha=0.4,
                       label=label if ax is axes[0, 0] else None)
            cum += length

    # (0,0) Total loss with stage shading
    ax = axes[0, 0]
    _shade_stages(ax)
    if train_key:
        ax.plot(history[train_key], label='Train', alpha=0.85, color='C0')
    if val_key:
        ax.plot(history[val_key], label='Val', alpha=0.85, color='C1')
    ax.set_title('Total Loss'); ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_yscale('log'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (0,1) Primary heads val loss
    ax = axes[0, 1]
    _shade_stages(ax)
    primary_heads = ['G_T', 'G_V', 'G_p']
    plotted = False
    for head in primary_heads:
        key = f'val_{head}'
        if key in history:
            ax.plot(history[key], label=head, alpha=0.8,
                    color=GROUP_COLORS.get(head, 'gray'))
            plotted = True
    ax.set_title('Primary Heads Val Loss'); ax.set_xlabel('Epoch')
    if plotted:
        ax.legend(fontsize=8); ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # (0,2) Near-constant heads val loss
    ax = axes[0, 2]
    _shade_stages(ax)
    skip_heads = ['G_Vs', 'G_ps', 'G_W']
    plotted = False
    for head in skip_heads:
        key = f'val_{head}'
        if key in history:
            ax.plot(history[key], label=head, alpha=0.8,
                    color=GROUP_COLORS.get(head, 'gray'))
            plotted = True
    ax.set_title('Near-Constant Heads Val Loss'); ax.set_xlabel('Epoch')
    if plotted:
        ax.legend(fontsize=8); ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # (1,0) Data + physics loss (Phase 6) OR per-head train loss (fallback)
    ax = axes[1, 0]
    _shade_stages(ax)
    if 'train_physics' in history:
        ax.plot(history.get('train_data', []), label='Train Data',
                alpha=0.8, color='blue')
        ax.plot(history.get('val_data', []), label='Val Data',
                alpha=0.8, color='blue', linestyle='--')
        ax.set_ylabel('Data Loss', color='blue')

        ax2 = ax.twinx()
        ax2.plot(history['train_physics'], label='Train Physics',
                 alpha=0.8, color='red')
        ax2.plot(history.get('val_physics', []), label='Val Physics',
                 alpha=0.8, color='red', linestyle='--')
        ax2.set_ylabel('Physics Loss', color='red')

        ax.set_title('Data + Physics Loss'); ax.set_xlabel('Epoch')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7)
    else:
        # Fallback: per-head training loss
        for head in primary_heads + skip_heads:
            key = f'train_{head}'
            if key in history:
                ax.plot(history[key], label=head, alpha=0.7,
                        color=GROUP_COLORS.get(head, 'gray'))
        ax.set_title('Per-Head Train Loss'); ax.set_xlabel('Epoch')
        ax.set_yscale('log'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (1,1) λ_physics schedule (Phase 6 only)
    ax = axes[1, 1]
    if 'physics_weight' in history and any(history['physics_weight']):
        _shade_stages(ax)
        ax.plot(history['physics_weight'], 'g-', linewidth=2, label='λ_physics')
        ax.set_title('Physics Weight Schedule')
        ax.set_xlabel('Epoch'); ax.set_ylabel('λ_physics')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    # (1,2) Learning rate
    ax = axes[1, 2]
    if 'lr' in history:
        _shade_stages(ax)
        ax.plot(history['lr'], linewidth=1.5, color='green')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule'); ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    else:
        ax.set_visible(False)

    phase_label = PHASE_TITLE_MAP.get(phase, f'Phase {phase}')
    plt.suptitle(title if title else f'{phase_label} — Training Details', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_learning_rate(
    history: Dict[str, List[float]],
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=figsize)
    if 'lr' in history:
        ax.plot(history['lr'], linewidth=2, color='green')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=13)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_loss_by_head(
    history: Dict[str, List[float]],
    heads: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot validation loss per decoder head."""
    if heads is None:
        heads = ['G_T', 'G_V', 'G_p', 'G_Vs', 'G_ps', 'G_W']

    fig, ax = plt.subplots(figsize=figsize)
    for head in heads:
        key = f'val_{head}'
        if key in history:
            ax.plot(history[key], label=head, alpha=0.8,
                    color=GROUP_COLORS.get(head, 'gray'))

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Loss', fontsize=12)
    ax.set_title('Validation Loss by Decoder Head', fontsize=13)
    ax.legend(); ax.set_yscale('log'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Evaluation Visualization
# ───────────────────────────────────────────────────────────────────────

def plot_r2_distribution(
    metrics_df: pd.DataFrame,
    group_by: str = 'Category',
    figsize: Tuple[int, int] = (18, 10),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Plot R² distribution histograms by category/group/domain.

    For `group_by='Category'`, uses the canonical 2×3 grid in fixed A–E order
    (matching Phase 1 notebook), with the overall histogram in the last slot.
    """
    r2_col = _get_r2_col(metrics_df)

    if group_by == 'Category':
        return _plot_r2_distribution_categories(metrics_df, r2_col, figsize, save_path)

    # Generic dynamic-layout path for non-category grouping
    if group_by not in metrics_df.columns:
        groups = ['All']
    else:
        groups = list(metrics_df[group_by].unique())

    if group_by == 'Domain':
        colors = DOMAIN_COLORS
    elif group_by == 'Group':
        colors = GROUP_COLORS
    else:
        colors = {}

    n_groups = len(groups)
    ncols = min(3, max(n_groups + 1, 1))
    nrows = (n_groups + 1 + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    last_idx = -1
    for idx, group in enumerate(groups):
        ax = axes[idx]
        last_idx = idx
        if group_by in metrics_df.columns:
            data = metrics_df[metrics_df[group_by] == group][r2_col]
        else:
            data = metrics_df[r2_col]
        if len(data) == 0:
            ax.set_visible(False); continue

        color = colors.get(group, 'steelblue')
        ax.hist(data, bins=min(30, len(data)), alpha=0.7,
                edgecolor='black', color=color)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1.5,
                   label=f'Mean: {data.mean():.4f}')
        ax.axvline(data.median(), color='orange', linestyle='--', linewidth=1.5,
                   label=f'Median: {data.median():.4f}')
        label = GROUP_LABELS.get(group, group)
        ax.set_title(f'{label} ({len(data)} outputs)', fontsize=10)
        ax.set_xlabel('R²'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Overall in next slot
    overall_idx = last_idx + 1
    if overall_idx < len(axes):
        ax = axes[overall_idx]
        ax.hist(metrics_df[r2_col], bins=30, alpha=0.7,
                edgecolor='black', color='gray')
        ax.axvline(metrics_df[r2_col].mean(), color='red', linestyle='--',
                   linewidth=1.5, label=f'Mean: {metrics_df[r2_col].mean():.4f}')
        ax.axvline(metrics_df[r2_col].median(), color='orange', linestyle='--',
                   linewidth=1.5, label=f'Median: {metrics_df[r2_col].median():.4f}')
        ax.set_title(f'All Outputs ({len(metrics_df)} total)', fontsize=10)
        ax.set_xlabel('R²'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        last_idx = overall_idx

    for j in range(last_idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f'R² Distribution by {group_by}', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _plot_r2_distribution_categories(
    metrics_df: pd.DataFrame,
    r2_col: str,
    figsize: Tuple[int, int],
    save_path: Optional[PathLike],
) -> plt.Figure:
    """Phase 1 canonical 2×3 layout for categories A–E + overall."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    if 'Category' in metrics_df.columns:
        for idx, cat in enumerate(CATEGORY_ORDER):
            ax = axes[idx]
            cat_data = metrics_df[metrics_df['Category'] == cat][r2_col]
            if len(cat_data) == 0:
                ax.set_visible(False); continue
            color = CATEGORY_COLORS[cat]
            ax.hist(cat_data, bins=min(20, len(cat_data)), alpha=0.7,
                    edgecolor='black', color=color)
            ax.axvline(cat_data.mean(), color='red', linestyle='--',
                       linewidth=1.5, label=f'Mean: {cat_data.mean():.4f}')
            ax.axvline(cat_data.median(), color='orange', linestyle='--',
                       linewidth=1.5, label=f'Median: {cat_data.median():.4f}')
            ax.set_title(f'{cat} ({len(cat_data)} outputs)', fontsize=11)
            ax.set_xlabel('R²'); ax.set_ylabel('Count')
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    else:
        for idx in range(5):
            axes[idx].set_visible(False)

    # Overall in slot 6
    ax = axes[5]
    ax.hist(metrics_df[r2_col], bins=30, alpha=0.7,
            edgecolor='black', color='gray')
    ax.axvline(metrics_df[r2_col].mean(), color='red', linestyle='--',
               linewidth=1.5, label=f'Mean: {metrics_df[r2_col].mean():.4f}')
    ax.axvline(metrics_df[r2_col].median(), color='orange', linestyle='--',
               linewidth=1.5, label=f'Median: {metrics_df[r2_col].median():.4f}')
    ax.set_title(f'All Outputs ({len(metrics_df)} total)', fontsize=11)
    ax.set_xlabel('R²'); ax.set_ylabel('Count')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle('R² Distribution by Output Category', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_r2_by_type(
    metrics_df: pd.DataFrame,
    type_col: str = 'Type',
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot mean R² by output type as bar chart."""
    r2_col = _get_r2_col(metrics_df)
    type_col = _resolve_type_col(metrics_df, type_col)

    output_types = list(metrics_df[type_col].unique())
    type_stats = metrics_df.groupby(type_col)[r2_col].agg(['mean', 'std']).reindex(output_types)

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(output_types))

    bars = ax.bar(x, type_stats['mean'], yerr=type_stats['std'], capsize=4,
                  color='steelblue', alpha=0.8, edgecolor='black')

    if color_by and color_by in metrics_df.columns:
        color_palette = {
            'Category': CATEGORY_COLORS,
            'Domain': DOMAIN_COLORS,
            'Group': GROUP_COLORS,
        }.get(color_by, {})
        # Robust mapping: groupby first non-null entry per type
        type_to_key = (metrics_df.dropna(subset=[color_by])
                                  .groupby(type_col)[color_by].first().to_dict())
        for i, otype in enumerate(output_types):
            key = type_to_key.get(otype, '')
            bars[i].set_facecolor(color_palette.get(key, 'gray'))

        # Inline category legend
        legend_handles = [Patch(facecolor=c, edgecolor='black', label=k)
                          for k, c in color_palette.items()
                          if k in type_to_key.values()]
        if legend_handles:
            ax.legend(handles=legend_handles, fontsize=8, loc='lower right',
                      title=color_by, framealpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(output_types, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('R² (mean ± std)', fontsize=12)
    ax.set_title('Mean R² by Output Type', fontsize=13)
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.5, label='R²=0.9')
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='R²=0.8')
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_r2_by_group(
    metrics_df: pd.DataFrame,
    group_col: str = 'Group',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot mean R² by decoder head group."""
    r2_col = _get_r2_col(metrics_df)
    if group_col not in metrics_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"Missing '{group_col}' column", ha='center', va='center')
        return fig

    canonical_order = ['G_T', 'G_V', 'G_p', 'G_Vs', 'G_ps', 'G_W']
    present = [g for g in canonical_order if g in metrics_df[group_col].values]
    extras = [g for g in metrics_df[group_col].unique() if g not in canonical_order]
    groups = present + extras

    group_stats = metrics_df.groupby(group_col)[r2_col].agg(['mean', 'std', 'count'])

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    means = [group_stats.loc[g, 'mean'] if g in group_stats.index else 0 for g in groups]
    stds = [group_stats.loc[g, 'std'] if g in group_stats.index else 0 for g in groups]
    colors = [GROUP_COLORS.get(g, 'gray') for g in groups]

    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.8, edgecolor='black')
    labels = [GROUP_LABELS.get(g, g) for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel('R² (mean ± std)', fontsize=12)
    ax.set_title('Mean R² by Decoder Head Group', fontsize=13)
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.5)
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_variance_ratio(
    metrics_df: pd.DataFrame,
    type_col: str = 'Type',
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot variance ratio by output type (Phase 1 styled)."""
    type_col = _resolve_type_col(metrics_df, type_col)
    output_types = list(metrics_df[type_col].unique())
    type_stats = (metrics_df.groupby(type_col)['Variance_Ratio']
                            .agg(['mean', 'std']).reindex(output_types))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(output_types))

    # Background shading for "acceptable" band [0.5, 1.5]
    ax.axhspan(0.5, 1.5, color='lightyellow', alpha=0.5, zorder=0,
               label='Acceptable band [0.5, 1.5]')

    ax.bar(x, type_stats['mean'], yerr=type_stats['std'], capsize=4,
           color='teal', alpha=0.8, edgecolor='black', zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(output_types, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Variance Ratio (Var(pred) / Var(actual))', fontsize=12)
    ax.set_title('Variance Ratio by Output Type', fontsize=13)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal (1.0)')
    ax.axhline(y=0.5, color='orange', linestyle=':', alpha=0.5,
               label='Conservative (0.5)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_rmse_by_type(
    metrics_df: pd.DataFrame,
    type_col: str = 'Type',
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot mean RMSE by output type as bar chart."""
    type_col = _resolve_type_col(metrics_df, type_col)
    output_types = list(metrics_df[type_col].unique())
    type_stats = (metrics_df.groupby(type_col)['RMSE']
                            .agg(['mean', 'std']).reindex(output_types))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(output_types))

    if color_by and color_by in metrics_df.columns:
        color_palette = {
            'Category': CATEGORY_COLORS, 'Domain': DOMAIN_COLORS, 'Group': GROUP_COLORS,
        }.get(color_by, {})
        type_to_key = (metrics_df.dropna(subset=[color_by])
                                  .groupby(type_col)[color_by].first().to_dict())
        colors = [color_palette.get(type_to_key.get(t, ''), 'indianred')
                  for t in output_types]
    else:
        colors = 'indianred'

    ax.bar(x, type_stats['mean'], yerr=type_stats['std'], color=colors,
           edgecolor='black', alpha=0.8, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(output_types, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('RMSE (physical units)', fontsize=12)
    ax.set_title('Mean RMSE by Output Type', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_model_vs_persistence(
    metrics_df: pd.DataFrame,
    type_col: str = 'Type',
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot Model R² vs Persistence R² and Skill Score."""
    type_col = _resolve_type_col(metrics_df, type_col)
    r2_col = _get_r2_col(metrics_df)
    pers_col = _get_persistence_col(metrics_df)
    output_types = list(metrics_df[type_col].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    x = np.arange(len(output_types))
    width = 0.35

    model_r2 = metrics_df.groupby(type_col)[r2_col].mean().reindex(output_types)
    pers_r2 = metrics_df.groupby(type_col)[pers_col].mean().reindex(output_types)

    ax1.bar(x - width/2, model_r2, width, label='Model', color='steelblue',
            edgecolor='black', alpha=0.8)
    ax1.bar(x + width/2, pers_r2, width, label='Persistence', color='gray',
            edgecolor='black', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(output_types, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('R²', fontsize=12)
    ax1.set_title('Model R² vs Persistence R²', fontsize=13)
    ax1.legend(fontsize=9);
    ax1.grid(True, alpha=0.3, axis='y')

    skill = metrics_df.groupby(type_col)['Skill_Score'].mean().reindex(output_types)
    colors_skill = ['green' if s > 0 else 'red' for s in skill]
    ax2.bar(x, skill, color=colors_skill, alpha=0.8, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(output_types, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Skill Score', fontsize=12)
    ax2.set_title('Skill Score (Model vs Persistence)', fontsize=13)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Model vs Persistence Baseline', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_correlation_by_type(
    metrics_df: pd.DataFrame,
    type_col: str = 'Type',
    color_by: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot mean Correlation Coefficient by output type."""
    type_col = _resolve_type_col(metrics_df, type_col)
    output_types = list(metrics_df[type_col].unique())
    type_stats = (metrics_df.groupby(type_col)['Correlation']
                            .agg(['mean', 'std']).reindex(output_types))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(output_types))

    if color_by and color_by in metrics_df.columns:
        color_palette = {
            'Category': CATEGORY_COLORS, 'Domain': DOMAIN_COLORS, 'Group': GROUP_COLORS,
        }.get(color_by, {})
        type_to_key = (metrics_df.dropna(subset=[color_by])
                                  .groupby(type_col)[color_by].first().to_dict())
        colors = [color_palette.get(type_to_key.get(t, ''), 'orchid')
                  for t in output_types]
    else:
        colors = 'orchid'

    ax.bar(x, type_stats['mean'], yerr=type_stats['std'], color=colors,
           edgecolor='black', alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(output_types, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Correlation Coefficient (ρ)', fontsize=12)
    ax.set_title('Mean Correlation by Output Type', fontsize=13)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, label='Perfect (1.0)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_loop_dichotomy(
    metrics_df: pd.DataFrame,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Primary vs secondary loop analysis (R², Skill, Beats Persistence, sorted R²)."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if 'Loop' not in metrics_df.columns:
        axes[0, 0].text(0.5, 0.5, "Missing 'Loop' column in metrics_df",
                        ha='center', va='center')
        for ax in axes.flatten()[1:]:
            ax.set_visible(False)
        plt.tight_layout()
        if save_path:
            save_figure(fig, save_path)
        return fig

    r2_col = _get_r2_col(metrics_df)
    primary_df = metrics_df[metrics_df['Loop'] == 'primary']
    secondary_df = metrics_df[metrics_df['Loop'] == 'secondary']

    # (a) R² boxplots
    ax = axes[0, 0]
    data_box = [primary_df[r2_col].dropna().values,
                secondary_df[r2_col].dropna().values]
    if len(data_box[0]) > 0 and len(data_box[1]) > 0:
        bp = ax.boxplot(data_box, labels=['Primary Loop', 'Secondary Loop'],
                        patch_artist=True, showmeans=True,
                        meanprops={'marker': 'D', 'markerfacecolor': 'red',
                                   'markersize': 8})
        bp['boxes'][0].set_facecolor('steelblue')
        bp['boxes'][1].set_facecolor('mediumpurple')
        for box in bp['boxes']:
            box.set_alpha(0.7)
    ax.set_ylabel('R²')
    ax.set_title('R² Distribution: Primary vs Secondary Loop')
    ax.grid(True, alpha=0.3)

    # (b) Mean Skill Score by Group
    ax = axes[0, 1]
    if 'Group' in metrics_df.columns:
        canonical = ['G_T', 'G_V', 'G_p', 'G_Vs', 'G_ps', 'G_W']
        groups = [g for g in canonical if g in metrics_df['Group'].unique()]
        if not groups:
            groups = list(metrics_df['Group'].unique())
        skill_by_group = [metrics_df[metrics_df['Group'] == g]['Skill_Score'].mean()
                          for g in groups]
        colors_bar = [GROUP_COLORS.get(g, 'steelblue') for g in groups]
        ax.bar(range(len(groups)), skill_by_group, color=colors_bar,
               edgecolor='black', alpha=0.8)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.set_ylabel('Mean Skill Score')
        ax.set_title('Mean Skill Score by Decoder Head')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.set_visible(False)

    # (c) Beats Persistence %
    ax = axes[1, 0]
    beats_prim = primary_df['Beats_Persistence'].mean() * 100 if len(primary_df) else 0
    beats_sec = secondary_df['Beats_Persistence'].mean() * 100 if len(secondary_df) else 0
    ax.bar(['Primary Loop', 'Secondary Loop'], [beats_prim, beats_sec],
           color=['steelblue', 'mediumpurple'], edgecolor='black', alpha=0.8)
    ax.set_ylabel('% Beating Persistence')
    ax.set_title('Percentage of Outputs Beating Persistence')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50%')
    for i, v in enumerate([beats_prim, beats_sec]):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.set_ylim(0, 110); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

    # (d) Sorted R² by output
    ax = axes[1, 1]
    sorted_df = metrics_df.sort_values(r2_col, ascending=False)
    sample_step = max(1, len(sorted_df) // 100)
    sampled = sorted_df.iloc[::sample_step]
    colors_sorted = ['steelblue' if l == 'primary' else 'mediumpurple'
                     for l in sampled['Loop']]
    ax.bar(range(len(sampled)), sampled[r2_col].values,
           color=colors_sorted, alpha=0.7)
    ax.set_xlabel('Output Index (sorted by R²)')
    ax.set_ylabel('R²')
    ax.set_title('Sorted R² by Output (Blue=Primary, Purple=Secondary)')
    ax.axhline(y=0.9, color='green', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Primary vs Secondary Loop Analysis', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_r2_heatmap(
    metrics_df: pd.DataFrame,
    cdu_col: str = 'CDU',
    type_col: str = 'Type',
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot R² heatmap across CDUs and output types."""
    r2_col = _get_r2_col(metrics_df)
    type_col = _resolve_type_col(metrics_df, type_col)

    pivot = metrics_df.pivot_table(values=r2_col, index=type_col,
                                   columns=cdu_col, aggfunc='mean')
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    n_cdus = len(pivot.columns)
    if n_cdus > 30:
        step = max(1, n_cdus // 30)
        ax.set_xticks(range(0, n_cdus, step))
        ax.set_xticklabels(pivot.columns[::step], rotation=90, fontsize=8)
    else:
        ax.set_xticks(range(n_cdus))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=8)

    ax.set_xlabel('CDU', fontsize=12)
    ax.set_ylabel('Output Type', fontsize=12)
    ax.set_title('R² Heatmap across CDUs', fontsize=13)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('R²', fontsize=10)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Time-Series Visualization
# ───────────────────────────────────────────────────────────────────────

def plot_timeseries_chained(
    predictions_dict: Dict[str, np.ndarray],
    dynamic_cols: List[str],
    output_types_list: List[str],
    output_patterns: Dict[str, str],
    prediction_steps: int = 1,
    subsample_factor: int = 30,
    n_cdus: int = 4,
    n_samples_show: int = 200,
    figsize_per_type: Tuple[int, int] = (6, 3),
    save_path: Optional[PathLike] = None,
    config: Optional[Any] = None,
    cdu_ids: Optional[List[int]] = None,
) -> plt.Figure:
    """
    Chain consecutive prediction windows into a continuous time-series.

    CDU selection: prefers explicit `cdu_ids`, then `config.CDU_IDS[:n_cdus]`,
    else range 1..n_cdus. Persistence is plotted as a constant per window
    (matching Phase 1 notebook behavior).
    """
    if 'pred_absolute' not in predictions_dict or 'target_absolute' not in predictions_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Absolute predictions/targets not found', ha='center')
        return fig

    pred = predictions_dict['pred_absolute']
    target = predictions_dict['target_absolute']
    last_dyn = predictions_dict.get('last_dynamic',
                                    predictions_dict.get('last_output', None))

    n_types = len(output_types_list)
    K = prediction_steps
    dt = subsample_factor
    N = min(n_samples_show, pred.shape[0])

    cdu_ids = _resolve_cdu_ids(cdu_ids, config, n_cdus)
    if not cdu_ids:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No CDUs to plot', ha='center')
        return fig

    fig, axes = plt.subplots(
        n_types, len(cdu_ids),
        figsize=(figsize_per_type[0] * len(cdu_ids), figsize_per_type[1] * n_types),
        squeeze=False,
    )

    for row, output_type in enumerate(output_types_list):
        pattern = output_patterns.get(output_type, "") if output_patterns else ""
        for col_idx, cdu_id in enumerate(cdu_ids):
            ax = axes[row, col_idx]
            col_name = pattern.format(cdu_id) if pattern and "{}" in pattern \
                else f"{output_type}_CDU{cdu_id}"
            if col_name not in dynamic_cols:
                ax.set_visible(False); continue
            idx = dynamic_cols.index(col_name)

            t_vals, y_actual, y_pred, y_persist = [], [], [], []
            for i in range(N):
                # Persistence is constant per window (notebook semantics)
                persist_val = (last_dyn[i, idx] if last_dyn is not None
                               and last_dyn.ndim == 2 else np.nan)
                for k in range(K):
                    t_sec = (i * K + k) * dt
                    t_vals.append(t_sec)
                    y_actual.append(target[i, k, idx])
                    y_pred.append(pred[i, k, idx])
                    y_persist.append(persist_val)

            t_min = np.array(t_vals) / 60.0
            y_actual = np.array(y_actual)
            y_pred = np.array(y_pred)
            y_persist = np.array(y_persist)

            ax.plot(t_min, y_actual, 'b-', label='Actual',
                    linewidth=0.8, alpha=0.9)
            ax.plot(t_min, y_pred, 'r-', label='Predicted',
                    linewidth=0.8, alpha=0.8)
            if last_dyn is not None and last_dyn.ndim == 2:
                ax.plot(t_min, y_persist, color='gray', linestyle=':',
                        label='Persistence', linewidth=0.6, alpha=0.5)

            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)

            ax.set_title(f'{output_type} CDU {cdu_id} (R²={r2:.4f})', fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(output_type.split('_')[0], fontsize=8)
            if row == n_types - 1:
                ax.set_xlabel('Time (min)', fontsize=8)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=6, loc='upper right')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Time-Series: Model vs Actual ({N} consecutive {K*dt}s windows)',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_timeseries_nearconst(
    predictions_dict: Dict[str, np.ndarray],
    dynamic_cols: List[str],
    nearconst_types: List[str],
    output_patterns: Dict[str, str],
    prediction_steps: int = 1,
    subsample_factor: int = 30,
    n_cdus: int = 4,
    n_samples_show: int = 200,
    figsize_per_type: Tuple[int, int] = (6, 3.5),
    save_path: Optional[PathLike] = None,
    config: Optional[Any] = None,
    cdu_ids: Optional[List[int]] = None,
) -> plt.Figure:
    """Time-series for near-constant outputs with ±3× range zoom."""
    if 'pred_absolute' not in predictions_dict or 'target_absolute' not in predictions_dict:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Absolute predictions/targets not found', ha='center')
        return fig

    pred = predictions_dict['pred_absolute']
    target = predictions_dict['target_absolute']
    last_dyn = predictions_dict.get('last_dynamic',
                                    predictions_dict.get('last_output', None))

    n_types = len(nearconst_types)
    if n_types == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No near-constant types provided', ha='center')
        return fig

    K = prediction_steps
    dt = subsample_factor
    N = min(n_samples_show, pred.shape[0])

    cdu_ids = _resolve_cdu_ids(cdu_ids, config, n_cdus)

    fig, axes = plt.subplots(
        n_types, len(cdu_ids),
        figsize=(figsize_per_type[0] * len(cdu_ids), figsize_per_type[1] * n_types),
        squeeze=False,
    )

    for row, output_type in enumerate(nearconst_types):
        pattern = output_patterns.get(output_type, "") if output_patterns else ""
        for col_idx, cdu_id in enumerate(cdu_ids):
            ax = axes[row, col_idx]
            col_name = pattern.format(cdu_id) if pattern and "{}" in pattern \
                else f"{output_type}_CDU{cdu_id}"
            if col_name not in dynamic_cols:
                ax.set_visible(False); continue
            idx = dynamic_cols.index(col_name)

            t_vals, y_actual, y_pred, y_persist = [], [], [], []
            for i in range(N):
                persist_val = (last_dyn[i, idx] if last_dyn is not None
                               and last_dyn.ndim == 2 else np.nan)
                for k in range(K):
                    t_sec = (i * K + k) * dt
                    t_vals.append(t_sec)
                    y_actual.append(target[i, k, idx])
                    y_pred.append(pred[i, k, idx])
                    y_persist.append(persist_val)

            t_min = np.array(t_vals) / 60.0
            y_actual = np.array(y_actual)
            y_pred = np.array(y_pred)
            y_persist = np.array(y_persist)

            ax.plot(t_min, y_actual, 'b-', label='Actual',
                    linewidth=0.8, alpha=0.9)
            ax.plot(t_min, y_pred, 'r-', label='Predicted',
                    linewidth=0.8, alpha=0.8)
            if last_dyn is not None and last_dyn.ndim == 2:
                ax.plot(t_min, y_persist, color='gray', linestyle=':',
                        label='Persistence', linewidth=0.6, alpha=0.5)

            # Phase 1 zoom: ±3× range
            y_mean = y_actual.mean()
            y_range = max(y_actual.max() - y_actual.min(), 1e-4)
            margin = y_range * 3
            ax.set_ylim(y_mean - margin, y_mean + margin)

            ss_res = np.sum((y_actual - y_pred) ** 2)
            ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)

            if last_dyn is not None and last_dyn.ndim == 2:
                ss_res_p = np.sum((y_actual - y_persist) ** 2)
                r2_p = 1 - ss_res_p / (ss_tot + 1e-10)
            else:
                r2_p = np.nan

            ax.set_title(f'{output_type} CDU {cdu_id}\n'
                         f'Model R²={r2:.4f}, Persist R²={r2_p:.4f}',
                         fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(output_type, fontsize=7)
            if row == n_types - 1:
                ax.set_xlabel('Time (min)', fontsize=8)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=6, loc='upper right')
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

    plt.suptitle(
        f'Near-Constant Outputs: Model vs Actual (zoomed, {N} windows)',
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_delta_quality(
    predictions_dict: Dict[str, np.ndarray],
    metrics_df: pd.DataFrame,
    dynamic_cols: List[str],
    output_types: List[str],
    type_col: str = 'Type',
    figsize: Tuple[int, int] = (14, 3),
    save_path: Optional[PathLike] = None,
    output_patterns: Optional[Dict[str, str]] = None,
    cdu_ids: Optional[List[int]] = None,
    config: Optional[Any] = None,
) -> plt.Figure:
    """
    Phase 1 delta-quality layout: per output type, two side-by-side panels:
    (left) overlaid histograms of step-0 actual vs predicted normalized deltas,
    (right) scatter of step-0 deltas with σ-ratio annotation.
    Picks ONE representative column per type (the first match).
    """
    if ('pred_normalized' not in predictions_dict
            or 'target_normalized' not in predictions_dict):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Normalized predictions/targets not found',
                ha='center', va='center')
        return fig

    pred_norm = predictions_dict['pred_normalized']
    tgt_norm = predictions_dict['target_normalized']

    # Resolve cdu_ids for pattern-based matching
    n_cdus_default = 1
    if config is not None and hasattr(config, 'CDU_IDS'):
        n_cdus_default = len(config.CDU_IDS)
    cdu_ids_resolved = _resolve_cdu_ids(cdu_ids, config, n_cdus_default)

    # Pick one representative column per type (the first match)
    type_to_col_idx: Dict[str, int] = {}
    for otype in output_types:
        idxs: List[int] = []
        if output_patterns:
            idxs = _match_type_columns(dynamic_cols, otype, output_patterns,
                                       cdu_ids_resolved)
        if not idxs:
            # Suffix fallback to avoid prefix collisions
            for i, col in enumerate(dynamic_cols):
                if col.endswith(otype) or col == otype:
                    idxs.append(i); break
        if idxs:
            type_to_col_idx[otype] = idxs[0]

    if not type_to_col_idx:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No matching columns for given output types',
                ha='center', va='center')
        return fig

    n_types = len(type_to_col_idx)
    fig, axes = plt.subplots(n_types, 2,
                             figsize=(figsize[0], figsize[1] * n_types),
                             squeeze=False)

    rng = np.random.default_rng(42)

    for row, (otype, col_idx) in enumerate(type_to_col_idx.items()):
        # Step-0 deltas only (Phase 1 semantics)
        pred_d = pred_norm[:, 0, col_idx]
        true_d = tgt_norm[:, 0, col_idx]

        # Histogram
        ax = axes[row, 0]
        ax.hist(true_d, bins=50, alpha=0.5, density=True,
                color='blue', label='Actual Delta')
        ax.hist(pred_d, bins=50, alpha=0.5, density=True,
                color='red', label='Predicted Delta')
        ax.set_title(f'{otype}: Delta Distribution', fontsize=10)
        ax.set_xlabel('Normalized Delta')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Scatter
        ax = axes[row, 1]
        n_plot = min(3000, len(pred_d))
        plot_idx = rng.choice(len(pred_d), n_plot, replace=False)
        ax.scatter(true_d[plot_idx], pred_d[plot_idx], alpha=0.2, s=5)
        lim = max(np.abs(true_d).max(), np.abs(pred_d).max(), 1e-8)
        ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)

        std_ratio = np.std(pred_d) / (np.std(true_d) + 1e-10)
        ax.set_title(f'{otype}: Delta Scatter (σ ratio={std_ratio:.3f})',
                     fontsize=10)
        ax.set_xlabel('Actual Delta')
        ax.set_ylabel('Predicted Delta')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Delta Prediction Quality', fontsize=13)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Prediction Visualization
# ───────────────────────────────────────────────────────────────────────

def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_samples: int = 5,
    output_indices: Optional[List[int]] = None,
    output_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot prediction vs ground truth time series for selected samples."""
    N, K, n_outputs = predictions.shape

    if output_indices is None:
        output_indices = list(range(min(4, n_outputs)))
    if output_names is None:
        output_names = [f'Output {i}' for i in output_indices]

    n_outputs_plot = len(output_indices)
    sample_indices = np.linspace(0, max(N - 1, 0), n_samples, dtype=int)

    fig, axes = plt.subplots(n_outputs_plot, n_samples, figsize=figsize)
    if n_outputs_plot == 1:
        axes = np.atleast_2d(axes)
    if n_samples == 1:
        axes = axes.reshape(-1, 1)

    time_steps = np.arange(K)

    for i, out_idx in enumerate(output_indices):
        for j, sample_idx in enumerate(sample_indices):
            ax = axes[i, j]
            pred = predictions[sample_idx, :, out_idx]
            target = targets[sample_idx, :, out_idx]

            ax.plot(time_steps, target, 'b-', label='Target',
                    linewidth=2, alpha=0.8)
            ax.plot(time_steps, pred, 'r--', label='Prediction',
                    linewidth=2, alpha=0.8)

            if i == 0:
                ax.set_title(f'Sample {sample_idx}', fontsize=10)
            if j == 0:
                ax.set_ylabel(output_names[i], fontsize=10)
            if i == n_outputs_plot - 1:
                ax.set_xlabel('Step', fontsize=9)
            ax.grid(True, alpha=0.3)
            if i == 0 and j == n_samples - 1:
                ax.legend(fontsize=8)

    plt.suptitle('Predictions vs Ground Truth', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_error_distribution(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot error distribution histograms per output."""
    n_outputs = predictions.shape[2]

    if output_names is None:
        output_names = [f'Output {i}' for i in range(n_outputs)]
    else:
        output_names = list(output_names)
        if len(output_names) < n_outputs:
            output_names.extend([f'Output {i}' for i in range(len(output_names), n_outputs)])
        elif len(output_names) > n_outputs:
            output_names = output_names[:n_outputs]

    ncols = min(4, n_outputs)
    nrows = (n_outputs + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_outputs):
        ax = axes[i]
        errors = (predictions[:, :, i] - targets[:, :, i]).flatten()
        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        ax.axvline(errors.mean(), color='orange', linestyle='--', alpha=0.7,
                   label=f'Mean: {errors.mean():.4f}')
        ax.set_title(output_names[i], fontsize=10)
        ax.set_xlabel('Error'); ax.set_ylabel('Count')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    for j in range(n_outputs, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Prediction Error Distribution', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# def plot_prediction_vs_target(
#     predictions: np.ndarray,
#     targets: np.ndarray,
#     output_indices: Optional[List[int]] = None,
#     output_names: Optional[List[str]] = None,
#     figsize: Tuple[int, int] = (14, 10),
#     save_path: Optional[PathLike] = None,
# ) -> plt.Figure:
#     """Parity plots (predicted vs target). Subsample cap = 5000 (Phase 1)."""
#     n_outputs = predictions.shape[2]

#     if output_indices is None:
#         output_indices = list(range(min(6, n_outputs)))
#     if output_names is None:
#         output_names = [f'Output {i}' for i in output_indices]

#     n_plots = len(output_indices)
#     ncols = min(3, n_plots)
#     nrows = (n_plots + ncols - 1) // ncols

#     fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
#     axes = np.atleast_1d(axes).flatten()
#     rng = np.random.default_rng(42)

#     for i, out_idx in enumerate(output_indices):
#         ax = axes[i]
#         pred = predictions[:, :, out_idx].flatten()
#         target = targets[:, :, out_idx].flatten()

#         # Phase 1: cap at 5000 samples
#         n_plot = min(5000, len(pred))
#         if n_plot < len(pred):
#             indices = rng.choice(len(pred), n_plot, replace=False)
#             pred_sub = pred[indices]; target_sub = target[indices]
#         else:
#             pred_sub = pred; target_sub = target

#         ax.scatter(target_sub, pred_sub, alpha=0.2, s=5, c='steelblue')
#         vmin = min(target.min(), pred.min())
#         vmax = max(target.max(), pred.max())
#         ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='Perfect')

#         ss_res = np.sum((target - pred) ** 2)
#         ss_tot = np.sum((target - target.mean()) ** 2)
#         r2 = 1 - ss_res / (ss_tot + 1e-10)
#         ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95),
#                     xycoords='axes fraction', fontsize=10,
#                     verticalalignment='top')

#         ax.set_xlabel('Target', fontsize=10)
#         ax.set_ylabel('Prediction', fontsize=10)
#         ax.set_title(output_names[i], fontsize=11)
#         ax.grid(True, alpha=0.3)

#     for j in range(n_plots, len(axes)):
#         axes[j].set_visible(False)

#     plt.suptitle('Prediction vs Target', fontsize=14)
#     plt.tight_layout()
#     if save_path:
#         save_figure(fig, save_path)
#     return fig

def plot_prediction_vs_target(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_indices: Optional[List[int]] = None,
    output_names: Optional[List[str]] = None,
    output_groups: Optional[Dict[str, List[int]]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[PathLike] = None,
    suptitle: str = 'Prediction vs Target',
) -> plt.Figure:
    """Parity plots (predicted vs target). Subsample cap = 5000 (Phase 1).

    Either pass `output_indices` (one subplot per output channel) or
    `output_groups` (one subplot per named group of channels, e.g. by type).
    """
    n_outputs = predictions.shape[2]

    # Build the list of (name, column_indices) pairs to plot
    if output_groups is not None:
        groups = [(name, idxs) for name, idxs in output_groups.items() if idxs]
    else:
        if output_indices is None:
            output_indices = list(range(min(6, n_outputs)))
        if output_names is None:
            output_names = [f'Output {i}' for i in output_indices]
        groups = [(name, [idx]) for name, idx in zip(output_names, output_indices)]

    n_plots = len(groups)
    ncols = min(3, n_plots)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()
    rng = np.random.default_rng(42)

    for i, (name, col_indices) in enumerate(groups):
        ax = axes[i]
        pred = predictions[:, :, col_indices].flatten()
        target = targets[:, :, col_indices].flatten()

        # Phase 1: cap at 5000 samples
        n_plot = min(5000, len(pred))
        if n_plot < len(pred):
            sub = rng.choice(len(pred), n_plot, replace=False)
            pred_sub, target_sub = pred[sub], target[sub]
        else:
            pred_sub, target_sub = pred, target

        ax.scatter(target_sub, pred_sub, alpha=0.2, s=5, c='steelblue')
        vmin = min(target.min(), pred.min())
        vmax = max(target.max(), pred.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='Perfect')

        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - target.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        ax.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95),
                    xycoords='axes fraction', fontsize=10,
                    verticalalignment='top')
        ax.set_xlabel('Target', fontsize=10)
        ax.set_ylabel('Prediction', fontsize=10)
        ax.set_title(name, fontsize=11)
        ax.grid(True, alpha=0.3)

    for j in range(n_plots, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Benchmark Visualization
# ───────────────────────────────────────────────────────────────────────

def plot_speedup(
    speedup_results: Dict[str, Dict[int, Dict[str, float]]],
    figsize: Tuple[int, int] = (18, 7),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Plot per-sample and throughput speedup vs batch size."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # All distinct batch sizes across phases (linear ticks)
    all_bs = sorted({bs for sp in speedup_results.values() for bs in sp.keys()})

    # Per-sample
    ax = axes[0]
    for phase_name, speedups in speedup_results.items():
        bs_list = sorted(speedups.keys())
        vals = [_get_speedup_per_sample(speedups[bs]) for bs in bs_list]
        color = _phase_color(phase_name)
        ax.plot(bs_list, vals, 'o-', label=phase_name.replace('Phase ', 'P'),
                color=color, linewidth=2, markersize=8)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('Batch Size', fontsize=12); ax.set_ylabel('Speedup (×)', fontsize=12)
    ax.set_title('Speedup per Sample vs FMU Simulator', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
    if all_bs:
        ax.set_xticks(all_bs)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    # Throughput
    ax = axes[1]
    for phase_name, speedups in speedup_results.items():
        bs_list = sorted(speedups.keys())
        vals = [_get_throughput_speedup(speedups[bs]) for bs in bs_list]
        color = _phase_color(phase_name)
        ax.plot(bs_list, vals, 's-', label=phase_name.replace('Phase ', 'P'),
                color=color, linewidth=2, markersize=8)
    ax.set_xscale('log', base=2); ax.set_yscale('log')
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Throughput Speedup (×)', fontsize=12)
    ax.set_title('Throughput Speedup vs FMU Simulator', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, which='both')
    if all_bs:
        ax.set_xticks(all_bs)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.suptitle('Surrogate Model Speedup over FMU Simulator',
                 fontsize=15, y=1.02)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_latency_heatmap(
    timing_results: Dict[str, Dict[int, Dict[str, float]]],
    batch_sizes: List[int],
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Per-sample latency heatmap (ms)."""
    fig, ax = plt.subplots(figsize=figsize)
    phase_order = list(timing_results.keys())

    heatmap_data, row_labels = [], []
    col_labels = [str(bs) for bs in batch_sizes]

    for phase_name in phase_order:
        row = []
        for bs in batch_sizes:
            timing = timing_results[phase_name].get(bs, {})
            if isinstance(timing, dict):
                per_sample = timing.get('per_sample', 0) * 1000
            else:
                per_sample = getattr(timing, 'per_sample', 0) * 1000
            row.append(per_sample if per_sample > 0 else np.nan)
        heatmap_data.append(row)
        row_labels.append(phase_name.replace('Phase ', 'P'))

    heatmap_arr = np.array(heatmap_data)
    valid_vals = heatmap_arr[~np.isnan(heatmap_arr)]
    norm = (LogNorm(vmin=max(0.001, np.nanmin(valid_vals)),
                    vmax=np.nanmax(valid_vals))
            if len(valid_vals) > 0 else None)

    im = ax.imshow(heatmap_arr, aspect='auto', cmap='YlOrRd_r', norm=norm)

    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel('Batch Size', fontsize=12)
    ax.set_ylabel('Model Phase', fontsize=12)
    ax.set_title('Per-Sample Latency (ms) — Lower is Better', fontsize=13)

    if len(valid_vals) > 0:
        median_val = np.nanmedian(heatmap_arr)
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = heatmap_arr[i, j]
                if not np.isnan(val):
                    color = 'white' if val > median_val else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                            fontsize=9, color=color, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Per-sample latency (ms)', fontsize=10)

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_speedup_summary(
    speedup_results: Dict[str, Dict[int, Dict[str, float]]],
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """Speedup summary bar chart at optimal batch size per phase."""
    fig, ax = plt.subplots(figsize=figsize)
    summaries = []

    for phase_name, speedups in speedup_results.items():
        if not speedups:
            continue
        best_bs = max(speedups.keys(),
                      key=lambda bs: _get_throughput_speedup(speedups[bs]))
        best = speedups[best_bs]
        summaries.append({
            'Phase': phase_name.replace('Phase ', 'P'),
            'Best Batch Size': best_bs,
            'Speedup (per sample)': _get_speedup_per_sample(best),
            'Throughput Speedup': _get_throughput_speedup(best),
        })

    if not summaries:
        ax.text(0.5, 0.5, 'No speedup data', ha='center', va='center')
        return fig

    summary_df = pd.DataFrame(summaries)
    x = np.arange(len(summary_df)); width = 0.4

    ax.bar(x - width/2, summary_df['Speedup (per sample)'], width,
           label='Per-Sample Speedup', color='steelblue',
           edgecolor='black', alpha=0.8)
    ax.bar(x + width/2, summary_df['Throughput Speedup'], width,
           label='Throughput Speedup', color='coral',
           edgecolor='black', alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df['Phase'], fontsize=10)
    ax.set_ylabel('Speedup (×) over FMU Simulator', fontsize=12)
    ax.set_title('Maximum Speedup at Optimal Batch Size', fontsize=14)
    ax.legend(fontsize=11); ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    for i, row in summary_df.iterrows():
        ax.annotate(f'{row["Throughput Speedup"]:.0f}×\n(BS={row["Best Batch Size"]})',
                    xy=(i + width/2, row['Throughput Speedup']),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.annotate(f'{row["Speedup (per sample)"]:.0f}×',
                    xy=(i - width/2, row['Speedup (per sample)']),
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='darkblue')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Physics Visualization (Phase 6)
# ───────────────────────────────────────────────────────────────────────

def plot_physics_evolution(
    history: Dict[str, Any],
    phase_epochs: Tuple[int, ...] = (50, 50, 50),
    figsize: Tuple[int, int] = (18, 12),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Plot physics constraint evolution during training (Phase 6).

    Reads both flat `train_phys_*` keys AND nested
    `physics_per_constraint`/`monitoring` dict-of-lists. Groups panels by tier:
    Tier 1 (hard) → row 1, Tier 2 (soft) → row 2, Tier 3 (monitoring) → row 3.
    Overlays λ_physics on a twin axis when `history['physics_weight']` exists.
    """
    # Collect per-constraint series
    train_series: Dict[str, List[float]] = {}
    monitor_series: Dict[str, List[float]] = {}

    # Flat keys
    for k, v in history.items():
        if k.startswith('train_phys_'):
            train_series[k.replace('train_phys_', '')] = list(v)

    # Nested dicts
    nested_train = history.get('physics_per_constraint', {})
    if isinstance(nested_train, dict):
        for k, v in nested_train.items():
            if k not in train_series:
                train_series[k] = list(v)
    nested_monitor = history.get('monitoring', {})
    if isinstance(nested_monitor, dict):
        for k, v in nested_monitor.items():
            monitor_series[k] = list(v)

    if not train_series and not monitor_series:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No physics constraint data recorded',
                ha='center', va='center', fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        if save_path:
            save_figure(fig, save_path)
        return fig

    # Order constraints by tier
    tier1 = [n for n in TIER1_NAMES if n in train_series]
    tier2 = [n for n in TIER2_NAMES if n in train_series]
    others = [n for n in train_series.keys()
              if n not in TIER1_NAMES and n not in TIER2_NAMES]
    tier2 = tier2 + others
    tier3 = [n for n in TIER3_NAMES if n in monitor_series]
    extras_monitor = [n for n in monitor_series.keys() if n not in TIER3_NAMES]
    tier3 = tier3 + extras_monitor

    rows: List[Tuple[str, List[str], Dict[str, List[float]], str]] = []
    if tier1:
        rows.append(('Tier 1: Hard Constraints', tier1, train_series, 'white'))
    if tier2:
        rows.append(('Tier 2: Soft Constraints', tier2, train_series, 'white'))
    if tier3:
        rows.append(('Tier 3: Monitor Only (no gradient)',
                     tier3, monitor_series, '#f0f0f0'))

    ncols = max(max(len(r[1]) for r in rows), 1)
    nrows = len(rows)

    # Phase boundaries
    boundaries = []
    cum = 0
    for L in phase_epochs:
        cum += L
        boundaries.append(cum)
    # Drop the trailing one (training end)
    if boundaries:
        boundaries = boundaries[:-1]

    physics_weight = history.get('physics_weight', None)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)

    for r_idx, (row_title, names, series_map, bg_color) in enumerate(rows):
        for c_idx in range(ncols):
            ax = axes[r_idx, c_idx]
            if c_idx >= len(names):
                ax.set_visible(False); continue

            name = names[c_idx]
            data = series_map.get(name, [])
            epochs = list(range(len(data)))

            ax.set_facecolor(bg_color)
            ax.plot(epochs, data, '-', linewidth=1.5, alpha=0.8,
                    color='C3' if 'Tier 3' in row_title else 'C0')

            # Overlay λ_physics on twin axis when available
            if physics_weight and len(physics_weight) == len(data) and len(data):
                ax2 = ax.twinx()
                ax2.plot(epochs, physics_weight, color='green',
                         linewidth=1.0, alpha=0.5, linestyle='--')
                ax2.set_ylabel('λ_physics', fontsize=7, color='green')
                ax2.tick_params(axis='y', labelsize=6, colors='green')

            for b in boundaries:
                ax.axvline(b, color='gray', linestyle='--', alpha=0.5)

            pretty = name.replace('_', ' ').title()
            ax.set_title(pretty, fontsize=9)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if c_idx == 0:
                ax.set_ylabel(row_title, fontsize=9, fontweight='bold')

    plt.suptitle('Physics Constraint Evolution During Training', fontsize=14)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Phase Comparison
# ───────────────────────────────────────────────────────────────────────

def plot_phase_comparison(
    phase_metrics: Union[Dict[str, Dict[str, float]], List[pd.DataFrame]],
    metric: Union[str, List[str]] = 'R2',
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Plot comparison of metrics across phases.

    Supports two input shapes:
    1) Dict[phase_name, Dict[metric_name, value]] — original API
    2) List[pd.DataFrame] of metrics_dfs + matching `labels` list — convenience
       form that aggregates each df's metric to its mean.

    `metric` accepts a single name (str) or a list of names. With a list,
    a multi-panel figure (one panel per metric) is produced.
    """
    # Normalize list-of-dataframes input → dict form
    if isinstance(phase_metrics, list):
        if labels is None or len(labels) != len(phase_metrics):
            labels = [f'Phase {i+1}' for i in range(len(phase_metrics))]
        agg: Dict[str, Dict[str, float]] = {}
        metric_list = [metric] if isinstance(metric, str) else list(metric)
        for lbl, df in zip(labels, phase_metrics):
            entry: Dict[str, float] = {}
            for m in metric_list:
                col = m if m in df.columns else (
                    'R²' if m in ('R2', 'R²') and 'R²' in df.columns else None
                )
                if col is None:
                    entry[m] = float('nan')
                else:
                    entry[m] = float(df[col].mean())
            agg[lbl] = entry
        phase_metrics = agg

    metric_list = [metric] if isinstance(metric, str) else list(metric)
    n_metrics = len(metric_list)
    phases = list(phase_metrics.keys())
    colors = [_phase_color(p) for p in phases]
    x = np.arange(len(phases))

    fig, axes = plt.subplots(1, n_metrics,
                             figsize=(figsize[0] * max(n_metrics, 1), figsize[1]),
                             squeeze=False)
    axes = axes.flatten()

    for i, m in enumerate(metric_list):
        ax = axes[i]
        # Resolve possible R²/R2 alias
        values = []
        for p in phases:
            v = phase_metrics[p].get(m, None)
            if v is None and m in ('R2', 'R²'):
                v = phase_metrics[p].get('R²' if m == 'R2' else 'R2', 0)
            values.append(v if v is not None else 0)

        bars = ax.bar(x, values, color=colors, edgecolor='black', alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace('Phase ', 'P') for p in phases], fontsize=10)
        ax.set_ylabel(m, fontsize=12)
        ax.set_title(f'{m} Comparison Across Phases', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, val),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Uncertainty Quantification Visualization (MC dropout / PI3NN)
# ───────────────────────────────────────────────────────────────────────

# Presentation colour convention for UQ traces (shared by all band plots).
UQ_ACTUAL_COLOR = '#222222'
UQ_MEAN_COLOR = 'tab:orange'
UQ_BAND_COLOR = 'tab:orange'
UQ_MISS_COLOR = 'tab:red'
UQ_PERSIST_COLOR = 'gray'


def _uq_arrays(predictions_dict: Dict[str, np.ndarray]):
    """
    Pull (mean, target, sigma, lower, upper) from a UQ predictions_dict.

    Supports both Gaussian (``pred_std``) and interval (``pred_lower``/
    ``pred_upper``) representations; returns ``None`` for absent pieces.
    """
    mean = predictions_dict.get('pred_mean', predictions_dict.get('pred_absolute'))
    target = predictions_dict.get('target_absolute')
    sigma = predictions_dict.get('pred_std')
    lower = predictions_dict.get('pred_lower')
    upper = predictions_dict.get('pred_upper')
    return mean, target, sigma, lower, upper


def _resolve_band(mean, sigma, lower, upper, n_sigma, recal_scale):
    """Resolve a prediction band as (lo, hi) from std or explicit bounds."""
    if lower is not None and upper is not None:
        return lower, upper
    if sigma is not None:
        half = n_sigma * recal_scale * sigma
        return mean - half, mean + half
    return None, None


def _uq_bands(predictions_dict, n_sigma=2.0, recal_scale=1.0):
    """
    Fully resolve (mean, target, lower, upper, sigma) arrays for a UQ dict.

    Works for both the Gaussian (``pred_std``) and interval (``pred_lower``/
    ``pred_upper``) cases. ``lower``/``upper`` are always returned when any
    uncertainty is present; ``sigma`` is the half-width / n_sigma proxy when only
    intervals are available. Returns ``None`` for mean/target if absent.
    """
    mean, target, sigma, lower, upper = _uq_arrays(predictions_dict)
    if mean is None:
        return None, None, None, None, None
    lo, hi = _resolve_band(mean, sigma, lower, upper, n_sigma, recal_scale)
    if sigma is None and lo is not None:
        sigma = (hi - lo) / (2.0 * n_sigma)
    return mean, target, lo, hi, sigma


def _chain_series(arr3d, idx, n, k):
    """Flatten ``arr3d[:n, :, idx]`` (shape (n, K)) into a continuous trace.

    Returns the 1-D values; the caller pairs them with a time vector. Vectorised
    replacement for the old nested ``for i / for k`` loops.
    """
    return np.asarray(arr3d[:n, :, idx]).reshape(-1)


def _time_minutes(n, k, subsample_factor):
    """Physical-time axis (minutes) for ``n`` chained windows of ``k`` steps."""
    return (np.arange(n * k) * subsample_factor) / 60.0


def _spearman(x, y):
    """Spearman rank correlation without scipy (NaN-safe)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float('nan')
    rx = np.argsort(np.argsort(x[m])); ry = np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1])


def calibration_error(pred_mean, pred_std, target, levels=None, recal_scale=1.0):
    """Expected calibration error = mean |empirical − nominal| over coverage levels."""
    lv, emp = coverage_curve(pred_mean, pred_std, target,
                             levels=levels, recal_scale=recal_scale)
    return float(np.mean(np.abs(emp - lv)))


def _group_columns(column_info):
    """Return an ordered dict {group_name: [column names]} for federated heads.

    Falls back to grouping ``output_cols`` by ``col_to_type`` when the federated
    ``*_cols`` keys are absent (phases 1–4).
    """
    fed_keys = [('G_T', 'temp_cols'), ('G_V', 'flow_cols'), ('G_p', 'pressure_cols'),
                ('G_Vs', 'flow_sec_cols'), ('G_ps', 'pressure_sec_cols'),
                ('G_W', 'power_cols')]
    out = {}
    if any(k in column_info for _, k in fed_keys):
        for name, key in fed_keys:
            cols = list(column_info.get(key, []))
            if cols:
                out[name] = cols
        return out
    # Non-federated: group by output type
    cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
    col_to_type = column_info.get('col_to_type', {})
    for c in cols:
        out.setdefault(col_to_type.get(c, 'all'), []).append(c)
    return out


def _draw_band_panel(ax, t_min, y_true, y_pred, y_lo, y_hi, n_sigma,
                     is_interval, persist=None, show_legend=False, title=None):
    """Draw one actual/mean/±band panel with miss highlighting + annotations."""
    inside = (y_true >= y_lo) & (y_true <= y_hi)
    coverage = float(np.mean(inside)) if len(y_true) else float('nan')
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if len(y_true) else float('nan')
    band_label = 'PI' if is_interval else f'±{n_sigma:g}σ'

    ax.fill_between(t_min, y_lo, y_hi, color=UQ_BAND_COLOR, alpha=0.25,
                    linewidth=0, label=f'{band_label} band')
    if persist is not None:
        ax.plot(t_min, persist, color=UQ_PERSIST_COLOR, linestyle=':',
                linewidth=0.8, alpha=0.6, label='Persistence')
    ax.plot(t_min, y_true, color=UQ_ACTUAL_COLOR, linewidth=1.0, alpha=0.9, label='Actual')
    ax.plot(t_min, y_pred, color=UQ_MEAN_COLOR, linewidth=1.1, alpha=0.95, label='Mean')
    miss = ~inside
    if miss.any():
        ax.scatter(t_min[miss], y_true[miss], s=9, color=UQ_MISS_COLOR, zorder=5,
                   label='Outside band')
    ax.text(0.02, 0.97, f'cov={coverage:.2f}\nRMSE={rmse:.3g}',
            transform=ax.transAxes, va='top', ha='left', fontsize=7,
            bbox=dict(boxstyle='round', fc='white', alpha=0.7, lw=0))
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(fontsize=7, loc='upper right', framealpha=0.85)
    return coverage


def plot_prediction_with_uncertainty(
    predictions_dict: Dict[str, np.ndarray],
    dynamic_cols: List[str],
    columns: Optional[List[str]] = None,
    n_sigma: float = 2.0,
    recal_scale: float = 1.0,
    n_samples_show: int = 150,
    prediction_steps: int = 1,
    config: Optional[Any] = None,
    show_persistence: bool = True,
    title: Optional[str] = None,
    figsize_per_plot: Tuple[int, int] = (7, 3),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Plot predictive mean ± band against the target for selected output columns.

    Bands are ``±n_sigma·σ`` (MC dropout, scaled by ``recal_scale``) or the explicit
    ``[pred_lower, pred_upper]`` interval (PI3NN). Windows are chained on a
    physical-time axis (minutes, via ``config.subsample_factor``); points falling
    outside the band are highlighted in red.
    """
    mean, target, lo_arr, hi_arr, _ = _uq_bands(predictions_dict, n_sigma, recal_scale)
    if mean is None or target is None or lo_arr is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Need pred mean/target + uncertainty in predictions_dict',
                ha='center', va='center')
        return fig

    if columns is None:
        columns = dynamic_cols[:4]
    columns = [c for c in columns if c in dynamic_cols]
    if not columns:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No requested columns found in dynamic_cols', ha='center')
        return fig

    is_interval = predictions_dict.get('pred_lower') is not None
    K = mean.shape[1]  # actual horizon (prediction_steps kept for back-compat only)
    N = min(n_samples_show, mean.shape[0])
    dt = _subsample(config)
    t_min = _time_minutes(N, K, dt)
    last = predictions_dict.get('last_dynamic', predictions_dict.get('last_output'))

    n = len(columns)
    ncols = min(2, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        squeeze=False)
    axes = axes.flatten()

    for ax_idx, col in enumerate(columns):
        ax = axes[ax_idx]
        idx = dynamic_cols.index(col)
        y_true = _chain_series(target, idx, N, K)
        y_pred = _chain_series(mean, idx, N, K)
        y_lo = _chain_series(lo_arr, idx, N, K)
        y_hi = _chain_series(hi_arr, idx, N, K)
        persist = None
        if show_persistence and last is not None and last.ndim == 2:
            persist = np.repeat(last[:N, idx], K)
        _draw_band_panel(ax, t_min, y_true, y_pred, y_lo, y_hi, n_sigma,
                         is_interval, persist=persist, show_legend=(ax_idx == 0),
                         title=col)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title or 'Predictions with Uncertainty', fontsize=13)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def coverage_curve(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    levels: Optional[np.ndarray] = None,
    recal_scale: float = 1.0,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Empirical central coverage at a sweep of nominal levels for a Gaussian
    predictive distribution. Returns ``(nominal_levels, empirical_coverage)``.
    """
    from statistics import NormalDist
    if levels is None:
        levels = np.linspace(0.05, 0.99, 20)
    sigma = pred_std.flatten() * recal_scale + eps
    absz = np.abs((target.flatten() - pred_mean.flatten()) / sigma)
    nd = NormalDist()
    emp = np.array([np.mean(absz <= nd.inv_cdf(0.5 + lv / 2.0)) for lv in levels])
    return np.asarray(levels), emp


def _reliability_axes(ax, mean, sigma, target, recal_scale, nominal=0.95):
    """Draw a reliability curve (raw + recalibrated) with shaded ECE gap."""
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Ideal')
    lv, emp = coverage_curve(mean, sigma, target, recal_scale=1.0)
    ax.fill_between(lv, emp, lv, color='tab:red', alpha=0.12)
    ece = float(np.mean(np.abs(emp - lv)))
    ax.plot(lv, emp, 'o-', color='tab:red', markersize=4,
            label=f'Raw (ECE={ece:.3f})')
    if abs(recal_scale - 1.0) > 1e-6:
        lv2, emp2 = coverage_curve(mean, sigma, target, recal_scale=recal_scale)
        ece2 = float(np.mean(np.abs(emp2 - lv2)))
        ax.plot(lv2, emp2, 's-', color='tab:green', markersize=4,
                label=f'Recal ×{recal_scale:.2f} (ECE={ece2:.3f})')
    ax.axvline(nominal, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Nominal coverage'); ax.set_ylabel('Empirical coverage')
    ax.set_title('Reliability')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='upper left'); ax.grid(True, alpha=0.3)


def _sharpness_axes(ax, widths):
    """Histogram of interval widths (sharpness): narrower = sharper."""
    w = widths[np.isfinite(widths)]
    if len(w) == 0:
        ax.set_visible(False); return
    ax.hist(w, bins=40, color='tab:purple', alpha=0.8, edgecolor='black', linewidth=0.3)
    med = float(np.median(w))
    ax.axvline(med, color='red', ls='--', linewidth=1.2, label=f'median={med:.3g}')
    ax.set_xlabel('Interval width (physical units)'); ax.set_ylabel('Count')
    ax.set_title('Sharpness (interval width)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')


def plot_reliability_diagram(
    predictions_dict: Dict[str, np.ndarray],
    recal_scale: float = 1.0,
    nominal: float = 0.95,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Calibration + sharpness: reliability diagram (with shaded ECE gap) next to a
    histogram of interval widths.

    The reliability curve below the diagonal ⇒ over-confident (too narrow); above
    ⇒ under-confident. The recalibrated curve is overlaid when ``recal_scale≠1``.
    Works for Gaussian (``pred_std``) and interval (``pred_lower/upper``) UQ.
    """
    mean, target, lo, hi, sigma = _uq_bands(predictions_dict, n_sigma=2.0,
                                            recal_scale=recal_scale)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if sigma is None or mean is None or target is None:
        axes[0].text(0.5, 0.5, 'Reliability needs mean/target + uncertainty',
                     ha='center', va='center')
        axes[1].set_visible(False)
        return fig

    _reliability_axes(axes[0], mean, sigma, target, recal_scale, nominal)
    widths = (hi - lo).reshape(-1) * recal_scale if lo is not None \
        else (2 * 1.96 * sigma * recal_scale).reshape(-1)
    _sharpness_axes(axes[1], widths)

    plt.suptitle('Calibration & Sharpness', fontsize=13)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _spans_decades(v, n=2):
    """True if positive values of ``v`` span more than ``n`` orders of magnitude."""
    p = v[np.isfinite(v) & (v > 0)]
    return len(p) > 0 and (p.max() / p.min()) > 10 ** n


def plot_uncertainty_vs_error(
    metrics_df: pd.DataFrame,
    color_by: Optional[str] = None,
    density: bool = False,
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Per-output predictive σ vs RMSE. A trustworthy UQ shows a positive trend —
    outputs the model is wrong about should be flagged as uncertain.

    Auto log–log when values span >2 decades (T/V/p/W differ wildly); coloured by
    group; a binned-median trend line and Spearman ρ summarise the relationship.
    Set ``density=True`` for a hexbin instead of a coloured scatter (huge N).
    """
    fig, ax = plt.subplots(figsize=figsize)
    if 'Mean_Sigma' not in metrics_df.columns or 'RMSE' not in metrics_df.columns:
        ax.text(0.5, 0.5, "Need 'Mean_Sigma' and 'RMSE' columns",
                ha='center', va='center')
        return fig

    x = metrics_df['RMSE'].values.astype(float)
    y = metrics_df['Mean_Sigma'].values.astype(float)
    logscale = _spans_decades(x) or _spans_decades(y)

    color_col = color_by if (color_by and color_by in metrics_df.columns) else None
    if color_col is None:
        for c in ('Group', 'Category', 'Output_Type', 'Type'):
            if c in metrics_df.columns:
                color_col = c
                break

    if density or color_col is None:
        m = np.isfinite(x) & np.isfinite(y)
        if logscale:
            m &= (x > 0) & (y > 0)
        hb = ax.hexbin(x[m], y[m], gridsize=40, cmap='viridis', mincnt=1,
                       xscale='log' if logscale else 'linear',
                       yscale='log' if logscale else 'linear')
        fig.colorbar(hb, ax=ax, label='count')
    else:
        palette = {'Group': GROUP_COLORS, 'Category': CATEGORY_COLORS,
                   'Domain': DOMAIN_COLORS}.get(color_col, {})
        for key, sub in metrics_df.groupby(color_col):
            ax.scatter(sub['RMSE'], sub['Mean_Sigma'], s=12, alpha=0.5,
                       label=str(key), color=palette.get(key, None),
                       edgecolors='none')
        ax.legend(fontsize=8, title=color_col, framealpha=0.85, markerscale=1.5)

    # y = x guide + binned-median trend
    pos = np.isfinite(x) & np.isfinite(y) & ((x > 0) & (y > 0) if logscale else True)
    if pos.sum() > 1:
        lim_lo = min(np.min(x[pos]), np.min(y[pos]))
        lim_hi = max(np.max(x[pos]), np.max(y[pos]))
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], 'k--', alpha=0.4, label='σ = RMSE')
        xb = x[pos]; yb = y[pos]
        edges = (np.geomspace(xb.min(), xb.max(), 11) if logscale
                 else np.linspace(xb.min(), xb.max(), 11))
        cx, cy = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = (xb >= a) & (xb < b)
            if sel.sum() >= 3:
                cx.append(np.sqrt(a * b) if logscale else 0.5 * (a + b))
                cy.append(np.median(yb[sel]))
        if cx:
            ax.plot(cx, cy, '-', color='black', linewidth=2, alpha=0.85,
                    label='binned median')

    if logscale:
        ax.set_xscale('log'); ax.set_yscale('log')
    rho = _spearman(x, y)
    ax.set_title(f'Predictive σ vs RMSE  (Spearman ρ={rho:.2f})', fontsize=13)
    ax.set_xlabel('RMSE (physical units)', fontsize=12)
    ax.set_ylabel('Mean predictive σ', fontsize=12)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def _group_order(metrics_df, group_col):
    if group_col == 'Group':
        canonical = ['G_T', 'G_V', 'G_p', 'G_Vs', 'G_ps', 'G_W']
        present = [g for g in canonical if g in metrics_df[group_col].unique()]
        extras = [g for g in metrics_df[group_col].unique() if g not in canonical]
        return present + extras
    return list(metrics_df[group_col].unique())


def plot_uncertainty_by_group(
    metrics_df: pd.DataFrame,
    nominal: float = 0.95,
    figsize: Tuple[int, int] = (16, 5),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Per-group coverage, normalized uncertainty, and sharpness.

    - **Coverage**: PICP raw vs recalibrated with the nominal target line — which
      subsystems are mis-calibrated.
    - **σ / RMSE** (dimensionless, comparable across groups): ≈1 means the
      predictive σ matches the actual error; ≫1 conservative, ≪1 over-confident.
    - **MPIW**: mean interval width per group (physical units, within-group sharpness).
    """
    group_col = None
    for c in ('Group', 'Category', 'Output_Type', 'Type'):
        if c in metrics_df.columns:
            group_col = c
            break

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if group_col is None or 'Mean_Sigma' not in metrics_df.columns:
        axes[0].text(0.5, 0.5, "Need a group column + 'Mean_Sigma'",
                     ha='center', va='center')
        for a in axes[1:]:
            a.set_visible(False)
        return fig

    groups = _group_order(metrics_df, group_col)
    colors = [GROUP_COLORS.get(g, 'steelblue') for g in groups]
    x = np.arange(len(groups))
    has_recal = 'PICP_95_recal' in metrics_df.columns

    # Panel 0: PICP raw (+ recal)
    ax = axes[0]
    if 'PICP_95' in metrics_df.columns:
        picp = [metrics_df[metrics_df[group_col] == g]['PICP_95'].mean() for g in groups]
        w = 0.38 if has_recal else 0.6
        ax.bar(x - (w/2 if has_recal else 0), picp, w, color=colors,
               edgecolor='black', alpha=0.85, label='raw')
        if has_recal:
            picp_r = [metrics_df[metrics_df[group_col] == g]['PICP_95_recal'].mean()
                      for g in groups]
            ax.bar(x + w/2, picp_r, w, color=colors, edgecolor='black',
                   alpha=0.5, hatch='//', label='recal')
        ax.axhline(nominal, color='red', ls='--', alpha=0.7, label=f'nominal {nominal:.0%}')
        ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('PICP (coverage)'); ax.set_title('Coverage by group')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 1: normalized σ / RMSE (dimensionless)
    ax = axes[1]
    ratio = metrics_df['Mean_Sigma'] / (metrics_df['RMSE'] + 1e-9)
    means, errs = [], []
    for g in groups:
        r = ratio[metrics_df[group_col] == g]
        means.append(float(r.mean())); errs.append(float(r.std()))
    ax.bar(x, means, yerr=errs, capsize=3, color=colors, edgecolor='black', alpha=0.85)
    ax.axhline(1.0, color='red', ls='--', alpha=0.6, label='σ = RMSE')
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('σ / RMSE'); ax.set_title('Normalized uncertainty')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: MPIW (physical units)
    ax = axes[2]
    if 'MPIW_95' in metrics_df.columns:
        mpiw = [metrics_df[metrics_df[group_col] == g]['MPIW_95'].mean() for g in groups]
        ax.bar(x, mpiw, color=colors, edgecolor='black', alpha=0.85)
        ax.set_ylabel('MPIW (physical units)')
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
    ax.set_title('Sharpness by group')
    ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Uncertainty by Group', fontsize=13)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Uncertainty Time-Series (presentation-grade band plots)
# ───────────────────────────────────────────────────────────────────────

def _subsample(config):
    if config is None:
        return 30
    return getattr(config, 'subsample_factor', getattr(config, 'SUBSAMPLE_FACTOR', 30))


def plot_timeseries_with_uncertainty(
    predictions_dict: Dict[str, np.ndarray],
    dynamic_cols: List[str],
    output_types: List[str],
    output_patterns: Optional[Dict[str, str]] = None,
    cdu_ids: Optional[List[int]] = None,
    config: Optional[Any] = None,
    n_sigma: float = 2.0,
    recal_scale: float = 1.0,
    zoom: bool = False,
    n_samples_show: int = 200,
    n_cdus: int = 4,
    show_persistence: bool = True,
    figsize_per: Tuple[int, int] = (6, 3),
    save_path: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Prediction-vs-actual time-series **with uncertainty bands**, as a
    ``output_type × CDU`` grid (mirrors :func:`plot_timeseries_chained`).

    Each panel chains the windows on a physical-time axis and shades the ±band;
    points outside the band are flagged. ``zoom=True`` applies the near-constant
    ±3× y-zoom. Select CDUs via ``cdu_ids`` (or ``config.CDU_IDS`` / first ``n_cdus``).
    """
    mean, target, lo_arr, hi_arr, _ = _uq_bands(predictions_dict, n_sigma, recal_scale)
    if mean is None or target is None or lo_arr is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Need pred mean/target + uncertainty', ha='center', va='center')
        return fig

    cdu_ids = _resolve_cdu_ids(cdu_ids, config, n_cdus)
    is_interval = predictions_dict.get('pred_lower') is not None
    K = mean.shape[1]
    N = min(n_samples_show, mean.shape[0])
    t_min = _time_minutes(N, K, _subsample(config))
    last = predictions_dict.get('last_dynamic', predictions_dict.get('last_output'))
    n_types = len(output_types)

    fig, axes = plt.subplots(
        n_types, len(cdu_ids),
        figsize=(figsize_per[0] * len(cdu_ids), figsize_per[1] * n_types),
        squeeze=False)

    for row, otype in enumerate(output_types):
        pattern = output_patterns.get(otype, "") if output_patterns else ""
        for col_idx, cdu_id in enumerate(cdu_ids):
            ax = axes[row, col_idx]
            col_name = pattern.format(cdu_id) if pattern and "{}" in pattern \
                else f"{otype}_CDU{cdu_id}"
            if col_name not in dynamic_cols:
                ax.set_visible(False); continue
            idx = dynamic_cols.index(col_name)
            y_true = _chain_series(target, idx, N, K)
            y_pred = _chain_series(mean, idx, N, K)
            y_lo = _chain_series(lo_arr, idx, N, K)
            y_hi = _chain_series(hi_arr, idx, N, K)
            persist = (np.repeat(last[:N, idx], K)
                       if show_persistence and last is not None and last.ndim == 2 else None)
            _draw_band_panel(ax, t_min, y_true, y_pred, y_lo, y_hi, n_sigma,
                             is_interval, persist=persist,
                             show_legend=(row == 0 and col_idx == 0),
                             title=f'{otype} · CDU {cdu_id}')
            if zoom:
                ymean = y_true.mean()
                yr = max(y_true.max() - y_true.min(), 1e-4) * 3
                ax.set_ylim(ymean - yr, ymean + yr)
            if col_idx == 0:
                ax.set_ylabel(otype.split('_')[0], fontsize=8)
            if row == n_types - 1:
                ax.set_xlabel('Time (min)', fontsize=8)
            ax.tick_params(labelsize=7)

    plt.suptitle(title or f'Predictions with Uncertainty ({N} windows)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_group_uncertainty_timeseries(
    predictions_dict: Dict[str, np.ndarray],
    column_info: Dict[str, Any],
    groups: Optional[List[str]] = None,
    config: Optional[Any] = None,
    n_sigma: float = 2.0,
    recal_scale: float = 1.0,
    n_samples_show: int = 200,
    figsize_per: Tuple[int, int] = (6, 3.2),
    save_path: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Per-group **aggregated** uncertainty time-series: for each decoder-head group
    (``G_T``, ``G_V``, …) the predicted/actual trajectories and band are averaged
    across all of that group's columns (consistent units within a group) — a
    subsystem-level view of model certainty over time.
    """
    mean, target, lo_arr, hi_arr, _ = _uq_bands(predictions_dict, n_sigma, recal_scale)
    if mean is None or lo_arr is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Need pred mean/target + uncertainty', ha='center', va='center')
        return fig

    gcols = _group_columns(column_info)
    dynamic_cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
    col_index = {c: i for i, c in enumerate(dynamic_cols)}
    groups = [g for g in (groups or list(gcols)) if gcols.get(g)]
    if not groups:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No groups resolved from column_info', ha='center')
        return fig

    is_interval = predictions_dict.get('pred_lower') is not None
    K = mean.shape[1]
    N = min(n_samples_show, mean.shape[0])
    t_min = _time_minutes(N, K, _subsample(config))

    ncols = min(3, len(groups))
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for gi, g in enumerate(groups):
        ax = axes[gi]
        idxs = [col_index[c] for c in gcols[g] if c in col_index]
        if not idxs:
            ax.set_visible(False); continue
        y_true = target[:N, :, idxs].mean(axis=2).reshape(-1)
        y_pred = mean[:N, :, idxs].mean(axis=2).reshape(-1)
        y_lo = lo_arr[:N, :, idxs].mean(axis=2).reshape(-1)
        y_hi = hi_arr[:N, :, idxs].mean(axis=2).reshape(-1)
        _draw_band_panel(ax, t_min, y_true, y_pred, y_lo, y_hi, n_sigma,
                         is_interval, show_legend=(gi == 0),
                         title=f'{GROUP_LABELS.get(g, g)}  ({len(idxs)} outputs)')
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(groups), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title or 'Per-Group Aggregated Uncertainty', fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_cdu_uncertainty_timeseries(
    predictions_dict: Dict[str, np.ndarray],
    column_info: Dict[str, Any],
    cdu_ids: Optional[List[int]] = None,
    config: Optional[Any] = None,
    n_sigma: float = 2.0,
    recal_scale: float = 1.0,
    n_samples_show: int = 200,
    n_cdus: int = 4,
    figsize_per: Tuple[int, int] = (6, 3.2),
    save_path: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Per-CDU **aggregated** uncertainty time-series for selectable CDUs.

    Each CDU's outputs are z-scored by their own target mean/std (so mixed units —
    °C/GPM/psig/kW — combine) and averaged into a single normalized "certainty"
    trace with band. One panel per CDU.
    """
    mean, target, lo_arr, hi_arr, _ = _uq_bands(predictions_dict, n_sigma, recal_scale)
    if mean is None or lo_arr is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Need pred mean/target + uncertainty', ha='center', va='center')
        return fig

    dynamic_cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
    col_to_cdu = column_info.get('col_to_cdu', {})
    cdu_ids = _resolve_cdu_ids(cdu_ids, config, n_cdus)
    is_interval = predictions_dict.get('pred_lower') is not None
    K = mean.shape[1]
    N = min(n_samples_show, mean.shape[0])
    t_min = _time_minutes(N, K, _subsample(config))

    ncols = min(3, len(cdu_ids))
    nrows = (len(cdu_ids) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ci, cdu in enumerate(cdu_ids):
        ax = axes[ci]
        idxs = [i for i, c in enumerate(dynamic_cols) if col_to_cdu.get(c) == cdu]
        if not idxs:
            ax.set_visible(False); continue
        # z-score each output by its own target stats, then average across outputs
        tn, pn, ln, hn = [], [], [], []
        for idx in idxs:
            tcol = target[:N, :, idx]
            mu, sd = float(tcol.mean()), float(tcol.std()) + 1e-9
            tn.append((tcol - mu) / sd)
            pn.append((mean[:N, :, idx] - mu) / sd)
            ln.append((lo_arr[:N, :, idx] - mu) / sd)
            hn.append((hi_arr[:N, :, idx] - mu) / sd)
        y_true = np.mean(tn, axis=0).reshape(-1)
        y_pred = np.mean(pn, axis=0).reshape(-1)
        y_lo = np.mean(ln, axis=0).reshape(-1)
        y_hi = np.mean(hn, axis=0).reshape(-1)
        _draw_band_panel(ax, t_min, y_true, y_pred, y_lo, y_hi, n_sigma,
                         is_interval, show_legend=(ci == 0),
                         title=f'CDU {cdu}  ({len(idxs)} outputs, normalized)')
        ax.set_ylabel('normalized (z)', fontsize=8)
        ax.set_xlabel('Time (min)', fontsize=8)
        ax.tick_params(labelsize=7)

    for j in range(len(cdu_ids), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(title or 'Per-CDU Aggregated Uncertainty (normalized)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# Standard 11-output-per-CDU layout (mirrors
# fmu2ml.visualization.plotters.output_plots.StandardInputOutputVisualizer.plot_outputs):
# column 0 = temperatures, column 1 = pressures, column 2 = flows + pump power.
STANDARD_IO_LAYOUT = [
    (0, 0, 'T_prim_s_C', 'Primary Supply Temp [°C]'),
    (1, 0, 'T_prim_r_C', 'Primary Return Temp [°C]'),
    (2, 0, 'T_sec_s_C', 'Secondary Supply Temp [°C]'),
    (3, 0, 'T_sec_r_C', 'Secondary Return Temp [°C]'),
    (0, 1, 'p_prim_s_psig', 'Primary Supply Pressure [psig]'),
    (1, 1, 'p_prim_r_psig', 'Primary Return Pressure [psig]'),
    (2, 1, 'p_sec_s_psig', 'Secondary Supply Pressure [psig]'),
    (3, 1, 'p_sec_r_psig', 'Secondary Return Pressure [psig]'),
    (0, 2, 'V_flow_prim_GPM', 'Primary Flow Rate [GPM]'),
    (1, 2, 'V_flow_sec_GPM', 'Secondary Flow Rate [GPM]'),
    (2, 2, 'W_flow_CDUP_kW', 'CDUP Power [kW]'),
]

# Unit suffixes appended to FMU column names; stripped when matching a layout
# entry against ``column_info['col_to_type']``. The ``col_to_type`` map stores the
# bare output-pattern *keys* (e.g. ``T_prim_s``, ``p_prim_s``, ``V_flow_prim``,
# ``W_flow``) while ``STANDARD_IO_LAYOUT`` carries the unit-qualified variable
# names (``T_prim_s_C``, ``p_prim_s_psig`` …). Normalising both ends lets the
# lookup succeed regardless of which naming convention a phase's column_info uses.
_IO_UNIT_SUFFIXES = ('_C', '_psig', '_GPM', '_kW')

# Normalized-key → axis label, derived from STANDARD_IO_LAYOUT. Lets callers pass
# a bare/qualified output name to plot_cdu_output_uncertainty and still get a
# descriptive, unit-annotated y-axis label.
_IO_LABELS = {}


def _norm_io_key(name: Optional[str]) -> Optional[str]:
    """Strip a trailing unit suffix so layout vars and ``col_to_type`` values match.

    ``W_flow_CDUP_kW`` → ``W_flow_CDUP`` and ``W_flow`` stays ``W_flow``; the power
    head only ever stores one of the two, so we also collapse the ``_CDUP``
    qualifier to make ``W_flow_CDUP`` and ``W_flow`` equivalent.
    """
    if name is None:
        return None
    s = str(name)
    for suf in _IO_UNIT_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    if s.endswith('_CDUP'):
        s = s[: -len('_CDUP')]
    return s


_IO_LABELS.update({_norm_io_key(v): lbl for _, _, v, lbl in STANDARD_IO_LAYOUT})


def plot_cdu_io_uncertainty(
    predictions_dict: Dict[str, np.ndarray],
    column_info: Dict[str, Any],
    cdu_ids: Optional[List[int]] = None,
    config: Optional[Any] = None,
    n_sigma: float = 2.0,
    recal_scale: float = 1.0,
    n_samples_show: int = 200,
    n_cdus: int = 3,
    layout: Optional[List[Tuple[int, int, str, str]]] = None,
    show_persistence: bool = True,
    figsize: Tuple[int, int] = (18, 14),
    save_dir: Optional[PathLike] = None,
) -> Dict[int, plt.Figure]:
    """
    One **StandardInputOutput-style figure per selected CDU**: the 11 output
    variables as separate panels (temperatures | pressures | flows+power), each
    showing actual + predicted mean + uncertainty band over physical time.

    Mirrors ``StandardInputOutputVisualizer.plot_outputs`` but for surrogate
    predictions with UQ. Columns are resolved via ``column_info['col_to_cdu']`` /
    ``['col_to_type']`` so it works regardless of the system's column naming.

    Returns ``{cdu_id: Figure}``; if ``save_dir`` is given, writes
    ``cdu_<id>_io_uncertainty.png`` for each.
    """
    mean, target, lo_arr, hi_arr, _ = _uq_bands(predictions_dict, n_sigma, recal_scale)
    if mean is None or target is None or lo_arr is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Need pred mean/target + uncertainty', ha='center', va='center')
        return {-1: fig}

    dynamic_cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
    col_to_cdu = column_info.get('col_to_cdu', {})
    col_to_type = column_info.get('col_to_type', {})
    # (cdu, normalized output_type) -> column index. Keys are normalised so a
    # layout entry like 'T_prim_s_C' matches a col_to_type value of 'T_prim_s'.
    idx_of = {(col_to_cdu.get(c), _norm_io_key(col_to_type.get(c))): i
              for i, c in enumerate(dynamic_cols)}

    cdu_ids = _resolve_cdu_ids(cdu_ids, config, n_cdus)
    is_interval = predictions_dict.get('pred_lower') is not None
    K = mean.shape[1]
    N = min(n_samples_show, mean.shape[0])
    t_min = _time_minutes(N, K, _subsample(config))
    last = predictions_dict.get('last_dynamic', predictions_dict.get('last_output'))
    layout = layout or STANDARD_IO_LAYOUT
    bottom = {(3, 0), (3, 1), (2, 2)}  # panels that get an x-axis label

    figures: Dict[int, plt.Figure] = {}
    for cdu in cdu_ids:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 3, hspace=0.38, wspace=0.26)
        first = True
        for row, col, var, ylabel in layout:
            ax = fig.add_subplot(gs[row, col])
            idx = idx_of.get((cdu, _norm_io_key(var)))
            if idx is None:
                ax.text(0.5, 0.5, f'{var}\n(not available)', ha='center', va='center',
                        fontsize=8, color='gray')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            persist = (np.repeat(last[:N, idx], K)
                       if show_persistence and last is not None and last.ndim == 2 else None)
            _draw_band_panel(ax, t_min,
                             _chain_series(target, idx, N, K),
                             _chain_series(mean, idx, N, K),
                             _chain_series(lo_arr, idx, N, K),
                             _chain_series(hi_arr, idx, N, K),
                             n_sigma, is_interval, persist=persist,
                             show_legend=first, title=ylabel.split('[')[0].strip())
            first = False
            ax.set_ylabel(ylabel, fontsize=9)
            if (row, col) in bottom:
                ax.set_xlabel('Time (min)', fontsize=9)
            ax.tick_params(labelsize=7)

        band = 'PI' if is_interval else f'±{n_sigma:g}σ'
        fig.suptitle(f'CDU {cdu} — Predicted Outputs with Uncertainty ({band})',
                     fontsize=15, y=0.995)
        if save_dir is not None:
            save_figure(fig, Path(save_dir) / f'cdu_{cdu}_io_uncertainty.png')
        figures[cdu] = fig

    return figures


def plot_cdu_output_uncertainty(
    predictions_dict: Dict[str, np.ndarray],
    column_info: Dict[str, Any],
    cdu_ids: Optional[List[int]] = None,
    outputs: Optional[List[str]] = None,
    config: Optional[Any] = None,
    n_sigma: float = 2.0,
    recal_scale: float = 1.0,
    n_samples_show: int = 200,
    n_cdus: int = 3,
    show_persistence: bool = True,
    figsize: Tuple[int, int] = (9, 4),
    save_dir: Optional[PathLike] = None,
) -> Dict[Tuple[int, str], plt.Figure]:
    """
    One **standalone figure per (CDU, output)** — the fine-grained companion to
    :func:`plot_cdu_io_uncertainty`, which packs all 11 outputs into a single
    grid. Here every selected CDU × output gets its own full-size plot showing
    actual + predicted mean + uncertainty band (+ optional persistence baseline)
    over physical time, with per-panel coverage and RMSE annotations.

    Parameters
    ----------
    cdu_ids
        CDUs to plot (resolved via ``config`` / default when ``None``).
    outputs
        Output variable names to plot, e.g. ``['T_prim_s_C', 'W_flow_CDUP_kW']``.
        Matching is suffix-insensitive, so both ``W_flow`` and ``W_flow_CDUP_kW``
        resolve to the same column. Defaults to all 11 standard outputs.
    save_dir
        When given, writes ``cdu_<id>_<output>_uncertainty.png`` per figure.

    Returns
    -------
    Dict[(cdu_id, output_name), Figure]
    """
    mean, target, lo_arr, hi_arr, _ = _uq_bands(predictions_dict, n_sigma, recal_scale)
    if mean is None or target is None or lo_arr is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Need pred mean/target + uncertainty', ha='center', va='center')
        return {(-1, ''): fig}

    dynamic_cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
    col_to_cdu = column_info.get('col_to_cdu', {})
    col_to_type = column_info.get('col_to_type', {})
    idx_of = {(col_to_cdu.get(c), _norm_io_key(col_to_type.get(c))): i
              for i, c in enumerate(dynamic_cols)}

    # (var, ylabel) pairs; default to the 11-output standard layout order.
    var_labels = ([(v, lbl) for _, _, v, lbl in STANDARD_IO_LAYOUT]
                  if outputs is None else
                  [(v, _IO_LABELS.get(_norm_io_key(v), v)) for v in outputs])

    cdu_ids = _resolve_cdu_ids(cdu_ids, config, n_cdus)
    is_interval = predictions_dict.get('pred_lower') is not None
    K = mean.shape[1]
    N = min(n_samples_show, mean.shape[0])
    t_min = _time_minutes(N, K, _subsample(config))
    last = predictions_dict.get('last_dynamic', predictions_dict.get('last_output'))
    band = 'PI' if is_interval else f'±{n_sigma:g}σ'

    figures: Dict[Tuple[int, str], plt.Figure] = {}
    for cdu in cdu_ids:
        for var, ylabel in var_labels:
            idx = idx_of.get((cdu, _norm_io_key(var)))
            if idx is None:
                continue  # output not present for this CDU; skip silently
            fig, ax = plt.subplots(figsize=figsize)
            persist = (np.repeat(last[:N, idx], K)
                       if show_persistence and last is not None and last.ndim == 2 else None)
            _draw_band_panel(ax, t_min,
                             _chain_series(target, idx, N, K),
                             _chain_series(mean, idx, N, K),
                             _chain_series(lo_arr, idx, N, K),
                             _chain_series(hi_arr, idx, N, K),
                             n_sigma, is_interval, persist=persist,
                             show_legend=True, title=None)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xlabel('Time (min)', fontsize=10)
            ax.set_title(f'CDU {cdu} — {ylabel.split("[")[0].strip()} ({band})',
                         fontsize=12)
            fig.tight_layout()
            if save_dir is not None:
                save_figure(fig, Path(save_dir) / f'cdu_{cdu}_{var}_uncertainty.png')
            figures[(cdu, var)] = fig

    return figures


# ───────────────────────────────────────────────────────────────────────
# UQ diagnostics, dashboard, and cross-phase comparison
# ───────────────────────────────────────────────────────────────────────

def plot_worst_calibrated(
    metrics_df: pd.DataFrame,
    n: int = 15,
    nominal: float = 0.95,
    by: str = 'calib',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Horizontal bar of the worst-calibrated outputs: largest ``|PICP − nominal|``
    (``by='calib'``) or largest predictive σ (``by='sigma'``). Names the outputs so
    a long tail (e.g. specific secondary-loop CDUs) is actionable.
    """
    fig, ax = plt.subplots(figsize=figsize)
    name_col = 'Output' if 'Output' in metrics_df.columns else None
    if by == 'sigma' and 'Mean_Sigma' in metrics_df.columns:
        score = metrics_df['Mean_Sigma']; xlabel = 'Mean predictive σ'
    elif 'PICP_95' in metrics_df.columns:
        score = (metrics_df['PICP_95'] - nominal).abs(); xlabel = f'|PICP − {nominal:.0%}|'
    else:
        ax.text(0.5, 0.5, "Need 'PICP_95' or 'Mean_Sigma'", ha='center', va='center')
        return fig

    df = metrics_df.assign(_score=score).nlargest(n, '_score').iloc[::-1]
    labels = df[name_col].astype(str).values if name_col else df.index.astype(str).values
    group_col = next((c for c in ('Group', 'Category', 'Output_Type', 'Type')
                      if c in df.columns), None)
    colors = ([GROUP_COLORS.get(g, 'steelblue') for g in df[group_col]]
              if group_col else 'indianred')
    y = np.arange(len(df))
    ax.barh(y, df['_score'].values, color=colors, edgecolor='black', alpha=0.85)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(f'Worst-calibrated outputs (top {len(df)})', fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_uncertainty_dashboard(
    results: Any,
    column_info: Optional[Dict[str, Any]] = None,
    config: Optional[Any] = None,
    columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (20, 11),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Single combined UQ dashboard for one ``UQResults`` (MC dropout or PI3NN).

    2×3 grid: reliability+ECE, sharpness histogram, σ-vs-error, coverage-by-group,
    a representative prediction band, and the worst-calibrated bar. One figure to
    scan/save per phase/method.
    """
    metrics_df = results.metrics_df
    pred = results.predictions_dict
    recal = getattr(results, 'recal_scale', 1.0)
    nominal = getattr(results, 'nominal_coverage', 0.95)
    method = getattr(results, 'method', 'uq')
    phase = getattr(results, 'phase', '?')

    mean, target, lo, hi, sigma = _uq_bands(pred, n_sigma=2.0, recal_scale=recal)
    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # (0,0) reliability  (0,1) sharpness
    if sigma is not None:
        _reliability_axes(axes[0, 0], mean, sigma, target, recal, nominal)
        widths = ((hi - lo).reshape(-1) * recal if lo is not None
                  else (2 * 1.96 * sigma * recal).reshape(-1))
        _sharpness_axes(axes[0, 1], widths)
    else:
        axes[0, 0].set_visible(False); axes[0, 1].set_visible(False)

    # (0,2) sigma vs error (colored by group, log if wide)
    ax = axes[0, 2]
    if {'Mean_Sigma', 'RMSE'}.issubset(metrics_df.columns):
        gcol = next((c for c in ('Group', 'Category', 'Output_Type', 'Type')
                     if c in metrics_df.columns), None)
        logsc = _spans_decades(metrics_df['RMSE'].values) or \
            _spans_decades(metrics_df['Mean_Sigma'].values)
        if gcol:
            pal = {'Group': GROUP_COLORS, 'Category': CATEGORY_COLORS}.get(gcol, {})
            for key, sub in metrics_df.groupby(gcol):
                ax.scatter(sub['RMSE'], sub['Mean_Sigma'], s=10, alpha=0.5,
                           color=pal.get(key, None), edgecolors='none', label=str(key))
            ax.legend(fontsize=7, title=gcol)
        else:
            ax.scatter(metrics_df['RMSE'], metrics_df['Mean_Sigma'], s=10, alpha=0.5)
        if logsc:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('RMSE'); ax.set_ylabel('Mean σ')
        ax.set_title(f'σ vs RMSE (ρ={_spearman(metrics_df["RMSE"], metrics_df["Mean_Sigma"]):.2f})')
        ax.grid(True, alpha=0.3, which='both')
    else:
        ax.set_visible(False)

    # (1,0) coverage by group
    ax = axes[1, 0]
    gcol = next((c for c in ('Group', 'Category', 'Output_Type', 'Type')
                 if c in metrics_df.columns), None)
    if gcol and 'PICP_95' in metrics_df.columns:
        groups = _group_order(metrics_df, gcol)
        picp = [metrics_df[metrics_df[gcol] == g]['PICP_95'].mean() for g in groups]
        ax.bar(np.arange(len(groups)), picp,
               color=[GROUP_COLORS.get(g, 'steelblue') for g in groups],
               edgecolor='black', alpha=0.85)
        ax.axhline(nominal, color='red', ls='--', alpha=0.7)
        ax.set_xticks(np.arange(len(groups)))
        ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=8)
        ax.set_ylim(0, 1.05); ax.set_ylabel('PICP'); ax.set_title('Coverage by group')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.set_visible(False)

    # (1,1) representative prediction band
    ax = axes[1, 1]
    if column_info is not None and lo is not None:
        dyn = column_info.get('dynamic_cols', column_info.get('output_cols', []))
        col = next((c for c in (columns or dyn[:1]) if c in dyn), None)
        if col is not None:
            idx = dyn.index(col)
            K = mean.shape[1]; N = min(150, mean.shape[0])
            t_min = _time_minutes(N, K, _subsample(config))
            _draw_band_panel(ax, t_min, _chain_series(target, idx, N, K),
                             _chain_series(mean, idx, N, K),
                             _chain_series(lo, idx, N, K),
                             _chain_series(hi, idx, N, K), 2.0,
                             pred.get('pred_lower') is not None,
                             show_legend=True, title=col)
            ax.set_xlabel('Time (min)', fontsize=8)
        else:
            ax.set_visible(False)
    else:
        ax.set_visible(False)

    # (1,2) worst-calibrated
    ax = axes[1, 2]
    if 'PICP_95' in metrics_df.columns:
        df = metrics_df.assign(_s=(metrics_df['PICP_95'] - nominal).abs()
                               ).nlargest(12, '_s').iloc[::-1]
        labels = (df['Output'].astype(str).values if 'Output' in df.columns
                  else df.index.astype(str).values)
        ax.barh(np.arange(len(df)), df['_s'].values, color='indianred',
                edgecolor='black', alpha=0.85)
        ax.set_yticks(np.arange(len(df))); ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel(f'|PICP − {nominal:.0%}|'); ax.set_title('Worst-calibrated')
        ax.grid(True, alpha=0.3, axis='x')
    else:
        ax.set_visible(False)

    plt.suptitle(f'UQ Dashboard — Phase {phase} · {method} · target {nominal:.0%}',
                 fontsize=15, y=1.0)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_uq_comparison(
    comparison_df: pd.DataFrame,
    nominal: float = 0.95,
    figsize: Tuple[int, int] = (16, 11),
    save_path: Optional[PathLike] = None,
) -> plt.Figure:
    """
    Cross-phase × method UQ comparison from the notebook's ``comparison`` table.

    Panels: (a) R² grouped MC vs PI3NN per phase, (b) calibration error
    |PICP−nominal|, (c) MPIW (sharpness), (d) coverage-vs-sharpness tradeoff
    scatter (ideal = high coverage at low width, top-left).
    """
    df = comparison_df.copy()
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    if df.empty:
        axes[0, 0].text(0.5, 0.5, 'No comparison rows', ha='center')
        return fig

    methods = list(dict.fromkeys(df['Method']))
    method_color = {'mc_dropout': 'steelblue', 'pi3nn': 'darkorange'}
    method_marker = {'mc_dropout': 'o', 'pi3nn': 's'}
    rows = list(dict.fromkeys(df['Model']))  # one row per model (incl. domains)
    x = np.arange(len(rows))
    w = 0.8 / max(len(methods), 1)

    def _vals(metric, method):
        sub = df[df['Method'] == method].set_index('Model')
        return [sub.loc[r, metric] if r in sub.index else np.nan for r in rows]

    # (a) R²
    ax = axes[0, 0]
    for mi, m in enumerate(methods):
        ax.bar(x + (mi - (len(methods) - 1) / 2) * w, _vals('Mean_R2', m), w,
               label=m.replace('_', ' '), color=method_color.get(m), edgecolor='black', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(rows, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('Mean R²'); ax.set_title('Accuracy'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (b) calibration error
    ax = axes[0, 1]
    picp_col = 'PICP_recal' if 'PICP_recal' in df.columns else 'PICP_raw'
    for mi, m in enumerate(methods):
        ce = [abs(v - nominal) if np.isfinite(v) else np.nan for v in _vals(picp_col, m)]
        ax.bar(x + (mi - (len(methods) - 1) / 2) * w, ce, w,
               label=m.replace('_', ' '), color=method_color.get(m), edgecolor='black', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(rows, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel(f'|{picp_col} − {nominal:.0%}|'); ax.set_title('Calibration error')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # (c) MPIW
    ax = axes[1, 0]
    for mi, m in enumerate(methods):
        ax.bar(x + (mi - (len(methods) - 1) / 2) * w, _vals('MPIW', m), w,
               label=m.replace('_', ' '), color=method_color.get(m), edgecolor='black', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(rows, rotation=60, ha='right', fontsize=7)
    ax.set_ylabel('MPIW'); ax.set_title('Sharpness (interval width)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')

    # (d) coverage vs sharpness tradeoff
    ax = axes[1, 1]
    for _, row in df.iterrows():
        m = row['Method']
        ax.scatter(row.get('MPIW', np.nan), row.get(picp_col, np.nan),
                   marker=method_marker.get(m, 'o'), s=70,
                   color=method_color.get(m), edgecolor='black', alpha=0.85)
        ax.annotate(str(row['Phase']),
                    (row.get('MPIW', np.nan), row.get(picp_col, np.nan)),
                    fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.axhline(nominal, color='red', ls='--', alpha=0.6, label=f'nominal {nominal:.0%}')
    ax.set_xlabel('MPIW (sharpness →)'); ax.set_ylabel('PICP (coverage ↑)')
    ax.set_title('Coverage vs sharpness  (ideal: top-left)')
    handles = [Line2D([], [], marker=method_marker.get(m, 'o'), ls='',
                      color=method_color.get(m), label=m.replace('_', ' '))
               for m in methods]
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Uncertainty Quantification — cross-phase comparison', fontsize=15)
    plt.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


# ───────────────────────────────────────────────────────────────────────
# Main Plotter Class
# ───────────────────────────────────────────────────────────────────────

class SurrogatePlotter:
    """
    Unified plotter for surrogate model visualizations.

    Example:
        plotter = SurrogatePlotter(output_dir='./plots')
        plotter.plot_training(history, phase="5")
        plotter.plot_evaluation_suite(metrics_df, predictions, targets, phase="5")
        plotter.plot_speedups(speedup_results)
    """

    def __init__(
        self,
        output_dir: Optional[PathLike] = None,
        style: str = 'default',
        dpi: int = 150,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.dpi = dpi
        set_style(style)
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_save_path(self, name: str) -> Optional[Path]:
        if self.output_dir:
            return self.output_dir / f"{name}.png"
        return None

    # ───────────────────────────────── Training plots ──────────────────────
    def plot_training(
        self,
        history: Dict[str, List[float]],
        phase: str = "1",
        title: Optional[str] = None,
    ) -> plt.Figure:
        return plot_training_curves(
            history, phase, title,
            save_path=self._get_save_path('training_curves'),
        )

    def plot_lr(self, history: Dict[str, List[float]]) -> plt.Figure:
        return plot_learning_rate(
            history, save_path=self._get_save_path('learning_rate'),
        )

    def plot_head_losses(
        self,
        history: Dict[str, List[float]],
        heads: Optional[List[str]] = None,
    ) -> plt.Figure:
        return plot_loss_by_head(
            history, heads, save_path=self._get_save_path('head_losses'),
        )

    # ─────────────────────────────── Evaluation plots ──────────────────────
    def plot_r2_dist(
        self,
        metrics_df: pd.DataFrame,
        group_by: str = 'Category',
    ) -> plt.Figure:
        return plot_r2_distribution(
            metrics_df, group_by,
            save_path=self._get_save_path('r2_distribution'),
        )

    def plot_r2_types(
        self,
        metrics_df: pd.DataFrame,
        color_by: Optional[str] = None,
    ) -> plt.Figure:
        return plot_r2_by_type(
            metrics_df, color_by=color_by,
            save_path=self._get_save_path('r2_by_type'),
        )

    def plot_r2_groups(self, metrics_df: pd.DataFrame) -> plt.Figure:
        return plot_r2_by_group(
            metrics_df, save_path=self._get_save_path('r2_by_group'),
        )

    def plot_variance(self, metrics_df: pd.DataFrame) -> plt.Figure:
        return plot_variance_ratio(
            metrics_df, save_path=self._get_save_path('variance_ratio'),
        )

    def plot_heatmap(self, metrics_df: pd.DataFrame) -> plt.Figure:
        return plot_r2_heatmap(
            metrics_df, save_path=self._get_save_path('r2_heatmap'),
        )

    # ───────────────────────────────── Prediction plots ────────────────────
    def plot_preds(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        n_samples: int = 5,
    ) -> plt.Figure:
        return plot_predictions(
            predictions, targets, n_samples,
            save_path=self._get_save_path('predictions'),
        )

    def plot_errors(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> plt.Figure:
        return plot_error_distribution(
            predictions, targets,
            save_path=self._get_save_path('error_distribution'),
        )

    def plot_parity(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> plt.Figure:
        return plot_prediction_vs_target(
            predictions, targets,
            save_path=self._get_save_path('parity_plots'),
        )

    # ─────────────────────────────── Benchmark plots ───────────────────────
    def plot_speedups(
        self,
        speedup_results: Dict[str, Dict[int, Dict[str, float]]],
    ) -> plt.Figure:
        return plot_speedup(
            speedup_results, save_path=self._get_save_path('speedup'),
        )

    def plot_latency(
        self,
        timing_results: Dict[str, Dict[int, Dict[str, float]]],
        batch_sizes: List[int],
    ) -> plt.Figure:
        return plot_latency_heatmap(
            timing_results, batch_sizes,
            save_path=self._get_save_path('latency_heatmap'),
        )

    def plot_speedup_bars(
        self,
        speedup_results: Dict[str, Dict[int, Dict[str, float]]],
    ) -> plt.Figure:
        return plot_speedup_summary(
            speedup_results,
            save_path=self._get_save_path('speedup_summary'),
        )

    # ────────────────────────────────── Physics plots ──────────────────────
    def plot_physics(
        self,
        history: Dict[str, Any],
        phase_epochs: Tuple[int, ...] = (50, 50, 50),
    ) -> plt.Figure:
        return plot_physics_evolution(
            history, phase_epochs,
            save_path=self._get_save_path('physics_evolution'),
        )

    # ────────────────────────────────── Comparison plots ───────────────────
    def plot_comparison(
        self,
        phase_metrics: Union[Dict[str, Dict[str, float]], List[pd.DataFrame]],
        metric: Union[str, List[str]] = 'R2',
        labels: Optional[List[str]] = None,
    ) -> plt.Figure:
        """
        Phase comparison. Accepts either a metrics dict or a list of metrics_dfs
        (paired with labels). The dual signature is preserved for backward
        compatibility — argument types are auto-detected.
        """
        return plot_phase_comparison(
            phase_metrics, metric=metric, labels=labels,
            save_path=self._get_save_path('phase_comparison'),
        )

    # ────────────────────────── Uncertainty plots ──────────────────────────
    def plot_uncertainty_dashboard(
        self,
        results: Any,
        column_info: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
        columns: Optional[List[str]] = None,
    ) -> plt.Figure:
        """One combined UQ dashboard figure for a ``UQResults`` (saved as ``uq_dashboard``)."""
        return plot_uncertainty_dashboard(
            results, column_info=column_info, config=config, columns=columns,
            save_path=self._get_save_path('uq_dashboard'),
        )

    def plot_cdu_io(
        self,
        results: Any,
        column_info: Dict[str, Any],
        cdu_ids: Optional[List[int]] = None,
        config: Optional[Any] = None,
        n_cdus: int = 3,
    ) -> Dict[int, plt.Figure]:
        """Per-CDU StandardInputOutput-style figures (11 outputs + uncertainty).

        Saved as ``cdu_<id>_io_uncertainty.png`` under ``output_dir`` when set.
        """
        save_dir = self.output_dir / 'cdu_io' if self.output_dir else None
        return plot_cdu_io_uncertainty(
            results.predictions_dict, column_info, cdu_ids=cdu_ids, config=config,
            n_cdus=n_cdus, recal_scale=getattr(results, 'recal_scale', 1.0),
            save_dir=save_dir,
        )

    def plot_cdu_outputs(
        self,
        results: Any,
        column_info: Dict[str, Any],
        cdu_ids: Optional[List[int]] = None,
        outputs: Optional[List[str]] = None,
        config: Optional[Any] = None,
        n_cdus: int = 3,
    ) -> Dict[Tuple[int, str], plt.Figure]:
        """Per-CDU, per-output **standalone** uncertainty figures.

        One full-size plot for each selected CDU × output (all 11 by default),
        saved as ``cdu_<id>_<output>_uncertainty.png`` under ``output_dir``.
        Use this when the packed 11-panel grid from :meth:`plot_cdu_io` is too
        small to read individual outputs.
        """
        save_dir = self.output_dir / 'cdu_outputs' if self.output_dir else None
        return plot_cdu_output_uncertainty(
            results.predictions_dict, column_info, cdu_ids=cdu_ids, outputs=outputs,
            config=config, n_cdus=n_cdus,
            recal_scale=getattr(results, 'recal_scale', 1.0), save_dir=save_dir,
        )

    def plot_uncertainty_suite(
        self,
        results: Any,
        column_info: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        cdu_ids: Optional[List[int]] = None,
        output_types: Optional[List[str]] = None,
        n_sigma: float = 2.0,
        dashboard: bool = True,
    ) -> Dict[str, plt.Figure]:
        """
        Full UQ visual suite for a ``UQResults`` (MC dropout or PI3NN).

        Produces (saving each if ``output_dir`` is set):
        - ``dashboard`` — combined per-phase dashboard (set ``dashboard=False`` to skip);
        - ``reliability`` (calibration + sharpness), ``uncertainty_vs_error``,
          ``uncertainty_by_group``;
        - when ``column_info`` is given: a representative ``predictions_uncertainty``
          band, a ``group_timeseries`` (per-group aggregated), and a ``cdu_timeseries``
          (per selected CDU). Pass ``config`` for a physical-time x-axis.
        """
        figures: Dict[str, plt.Figure] = {}
        metrics_df = results.metrics_df
        pred = results.predictions_dict
        recal_scale = getattr(results, 'recal_scale', 1.0)
        nominal = getattr(results, 'nominal_coverage', 0.95)

        if dashboard:
            figures['dashboard'] = self.plot_uncertainty_dashboard(
                results, column_info=column_info, config=config, columns=columns)

        figures['reliability'] = plot_reliability_diagram(
            pred, recal_scale=recal_scale, nominal=nominal,
            save_path=self._get_save_path('uq_reliability'))
        figures['uncertainty_vs_error'] = plot_uncertainty_vs_error(
            metrics_df, save_path=self._get_save_path('uq_sigma_vs_error'))
        figures['uncertainty_by_group'] = plot_uncertainty_by_group(
            metrics_df, nominal=nominal,
            save_path=self._get_save_path('uq_by_group'))

        if column_info is not None:
            dynamic_cols = column_info.get('dynamic_cols', column_info.get('output_cols', []))
            steps = pred.get('pred_mean')
            steps = steps.shape[1] if steps is not None else 1
            figures['predictions_uncertainty'] = plot_prediction_with_uncertainty(
                pred, dynamic_cols, columns=columns, n_sigma=n_sigma,
                recal_scale=recal_scale, prediction_steps=steps, config=config,
                save_path=self._get_save_path('uq_predictions'))
            figures['group_timeseries'] = plot_group_uncertainty_timeseries(
                pred, column_info, config=config, recal_scale=recal_scale,
                save_path=self._get_save_path('uq_group_timeseries'))
            if cdu_ids is not None or config is not None:
                figures['cdu_timeseries'] = plot_cdu_uncertainty_timeseries(
                    pred, column_info, cdu_ids=cdu_ids, config=config,
                    recal_scale=recal_scale,
                    save_path=self._get_save_path('uq_cdu_timeseries'))
        return figures

    # ───────────────────────── Extended evaluation suite ───────────────────
    def plot_evaluation_suite_extended(
        self,
        metrics_df: pd.DataFrame,
        predictions_dict: Optional[Dict[str, np.ndarray]] = None,
        column_info: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate the extended visualization suite: distributions, bars,
        persistence comparison, correlation, optional loop dichotomy/heatmap,
        and time-series + delta-quality plots when predictions and column_info
        are supplied.
        """
        figures: Dict[str, plt.Figure] = {}

        # Distributions and bars
        figures['r2_distribution'] = self.plot_r2_dist(metrics_df)
        figures['r2_by_type'] = self.plot_r2_types(metrics_df)
        if 'Group' in metrics_df.columns:
            figures['r2_by_group'] = self.plot_r2_groups(metrics_df)
        figures['variance_ratio'] = self.plot_variance(metrics_df)

        figures['rmse_by_type'] = plot_rmse_by_type(
            metrics_df, save_path=self._get_save_path('rmse_by_type'),
        )
        figures['model_vs_persistence'] = plot_model_vs_persistence(
            metrics_df, save_path=self._get_save_path('model_vs_persistence'),
        )
        figures['correlation'] = plot_correlation_by_type(
            metrics_df, save_path=self._get_save_path('correlation'),
        )

        if 'Loop' in metrics_df.columns:
            figures['loop_dichotomy'] = plot_loop_dichotomy(
                metrics_df, save_path=self._get_save_path('loop_dichotomy'),
            )

        if 'CDU' in metrics_df.columns:
            figures['r2_heatmap'] = self.plot_heatmap(metrics_df)

        if predictions_dict and column_info and config is not None:
            dynamic_cols = column_info.get(
                'dynamic_cols', column_info.get('output_cols', [])
            )

            # Determine output types for time-series
            if hasattr(config, 'ALL_DYNAMIC_OUTPUTS'):
                all_dynamic = list(config.ALL_DYNAMIC_OUTPUTS)
            elif hasattr(config, 'OUTPUT_NAMES'):
                all_dynamic = list(config.OUTPUT_NAMES)
            elif 'Type' in metrics_df.columns:
                all_dynamic = list(metrics_df['Type'].unique())
            else:
                all_dynamic = []

            output_patterns = (config.OUTPUT_PATTERNS
                               if hasattr(config, 'OUTPUT_PATTERNS') else {})
            prediction_steps = getattr(config, 'PREDICTION_STEPS', 1)
            subsample_factor = getattr(config, 'SUBSAMPLE_FACTOR', 30)

            figures['timeseries_predictions'] = plot_timeseries_chained(
                predictions_dict, dynamic_cols, all_dynamic, output_patterns,
                prediction_steps=prediction_steps,
                subsample_factor=subsample_factor,
                config=config,
                save_path=self._get_save_path('timeseries_predictions'),
            )

            # Near-constant types are categories D + E in Phase 1, or
            # FLOW_SEC/PRESSURE_SEC/POWER outputs in Phases 5/6.
            nearconst_types: List[str] = []
            if hasattr(config, 'CATEGORY_D'):
                nearconst_types.extend(config.CATEGORY_D)
            if hasattr(config, 'CATEGORY_E'):
                nearconst_types.extend(config.CATEGORY_E)
            if hasattr(config, 'FLOW_SEC_OUTPUTS'):
                nearconst_types.extend(config.FLOW_SEC_OUTPUTS)
            if hasattr(config, 'PRESSURE_SEC_OUTPUTS'):
                nearconst_types.extend(config.PRESSURE_SEC_OUTPUTS)
            if hasattr(config, 'POWER_OUTPUTS'):
                nearconst_types.extend(config.POWER_OUTPUTS)
            # Deduplicate while preserving order
            nearconst_types = list(dict.fromkeys(nearconst_types))

            if nearconst_types:
                figures['timeseries_nearconst'] = plot_timeseries_nearconst(
                    predictions_dict, dynamic_cols, nearconst_types,
                    output_patterns,
                    prediction_steps=prediction_steps,
                    subsample_factor=subsample_factor,
                    config=config,
                    save_path=self._get_save_path('timeseries_nearconst'),
                )

            figures['delta_quality'] = plot_delta_quality(
                predictions_dict, metrics_df, dynamic_cols, all_dynamic,
                output_patterns=output_patterns,
                config=config,
                save_path=self._get_save_path('delta_quality'),
            )

        return figures

    # ────────────────────────────── Main evaluation suite ──────────────────
    def plot_evaluation_suite(
        self,
        metrics_df: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        phase: str = "1",
        predictions_dict: Optional[Dict[str, np.ndarray]] = None,
        column_info: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Generate the complete evaluation visualization suite.

        Phase-aware grouping:
            - Phases 1     → group_by='Category', color_by='Category'
            - Phases 2/3   → group_by='Pathway',  color_by='Pathway' (if present)
            - Phase 4      → group_by='Domain',   color_by='Domain'  (if present)
            - Phases 5/6   → group_by='Group',    color_by='Group'   (if present)

        Optional `predictions_dict`/`column_info`/`config` enable the extended
        time-series and delta-quality plots.
        """
        figures: Dict[str, plt.Figure] = {}

        # Phase-aware grouping
        if phase in ("5", "6"):
            group_by = 'Group'
        elif phase == "4":
            group_by = 'Domain'
        elif phase in ("2", "3"):
            group_by = 'Pathway'
        else:
            group_by = 'Category'

        # Fall back gracefully if expected grouping column is absent
        if group_by not in metrics_df.columns and 'Category' in metrics_df.columns:
            group_by = 'Category'

        figures['r2_distribution'] = self.plot_r2_dist(metrics_df, group_by)
        color_by = group_by if group_by in metrics_df.columns else None
        figures['r2_by_type'] = self.plot_r2_types(metrics_df, color_by=color_by)
        figures['variance_ratio'] = self.plot_variance(metrics_df)

        figures['rmse_by_type'] = plot_rmse_by_type(
            metrics_df, color_by=color_by,
            save_path=self._get_save_path('rmse_by_type'),
        )
        figures['correlation'] = plot_correlation_by_type(
            metrics_df, color_by=color_by,
            save_path=self._get_save_path('correlation'),
        )
        figures['model_vs_persistence'] = plot_model_vs_persistence(
            metrics_df, save_path=self._get_save_path('model_vs_persistence'),
        )

        if phase in ("5", "6") and 'Group' in metrics_df.columns:
            figures['r2_by_group'] = self.plot_r2_groups(metrics_df)

        if 'Loop' in metrics_df.columns:
            figures['loop_dichotomy'] = plot_loop_dichotomy(
                metrics_df, save_path=self._get_save_path('loop_dichotomy'),
            )

        if 'CDU' in metrics_df.columns:
            figures['r2_heatmap'] = self.plot_heatmap(metrics_df)

        # Direct (predictions, targets) array path
        if predictions is not None and targets is not None:
            figures['predictions'] = self.plot_preds(predictions, targets)
            figures['error_distribution'] = self.plot_errors(predictions, targets)
            figures['parity_plots'] = self.plot_parity(predictions, targets)

        # Dict-based time-series + delta-quality
        if predictions_dict is not None and column_info is not None and config is not None:
            extended = self.plot_evaluation_suite_extended(
                metrics_df, predictions_dict, column_info, config,
            )
            # Only merge keys that are not already present (avoid duplicate work)
            for k, v in extended.items():
                figures.setdefault(k, v)

        return figures