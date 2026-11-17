from typing import List, Dict, Optional
from pathlib import Path

import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from src.simulation.settings.plots import _format_ax, _global_legend, _save_or_show, _extract_algorithms, _rewrite_label, _get_style
from src.simulation.config import RESULTS_DIR
# =============================================================================
#  PUBLIC PLOTTING FUNCTIONS
# =============================================================================

def plot_metrics_summary(
    all_histories: List[Dict],
    metrics: List[str],
    store_plots: bool = False,
    store_path: Optional[str] = None,
    orientation: str = "horizontal",
    only_fed_algorithms: bool = False,
) -> None:
    """
    Plot mean ± standard deviation of metrics across MULTIPLE runs.
 
    Aggregates N independent experiment histories and shows the central
    tendency (mean) plus uncertainty band (±1 std) for each algorithm.
 
    Optional features
    -----------------
    zoom_exec_time : add an extra subplot that hides CentRF for scale clarity
    zoom_r2        : embed an inset that zooms into the stable region of R²
    only_fed_algorithms : restrict exec_time subplot to Fed* algorithms only
 
    Parameters
    ----------
    all_histories : list of per-run history dicts (same structure as `history`)
    """
    # Extract algorithm names from the first run (assumed consistent across runs)
    algorithms = _extract_algorithms(all_histories[0])
    display_name = {}

    for alg in algorithms:
        name = _rewrite_label(alg)
        display_name[alg] = name

    n_subplots = len(metrics)
    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    if orientation == "vertical":
        fig, axes = plt.subplots(n_subplots, 1, figsize=(4.2, 2.1 * n_subplots), sharex=True)
    else:
        fig, axes = plt.subplots(1, n_subplots, figsize=(3.3 * n_subplots, 2.6))
 
    if not isinstance(axes, np.ndarray):
        axes = [axes]
 
    metric_axes = axes[:len(metrics)]       # one axis per metric
 
    # ------------------------------------------------------------------
    # Loop: one subplot per metric
    # ------------------------------------------------------------------
    for ax, metric in zip(metric_axes, metrics):
        max_rounds = 0
        stored_data: Dict[str, tuple] = {}  # alg → (rounds, mean, std)
        all_values: List[float] = []
 
        for alg in algorithms:
            # For exec_time, optionally skip non-federated baselines
            if only_fed_algorithms and metric == "exec_time" and not alg.startswith("Fed"):
                continue
 
            # Collect this metric across all runs for the current algorithm
            runs = [
                h[alg]["global"].get(metric, [])
                for h in all_histories
                if h[alg]["global"].get(metric)
            ]
            if not runs:
                continue
 
            # Align run lengths to the shortest one (handles early-stop cases)
            min_len = min(len(r) for r in runs)
            runs_arr = np.array([r[:min_len] for r in runs])
 
            mean_v = np.mean(runs_arr, axis=0)

            if len(runs) > 1:
                std_v  = np.std(runs_arr, axis=0, ddof=1)   # unbiased std
            else:
                std_v = np.zeros_like(mean_v)
    
            rounds = np.arange(1, len(mean_v) + 1)
            max_rounds = max(max_rounds, len(mean_v))
 
            stored_data[alg] = (rounds, mean_v, std_v)
            all_values.extend(mean_v.tolist())
 
            # -- Mean line --
            style = _get_style(alg, algorithms)
            ax.plot(
                rounds, mean_v,
                label=display_name[alg],
                markevery=max(1, len(rounds) // 10),
                **style,
            )
            if len(runs) > 1:
                # -- Uncertainty band (±1 std) --
                ax.fill_between(
                    rounds,
                    mean_v - std_v,
                    mean_v + std_v,
                    color = style["color"],
                    alpha = 0.15,
                    linewidth = 0,
                )
 
        # Shared axis formatting
        _format_ax(ax, metric, max_rounds, all_values)
 
    # ------------------------------------------------------------------
    # Global legend and layout
    # ------------------------------------------------------------------
    n_labels = len(metric_axes[0].get_legend_handles_labels()[1])
    _global_legend(fig, metric_axes[0], n_labels)

    if orientation == "horizontal":
        for ax in metric_axes:
            ax.set_xlabel("Round", fontsize=8)
        fig.tight_layout(w_pad=1.5)
    else:
        metric_axes[-1].set_xlabel("Round", fontsize=8)
        fig.tight_layout(h_pad=0.6)

    _save_or_show(fig, store_path, store_plots)

def plot_cfg_comparison(
    per_cfg_histories: Dict[str, dict],
    metrics: List[str],
    store_plots: bool = False,
    store_path: Optional[Path] = None,
) -> None:
    """
    Line plot: evolution across rounds for each sample_cfg (FedDelta only).
    Rounds aligned to the minimum common length across all configs.
    """
    cfg_keys = list(per_cfg_histories.keys())

    min_rounds = min(
        len(per_cfg_histories[key]["global"].get(metric, []))
        for key in cfg_keys
        for metric in metrics
        if per_cfg_histories[key]["global"].get(metric)
    )

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        all_values = []

        for key in cfg_keys:
            values = per_cfg_histories[key]["global"].get(metric, [])
            if not values:
                continue
            values = values[:min_rounds]
            rounds = np.arange(1, min_rounds + 1)
            all_values.extend(values)
            ax.plot(rounds, values, label=key, linewidth=1.5)

        ax.set_xlabel("Communication Round", fontsize=10)
        ax.set_xlim(1, min_rounds)
        ax.grid(True, linestyle="--", alpha=0.4)
        _format_ax(ax, metric, min_rounds, all_values)

    _global_legend(fig, axes[0], n_cols=len(cfg_keys))

    if store_plots and store_path is not None:
        store_path.mkdir(parents=True, exist_ok=True)

    _save_or_show(fig, store_path / "cfg_comparison", store_plots)