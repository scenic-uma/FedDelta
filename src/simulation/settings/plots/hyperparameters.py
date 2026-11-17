from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging

from src.simulation.settings.plots import _SWEEP_COLORS, _format_ax, _global_legend, _save_or_show
 
logger = logging.getLogger(__name__)

def plot_hyperparameter_comparison(
    all_histories: List[Dict],
    parameter: str,
    store_plots: bool = False,
    store_path: Optional[str] = None,
    show_std: bool = True,
    store_csv: bool = False,
    lambda_reg: Optional[float] = None,
) -> None:
    """
    Compare FedDelta performance across different values of a single parameter.
    """
    # ------------------------------------------------------------------
    # Optional pre-filter: only keep runs with a specific lambda_reg value
    # ------------------------------------------------------------------
    if lambda_reg is not None:
        all_histories = [h for h in all_histories if h.get("lambda_reg") == lambda_reg]
 
    # ------------------------------------------------------------------
    # Group histories by the parameter value being swept
    # ------------------------------------------------------------------
    grouped: Dict = {}
    for h in all_histories:
        grouped.setdefault(h.get(parameter), []).append(h)
 
    # Sort parameter values numerically; fall back to last for non-numeric
    def _sort_key(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.inf
 
    sorted_params = sorted(grouped.keys(), key=_sort_key)
 
    # ------------------------------------------------------------------
    # Figure: two subplots (R² and execution time)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    csv_rows: List[Dict] = []
 
    for ax, metric in zip(axes, ["r2", "exec_time"]):
        max_rounds = 0
 
        for i, param_val in enumerate(sorted_params):
            color = _SWEEP_COLORS[i % len(_SWEEP_COLORS)]
 
            # Collect metric values from all runs that share this parameter value
            runs = [
                h["FedDelta"]["global"].get(metric, [])
                for h in grouped[param_val]
                if "FedDelta" in h
                and "global" in h["FedDelta"]
                and h["FedDelta"]["global"].get(metric)
            ]
            if not runs:
                continue
 
            # Align run lengths
            min_len = min(len(r) for r in runs)
            runs_arr = np.array([r[:min_len] for r in runs])
            mean_v = np.mean(runs_arr, axis=0)
            std_v  = np.std(runs_arr, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean_v)
            rounds = np.arange(1, len(mean_v) + 1)
            max_rounds = max(max_rounds, len(mean_v))
 
            ax.plot(rounds, mean_v, label=f"{parameter}={param_val}", color=color)
 
            if show_std:
                ax.fill_between(
                    rounds, mean_v - std_v, mean_v + std_v, alpha=0.2, color=color
                )
 
            # ----------------------------------------------------------
            # Accumulate CSV rows (built metric-by-metric, then merged)
            # ----------------------------------------------------------
            if metric == "r2":
                for r, m, s in zip(rounds, mean_v, std_v):
                    csv_rows.append({
                        parameter: param_val,
                        "round": r,
                        "mean_r2": m,
                        "std_r2": s,
                        "mean_exec_time": None,   # filled in the exec_time pass
                        "std_exec_time": None,
                    })
            else:  # exec_time — patch the rows already created during the R² pass
                for row, m, s in zip(
                    [row for row in csv_rows if row[parameter] == param_val],
                    mean_v,
                    std_v,
                ):
                    row["mean_exec_time"] = m
                    row["std_exec_time"] = s
 
        _format_ax(ax, metric, max_rounds)
 
    # ------------------------------------------------------------------
    # Global legend and layout
    # ------------------------------------------------------------------
    n_labels = len(axes[0].get_legend_handles_labels()[1])
    _global_legend(fig, axes[0], n_labels)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    _save_or_show(fig, store_path, store_plots)
 
    # Optionally persist the aggregated table
    if store_csv and store_path:
        df = pd.DataFrame(csv_rows)
        out = f"{store_path}.csv"
        df.to_csv(out, index=False)
        logger.info("[SYSTEM] Parameter variability CSV saved.")