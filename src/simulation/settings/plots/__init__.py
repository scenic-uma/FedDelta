from __future__ import annotations
 
import logging
from typing import List, Optional
from matplotlib.ticker import MaxNLocator
 
import matplotlib.pyplot as plt
import numpy as np
 
logger = logging.getLogger(__name__)
 
# -----------------------------------------------------------------------------
#  Visual style tables
#  Each algorithm gets a fixed line style and colour so plots are consistent
#  across all figure types.
# -----------------------------------------------------------------------------
_COLOR_PALETTE = [
    "#0095ff", "#238023",
    #"#00ff0dff",
    #"#000000",
    "#d334d8", "#FFE600", "#7A0E00", "#7A007A",
    "#ff7f0e", "#17becf",
]

_LINE_STYLE_PALETTE = [
    {"linestyle": "-",  "linewidth": 1.2, "marker": "o", "markersize": 3},
    {"linestyle": "--", "linewidth": 2.0, "marker": "s", "markersize": 3.5},
    {"linestyle": "-.", "linewidth": 1.0, "marker": "^", "markersize": 2},
    {"linestyle": ":", "linewidth": 2.0,  "marker": "d", "markersize": 3},
    {"linestyle": "-",  "linewidth": 0.7, "marker": "P", "markersize": 2},
    {"linestyle": "--", "linewidth": 0.7, "marker": "v", "markersize": 3},
    {"linestyle": "-.", "linewidth": 1.2, "marker": "x", "markersize": 3},
    {"linestyle": ":",  "linewidth": 1.2, "marker": "*", "markersize": 3.5},
]

_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
 
# Keys that appear in history dicts but are NOT algorithm names
_NON_ALGORITHM_KEYS = {"Total samples", "lambda_reg", "node_status", "effective_rounds", "run_seed", "faults", "integrity", "n_features", "n_nodes"}
 
# Colour palette used when sweeping over a parameter (one colour per value)
_SWEEP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# =============================================================================
#  PRIVATE HELPERS
# =============================================================================
# Fixed style per algorithm: same look in every figure, no repeated colours.
_ALG_STYLES = {
    "FedDelta":         {"color": "#2a78d6", "linestyle": "-",  "linewidth": 1.6, "marker": "o", "markersize": 3},
    "FedGD (FedAvg)":   {"color": "#eb6834", "linestyle": "-",  "linewidth": 1.0, "marker": "s", "markersize": 3},
    "FedGD (FedNova)":  {"color": "#1baf7a", "linestyle": "-",  "linewidth": 1.0, "marker": "^", "markersize": 3},
    "FedGD (FedProx)":  {"color": "#eda100", "linestyle": "-",  "linewidth": 1.0, "marker": "D", "markersize": 3},
    "FedSCAFFOLD":      {"color": "#e87ba4", "linestyle": "-",  "linewidth": 1.0, "marker": "v", "markersize": 3},
    "FedGD (FedAvgM)":  {"color": "#008300", "linestyle": "-",  "linewidth": 1.0, "marker": "P", "markersize": 3},
    "FedMLP (FedAvg)":  {"color": "#6250d6", "linestyle": "-",  "linewidth": 1.0, "marker": "X", "markersize": 3},
    "FedMLP (FedNova)": {"color": "#9a3fb5", "linestyle": "-",  "linewidth": 1.0, "marker": "*", "markersize": 3.5},
    "Cent":             {"color": "#1a1a19", "linestyle": "--", "linewidth": 1.4, "marker": "", "markersize": 0, "zorder": 5},
    "CentRF":           {"color": "#6b6b66", "linestyle": ":",  "linewidth": 1.6, "marker": "", "markersize": 0},
    "CentMLP":          {"color": "#a3a39c", "linestyle": "-.", "linewidth": 1.4, "marker": "", "markersize": 0},
}

_LABELS = {
    "FedGD (FedAvg)": "FedAvg", "FedGD (FedProx)": "FedProx", "FedGD (FedNova)": "FedNova",
    "FedGD (FedAvgM)": "FedAvgM", "FedSCAFFOLD": "SCAFFOLD", "FedMLP (FedAvg)": "FedAvg-MLP",
}

_FALLBACK_COLORS = plt.get_cmap("tab20").colors   # unknown algorithms

def _get_style(alg: str, algorithms: List[str]) -> dict:
    """Fixed colour and line style per algorithm name (never repeated in a figure)."""
    if alg in _ALG_STYLES:
        return dict(_ALG_STYLES[alg])
    unknown = [a for a in algorithms if a not in _ALG_STYLES]
    idx = unknown.index(alg) if alg in unknown else 0
    return {"color": _FALLBACK_COLORS[(2 * idx) % 20 + (idx // 10) % 2],
            **_LINE_STYLE_PALETTE[idx % len(_LINE_STYLE_PALETTE)]}

def _metric_ylabel(metric: str) -> str:
    """
    Return an axis label for a given metric key.
    """
    return {"mse": "MSE", "r2": "R² score", "exec_time": "Time (s)"}.get(
        metric, metric.upper()
    )

def _rewrite_label(name: str) -> str:
    """Display name of an algorithm, as used in the paper."""
    return _LABELS.get(name, name)
 
def _format_ax(
    ax,
    metric: str,
    max_rounds: int,
    all_values: Optional[list] = None,
) -> None:
    """
    Publication style for one subplot: y-axis on the left, light grid, small
    fonts, integer rounds, scientific notation for large values and
    metric-specific limits (R² in [0, 1]; MSE/time clipped to ignore early spikes).
    """
    ax.set_ylabel(_metric_ylabel(metric), fontsize=8)
    ax.set_xlim(1, max_rounds)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    ax.grid(True, linestyle="-", linewidth=0.4, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    ax.yaxis.get_offset_text().set_fontsize(7)

    if metric.startswith("r2"):
        ax.set_ylim(0, 1)
    elif metric in ("mse", "exec_time") and all_values:
        # Use a robust upper bound that ignores early spikes
        upper = visual_ylim(all_values, skip_first=1, upper_percentile=97.5)
        if upper:
            ax.set_ylim(0, upper)

def _save_or_show(fig, store_path: Optional[str], store_plots: bool) -> None:
    """
    Save the figure to <store_path>.png and <store_path>.pdf (300 dpi),
    or call plt.show() when saving is disabled.
    """
    if store_path and store_plots:
        plt.savefig(f"{store_path}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{store_path}.pdf", dpi=300, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()
 
def _global_legend(fig, reference_ax, n_cols: int, bbox_y: float = 0.0) -> None:
    """
    Single shared legend below the figure, without frame (never covers curves;
    savefig uses bbox_inches="tight").
    """
    handles, labels = reference_ax.get_legend_handles_labels()
    if not labels:
        return
    ncol = min(len(labels), 3 if fig.get_figwidth() < 6 else 6)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        fontsize=7,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, bbox_y),
    )

def _extract_algorithms(history: dict) -> List[str]:
    """
    Return sorted algorithm names, excluding bookkeeping keys.
    """
    return sorted(k for k in history.keys() if k not in _NON_ALGORITHM_KEYS)
 
# =============================================================================
#  STANDALONE UTILITY
# =============================================================================
def visual_ylim(
    values,
    skip_first: int = 1,
    upper_percentile: float = 97.5,
    margin: float = 1.35,
) -> Optional[float]:
    """
    Compute a robust upper y-limit that ignores early transient spikes.
 
    Strategy
    --------
    1. Drop the first `skip_first` values (common warm-up artefact).
    2. Take the `upper_percentile` of the remainder.
    3. Multiply by `margin` to leave headroom.
 
    Returns None when there are too few data points to be meaningful.
    """
    values = np.asarray(values)
    if len(values) <= skip_first + 3:
        return None
    trimmed = values[skip_first:]
    return np.percentile(trimmed, upper_percentile) * margin

