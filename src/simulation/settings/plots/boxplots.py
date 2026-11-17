"""
Boxplots of final-round metrics across runs (one dot per run).
  - plot_boxplots:        one panel per metric, one box per algorithm.
                          Colour encodes the role: reference / federated / centralized.
  - plot_boxplots_by_cfg: same layout, colour encodes the sample configuration.
"""
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.simulation.settings.statistics import final_round_table

logger = logging.getLogger(__name__)

# Validated colour-blind-safe categorical slots (fixed order)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ROLE_COLORS = {"reference": "#2a78d6", "federated": "#eb6834", "centralized": "#8a8a85"}
LABELS = {"r2": "R² score", "mse": "MSE", "exec_time": "Time per round (s)",
          "r2_macro": "R² (mean over nodes)", "r2_worst": "R² (worst node)",
          "w_rel_error": "‖W − W*‖ / ‖W*‖"}
LOG_METRICS = {"exec_time", "w_rel_error"}


def _role(alg: str, reference: str) -> str:
    if alg == reference:
        return "reference"
    return "federated" if alg.startswith("Fed") else "centralized"


def _order(algs: List[str], reference: str) -> List[str]:
    """Reference first, then federated, then centralized (alphabetical inside)."""
    rank = {"reference": 0, "federated": 1, "centralized": 2}
    return sorted(algs, key=lambda a: (rank[_role(a, reference)], a))


def _style(ax, metric):
    ax.set_ylabel(LABELS.get(metric, metric), fontsize=9)
    if metric in LOG_METRICS:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    for side in ("top", "right", "bottom", "left"):  # full frame
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(labelsize=8)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[Boxplot] Saved {path}.png/.pdf")


def _box(ax, data, positions, width, color, alpha=0.35):
    bp = ax.boxplot(data, positions=positions, widths=width, patch_artist=True, showfliers=False,
                    medianprops=dict(color="#1a1a19", linewidth=1.1),
                    whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(alpha)


def plot_boxplots(histories: List[Dict], metrics: List[str], store_path: Path,
                  reference: str = "FedDelta") -> None:
    tables = {m: final_round_table(histories, m) for m in metrics}
    tables = {m: t for m, t in tables.items() if not t.empty}
    if not tables:
        return
    rng = np.random.default_rng(0)
    n_algs = max(t.shape[1] for t in tables.values())

    if n_algs > 8:
        fig, axes = plt.subplots(len(tables), 1, figsize=(max(6, 0.5 * n_algs), 3.0 * len(tables)),
                                 squeeze=False)
        axes = axes.T
    else:
        fig, axes = plt.subplots(1, len(tables), figsize=(max(3.6, 0.42 * n_algs + 1.2) * len(tables), 3.8),
                                 squeeze=False)

    for ax, (metric, df) in zip(axes[0], tables.items()):
        algs = _order(list(df.columns), reference)
        for i, a in enumerate(algs, start=1):
            color = ROLE_COLORS[_role(a, reference)]
            vals = df[a].to_numpy()
            _box(ax, [vals], [i], 0.6, color)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=11, color=color,
                       edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_xticks(range(1, len(algs) + 1))
        ax.set_xticklabels(algs, rotation=45, ha="right")
        _style(ax, metric)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, alpha=0.6) for c in ROLE_COLORS.values()]
    fig.legend(handles, [reference, "Federated baselines", "Centralized baselines"],
               loc="upper center", ncol=3, fontsize=8, frameon=False,
               title=f"Final-round values across {len(histories)} runs", title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95 if n_algs > 8 else 0.92))
    _save(fig, store_path)


def plot_boxplots_by_cfg(histories_by_cfg: Dict[str, List[Dict]], metrics: List[str],
                         store_path: Path, reference: str = "FedDelta") -> None:
    """x = algorithm, colour = sample configuration (few configs -> few colours)."""
    cfgs = list(histories_by_cfg.keys())[:len(PALETTE)]
    if not cfgs:
        return
    tables = {m: {c: final_round_table(histories_by_cfg[c], m) for c in cfgs} for m in metrics}
    all_algs = sorted({a for t in tables.values() for df in t.values() for a in df.columns})
    algs = _order(all_algs, reference)
    width = 0.8 / len(cfgs)

    fig, axes = plt.subplots(len(metrics), 1, figsize=(max(6, 0.6 * len(algs) * len(cfgs) ** 0.5), 3.2 * len(metrics)),
                             squeeze=False)
    for ax, metric in zip(axes[:, 0], metrics):
        for j, c in enumerate(cfgs):
            df = tables[metric][c]
            pos, data = [], []
            for i, a in enumerate(algs):
                if a in df.columns:
                    pos.append(i + (j - (len(cfgs) - 1) / 2) * width)
                    data.append(df[a].to_numpy())
            if data:
                _box(ax, data, pos, width * 0.85, PALETTE[j], alpha=0.6)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(algs, rotation=45, ha="right")
        _style(ax, metric)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PALETTE[j], alpha=0.6) for j in range(len(cfgs))]
    fig.legend(handles, [f"{c} (min:max:train)" for c in cfgs], loc="upper center",
               ncol=min(len(cfgs), 4), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, store_path)
