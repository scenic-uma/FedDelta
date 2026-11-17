from pathlib import Path
from matplotlib.ticker import MaxNLocator
from typing import Dict, Optional
 
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np
 
logger = logging.getLogger(__name__)

def save_erank_analysis_csv(
    history: Dict,
    run_idx: int,
    alg_name: str = "FedDelta",
    store_path: Optional[str] = None,
) -> None:
    """
    Persist per-round effective rank and R² data for a single run to CSV.
 
    Columns: round, erank_update, erank_global, r2
    """
    global_data = history[alg_name]["global"]
 
    r2         = global_data.get("r2", [])
    erank_upd  = global_data.get("erank_round", [])
    erank_glob = global_data.get("erank_global", [])
 
    # Align all series to the shortest length
    min_len = min(len(r2), len(erank_upd), len(erank_glob))
 
    df = pd.DataFrame({
        "round":        np.arange(1, min_len + 1),
        "erank_update": erank_upd[:min_len],
        "erank_global": erank_glob[:min_len],
        "r2":           r2[:min_len],
    })
 
    out = Path(f"{store_path}_analysis_run{run_idx + 1}.csv")
    df.to_csv(out, index=False)
    logger.info("[SYSTEM] Erank analysis CSV saved.")

def plot_erank_analysis(
    history: Dict,
    run_idx: int,
    alg_name: str = "FedDelta",
    store_plots: bool = False,
    store_path: Optional[str] = None,
) -> None:
    """
    Visualise the relationship between effective rank and R² for ONE run.
    """
    if alg_name not in history or "global" not in history[alg_name]:
        logger.warning("Algorithm %s not found in history — skipping erank plot.", alg_name)
        return

    global_data = history[alg_name]["global"]
    r2         = np.array(global_data.get("r2", []))
    erank_upd  = np.array(global_data.get("erank_round", []))
    erank_glob = np.array(global_data.get("erank_global", []))

    # Guard: all three series must be non-empty
    if not (len(r2) and len(erank_upd) and len(erank_glob)):
        logger.warning("Insufficient erank data for run %d — skipping.", run_idx)
        return

    # Align lengths
    min_len = min(len(r2), len(erank_upd), len(erank_glob))
    r2 = r2[:min_len]
    erank_upd = erank_upd[:min_len]
    erank_glob = erank_glob[:min_len]
    rounds = np.arange(1, min_len + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # ------------------------------------------------------------------
    # Subplot 1: Erank bars + dual-axis R² line
    # ------------------------------------------------------------------
    ax1 = axes[0]
    ax2 = ax1.twinx()

    ax1.bar(
        rounds,
        erank_upd,
        alpha=0.35,
        color="gray",
        label="Erank(update)",
        zorder=1,
    )

    ax1.plot(
        rounds,
        erank_glob,
        color="orange",
        linewidth=2,
        marker="o",
        markersize=5,
        label="Erank(global)",
        zorder=2,
    )

    ax1.set_ylabel("Effective Rank", fontsize=10)
    ax1.set_xlabel("Communication Round", fontsize=10)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.tick_params(axis="both", labelsize=9)

    ax2.plot(
        rounds,
        r2,
        color="steelblue",
        linewidth=2,
        marker="s",
        markersize=5,
        label="R²",
        zorder=3,
    )

    ax2.set_ylabel("R² score", color="steelblue", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax2.axhline(0, color="steelblue", linestyle=":", alpha=0.4)

    # ------------------------------------------------------------------
    # Global legend data (subplot 1)
    # ------------------------------------------------------------------
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    all_handles = lines1 + lines2
    all_labels = labels1 + labels2

    ax1.set_title(f"Erank vs R² — Run {run_idx + 1}", fontsize=10)

    # ------------------------------------------------------------------
    # Subplot 2: Scatter Erank(global) vs R²
    # ------------------------------------------------------------------
    ax3 = axes[1]

    scatter = ax3.scatter(
        erank_glob,
        r2,
        c=rounds,
        cmap="viridis",
        s=80,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )

    if len(erank_glob) > 1:
        z = np.polyfit(erank_glob, r2, 1)
        p = np.poly1d(z)
        x_line = np.linspace(erank_glob.min(), erank_glob.max(), 100)
        ax3.plot(
            x_line,
            p(x_line),
            color="red",
            linestyle="--",
            linewidth=1.5,
            label=f"Trend (slope={z[0]:.3f})",
        )

    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax3.set_xlabel("Erank(global)", fontsize=10)
    ax3.set_ylabel("R² score", fontsize=10)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.tick_params(axis="both", labelsize=9)
    ax3.legend(fontsize=9)

    plt.colorbar(scatter, ax=ax3, label="Round")

    ax3.set_title(f"Erank(global) vs R² — Run {run_idx + 1}", fontsize=10)

    # ------------------------------------------------------------------
    # Global figure legend + title
    # ------------------------------------------------------------------
    fig.legend(
        all_handles,
        all_labels,
        loc="upper center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.suptitle(
        f"Effective Rank Analysis — Run {run_idx + 1}",
        fontsize=11,
        y=1.08,
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # ------------------------------------------------------------------
    # Save or show
    # ------------------------------------------------------------------
    if store_path and store_plots:

        Path(store_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(f"{store_path}_analysis_run{run_idx + 1}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{store_path}_analysis_run{run_idx + 1}.pdf", dpi=300, bbox_inches="tight")
        plt.close("all")
    else:
        plt.show()