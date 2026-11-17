"""
Convergence figure of the paper (Fig. 2): test MSE and R² per round on the four datasets.

Reads the per-round metrics written by exp_convergence.py (SOURCE = "convergence",
results/convergence/framework/<tag>/...) or by exp_default.py (SOURCE = "default",
results/default/<dataset>/target_<target>/per_round_metrics/lambda<λ>/*_run*_rounds.csv)
and draws a 2 x 4 grid: MSE (top row) and R² (bottom row) for each dataset, with the
rounds on the x-axis of the R² row and a single legend for the whole figure.
Lines: mean over runs; band: ± one standard deviation over runs. Runs of different
length (datasets that run out of data) are cut to the shortest run, as in the tables.
All parameters are set at the top of main().

Outputs (results/<SOURCE>/figures/):
  convergence.pdf (+ .png preview)
  last_round.csv: MSE and R² of the last plotted round (mean and sd over runs). Only to
  check the figure: the values of the tables come from table.csv of exp_convergence.py,
  which evaluates the final models on the test sets of all nodes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

# Display name, colour, line style and marker of each algorithm (column name in the CSVs).
# Centralized references in black, red and grey (dashed/dotted lines); federated methods in a fixed, validated
# categorical order (colour-blind safe adjacent pairs), each also with its own marker.
STYLE = {
    "Cent":            ("Cent",       "#000000", "--", None),
    "CentRF":          ("CentRF",     "#a51c30", ":",  None),
    "CentMLP":         ("CentMLP",    "#8a8a8a", "-.", None),
    "FedDelta":        ("FedDelta",   "#2a78d6", "-",  "o"),
    "FedGD (FedAvg)":  ("FedAvg",     "#eb6834", "-",  "s"),
    #"FedGD (FedNova)": ("FedNova",    "#1baf7a", "-",  "^"),
    "FedGD (FedProx)": ("FedProx",    "#eda100", "-",  "D"),
    "FedSCAFFOLD":     ("SCAFFOLD",   "#e87ba4", "-",  "v"),
    "FedGD (FedAvgM)": ("FedAvgM",    "#008300", "-",  "P"),
    "FedMLP (FedAvg)": ("FedAvg-MLP", "#6250d6", "-",  "X"),
}


def find_folder(source: str, ds: str) -> Path:
    """Folder with the per-run CSVs of a dataset (the only lambda subfolder of the run)."""
    if source == "default":
        pattern = f"results/default/{ds}/target_*/per_round_metrics/lambda*"
    else:                                     # exp_convergence.py: framework/<tag>/<ds>/...
        pattern = f"results/{source}/framework/{ds}_*/{ds}/target_*/per_round_metrics/lambda*"
    found = sorted(ROOT.glob(pattern))
    if not found:
        raise FileNotFoundError(f"No results for {ds}: {ROOT / pattern}")
    if len(found) > 1:
        print(f"  {ds}: several result folders, using the most recent one: {found[-1]}")
        found.sort(key=lambda p: p.stat().st_mtime)
    return found[-1]


def load_runs(folder: Path) -> dict:
    """{algorithm: {metric: array (runs x rounds)}} from the per-run CSV files of a folder."""
    files = sorted(folder.glob("*_run*_rounds.csv"))
    if not files:
        raise FileNotFoundError(f"No *_run*_rounds.csv in {folder}")
    runs = []
    for f in files:
        df = pd.read_csv(f, header=[0, 1], index_col=0)
        df = df[pd.to_numeric(df.index, errors="coerce").notna()].astype(float)   # drop the "Round" row
        runs.append(df)
    n = min(len(r) for r in runs)                     # cut to the shortest run
    out = {}
    for alg, metric in runs[0].columns:
        arr = np.vstack([r[(alg, metric)].to_numpy()[:n] for r in runs])
        out.setdefault(alg, {})[metric] = arr
    return out


def last_round_rows(data: dict, label: str, algs: list) -> list:
    """Mean and sd over runs of the last plotted round, for each algorithm."""
    rows = []
    for alg in algs:
        if alg not in data:
            continue
        row = {"Dataset": label, "Algorithm": STYLE[alg][0], "Runs": None, "Rounds": None}
        for metric in ("r2", "mse"):
            if metric not in data[alg]:
                continue
            v = data[alg][metric][:, -1]
            row["Runs"], row["Rounds"] = len(v), data[alg][metric].shape[1]
            row[f"{metric.upper()} mean"] = np.nanmean(v)
            row[f"{metric.upper()} sd"] = np.nanstd(v, ddof=1) if len(v) > 1 else 0.0
        rows.append(row)
    return rows


def main():
    # ---- parameters ------------------------------------------------------
    DATASETS = {                      # folder name -> (label, MSE scale for the axis)
        "housing":               ("Housing", 1e9),
        "discharges":            ("SPARCS",  1e9),
        "european_survey_adult": ("EESE",    1.0),
        "length_of_stay":        ("LOS",     1.0),
    }
    SOURCE = "convergence"            # "convergence": exp_convergence.py (tuned)
                                      # "default": exp_default.py (results/default)
    ALGS = list(STYLE)                # order of drawing and of the legend
    FONT = 10                         # axis labels / titles
    TICK_FONT = 8.5
    FIGSIZE = (7.2, 4.0)              # inches; included at \textwidth
    R2_YLIM = (0.0, 1.0)              # common R² scale for all datasets (None: automatic)
    LW_FEDDELTA = 2.0                 # line widths (FedDelta wider, drawn under Cent)
    LW_CENT = 1.1
    LW_OTHERS = 1.0
    OUT = ROOT / "results" / SOURCE / "figures" / "convergence"
    # ----------------------------------------------------------------------

    plt.rcParams.update({"font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
                         "xtick.labelsize": TICK_FONT, "ytick.labelsize": TICK_FONT,
                         "legend.fontsize": FONT - 1, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, len(DATASETS), figsize=FIGSIZE, sharex="col")
    rows = []

    for j, (ds, (label, scale)) in enumerate(DATASETS.items()):
        folder = find_folder(SOURCE, ds)
        data = load_runs(folder)
        rows += last_round_rows(data, label, ALGS)
        for i, metric in enumerate(("mse", "r2")):
            ax = axes[i, j]
            for alg in ALGS:
                if alg not in data or metric not in data[alg]:
                    continue
                name, color, ls, marker = STYLE[alg]
                arr = data[alg][metric] / (scale if metric == "mse" else 1.0)
                m = arr.mean(axis=0)
                s = arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros_like(m)
                x = np.arange(1, len(m) + 1)
                # FedDelta is drawn wider and underneath Cent so that both stay visible
                lw = LW_FEDDELTA if alg == "FedDelta" else (LW_CENT if alg.startswith("Cent") else LW_OTHERS)
                z = 2 if alg == "FedDelta" else 3
                ax.plot(x, m, color=color, ls=ls, lw=lw, marker=marker, ms=2.5,
                        markevery=max(1, len(x) // 6), label=name, zorder=z)
                if len(arr) > 1:
                    ax.fill_between(x, m - s, m + s, color=color, alpha=0.12, lw=0, zorder=1)
            ax.grid(True, color="#e6e6e6", lw=0.6)
            ax.set_axisbelow(True)
            for side in ax.spines.values():       # closed frame around each panel
                side.set_visible(True)
                side.set_linewidth(0.8)
            if i == 0:
                ax.set_title(label)
            if i == 1:
                ax.set_xlabel("Round")
                ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=5))
        if scale != 1.0:                 # unit of the MSE axis, as in Table 4
            axes[0, j].text(0.0, 1.02, r"$\times10^{9}$", transform=axes[0, j].transAxes,
                            fontsize=TICK_FONT, ha="left", va="bottom")
    if R2_YLIM is not None:            # same R² scale in every panel
        for ax in axes[1]:
            ax.set_ylim(*R2_YLIM)
            ax.set_yticks(np.arange(R2_YLIM[0], R2_YLIM[1] + 1e-9, 0.2))
    axes[0, 0].set_ylabel("MSE")
    axes[1, 0].set_ylabel(r"$R^2$")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.4, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {OUT.with_suffix('.pdf')}")

    csv = OUT.with_name("last_round.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"Saved {csv}")


if __name__ == "__main__":
    main()
