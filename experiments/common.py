"""
Shared utilities for the experiments.
"""

import copy
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_ROOT = ROOT / "results"

# Target variable and display label for each dataset
DATASETS = {
    "housing":               {"target": "median_house_value", "label": "Housing"},
    "discharges":            {"target": "Total Charges",      "label": "SPARCS"},
    "european_survey_adult": {"target": "G21",                "label": "EESE"},
    "length_of_stay":        {"target": "lengthofstay",       "label": "LOS"},
}

# Algorithm definitions and plotting styles
FEDDELTA = "FedDelta"
CENT = "Cent"

# Short labels for tables and figures
SHORT = {
    "FedGD (FedAvg)"  : "FedAvg", 
    "FedGD (FedProx)" : "FedProx", 
    "FedGD (FedNova)" : "FedNova",
    "FedGD (FedAvgM)" : "FedAvgM", 
    "FedSCAFFOLD"     : "SCAFFOLD", 
    "FedMLP (FedAvg)" : "FedMLP",
    "Cent"            : "Centralized", 
    "CentRF"          : "Cent. RF", 
    "CentMLP"         : "Cent. MLP", 
    "FedDelta"        : "FedDelta",
}

# Consistent algorithm colours across figures
COLORS = {
    "FedDelta": "#2a78d6", "FedGD (FedAvg)": "#eb6834", "FedGD (FedNova)": "#1baf7a",
    "FedGD (FedProx)": "#eda100", "FedSCAFFOLD": "#e87ba4", "FedGD (FedAvgM)": "#008300",
    "FedMLP (FedAvg)": "#6250d6", "Cent": "#4a4a46", "CentRF": "#8a8a85", "CentMLP": "#b5b5ae",
}
MARKERS = {"FedDelta": "o", "FedGD (FedAvg)": "s", "FedGD (FedNova)": "^", "FedGD (FedProx)": "D",
           "FedSCAFFOLD": "v", "FedGD (FedAvgM)": "P", "FedMLP (FedAvg)": "X",
           "Cent": "", "CentRF": "", "CentMLP": ""}

LINES = {"Cent": "--", "CentRF": ":", "CentMLP": "-."}


def short(alg: str) -> str:
    """
    Return the short display name of an algorithm.
    """
    return SHORT.get(alg, alg)


def style(alg: str) -> dict:
    """
    Return the plotting style associated with an algorithm.
    """
    return {
        "color": COLORS.get(alg, "#555555"), 
        "marker": MARKERS.get(alg, "o") or None,
        "linestyle": LINES.get(alg, "-"), 
        "linewidth": 1.4, 
        "markersize": 4
    }

# Global Matplotlib style settings
plt.rcParams.update({
    "font.size": 8, 
    "axes.titlesize": 9, 
    "axes.labelsize": 8, 
    "legend.fontsize": 7,
    "xtick.labelsize": 7, 
    "ytick.labelsize": 7, 
    "axes.grid": True, 
    "grid.alpha": 0.3,
    "grid.linestyle": "--", 
    "savefig.dpi": 300, 
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Configuration and experiment execution
# ---------------------------------------------------------------------------

def exp_dir(name: str) -> Path:
    """
    Create and return the experiment output directories.
    """
    d = OUT_ROOT / name
    
    # Create directories for raw results, configurations, logs, and framework outputs
    for sub in ("raw", "configs", "logs", "framework"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d

def base_config() -> dict:
    """
    Load the shared YAML experiment configuration.
    """
    with open(ROOT / "config_experiment.yaml") as f:
        return yaml.safe_load(f)

def make_config(args, datasets: List[str], algorithms: List[str], **overrides) -> dict:
    """
    Build an experiment configuration from the YAML template.
    """
    c = copy.deepcopy(base_config())

    # Select datasets and their target variables
    c["datasets_list"] = {d: {"variable_target": DATASETS[d]["target"]} for d in datasets}

    # Validate the requested algorithm identifiers
    missing = [a for a in algorithms if a not in c["algorithms"]]
    if missing:
        raise KeyError(f"Algorithms not defined in config_experiment.yaml: {missing}")

    # Enable only the selected algorithms
    for name, params in c["algorithms"].items():
        params["use"] = name in algorithms

    # Set the number of runs, rounds, and output options
    c.update(
        num_runs=args.runs, 
        num_rounds=args.rounds, 
        verbose=False, 
        store_plots=True
    )

    # Optionally override hyperparameter tuning settings
    if args.tune != "yaml":
        c["tune_hyperparameters"] = args.tune == "on"

    # Reduce tuning rounds in quick mode
    if args.quick:
        c["tuning_rounds"] = 3

    # Set default node-level sample constraints if not already defined
    c.setdefault("node_samples_list", [{"node_min_samples": 20, "node_max_samples": 40,
                                         "min_samples_train": 25}])
    
    # Allow experiments to continue despite skipped rounds
    c["max_skipped_rounds"] = max(c.get("max_skipped_rounds", 3), 3)

    # Apply experiment-specific configuration overrides
    c.update(overrides)
    
    return c


def run(cfg: dict, tag: str, edir: Path, force: bool = False) -> Dict:
    """
    Execute the experiment in a separate process and return its histories.
    Results are loaded from cache when available.
    """

    # Define the cached output file
    out = edir / "raw" / f"{tag}.pkl"

    # Reuse cached results unless recomputation is requested
    if out.exists() and not force:
        with open(out, "rb") as f:
            return pickle.load(f)

    # Save the experiment configuration
    cfg_path = edir / "configs" / f"{tag}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    # Pass configuration and output paths to the subprocess
    env = {
        **os.environ,
        "FEDDELTA_CONFIG": str(cfg_path),
        "FEDDELTA_RESULTS_DIR": str(edir / "framework" / tag),
        "FEDDELTA_HISTORY_OUT": str(out)
    }

    # Capture execution logs and measure runtime
    log = edir / "logs" / f"{tag}.log"
    t0 = time.time()
    
    print(f"  -> running {tag} ...", flush=True)
    
    with open(log, "w") as f:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "exp_default.py")],
            cwd=ROOT, 
            env=env, 
            stdout=f, 
            stderr=subprocess.STDOUT
        )

    # Raise an error if execution failed or no output was generated
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"Run '{tag}' failed (exit {proc.returncode}). See {log}")
    
    print(f"     done in {time.time() - t0:.0f}s", flush=True)

    # Load and return the experiment histories
    with open(out, "rb") as f:
        return pickle.load(f)


def histories_of(res: Dict, dataset: str, condition: Optional[str] = None) -> List[Dict]:
    """
    Extract the run histories for a specific dataset and condition.
    """
    key = f"{dataset}|{DATASETS[dataset]['target']}"
    conds = res[key]

    # Use the first available condition when none is specified
    if condition is None:
        condition = next(iter(conds))
    
    return conds[condition]


# ---------------------------------------------------------------------------
# Metric extraction and statistical summaries
# ---------------------------------------------------------------------------

def final_values(histories: List[Dict], alg: str, metric: str) -> np.ndarray:
    """
    Extract the final metric value from each run.
    """
    
    vals = []
    
    for h in histories:
        a = h.get(alg, {})

        # Prefer the final evaluation over the last recorded round
        if metric in a.get("final", {}):
            vals.append(a["final"][metric])
        else:
            seq = a.get("global", {}).get(metric, [])
            vals.append(seq[-1] if seq else np.nan)
    
    return np.asarray(vals, dtype=float)


def curves(histories: List[Dict], alg: str, metric: str, cumulative: bool = False) -> np.ndarray:
    """
    Extract aligned run-by-round metric curves.
    """

    seqs = [np.asarray(h.get(alg, {}).get("global", {}).get(metric, []), dtype=float) for h in histories]

    # Exclude runs without recorded values
    seqs = [s for s in seqs if len(s)]
    
    if not seqs:
        return np.empty((0, 0))

    # Align all runs to the shortest available sequence
    L = min(len(s) for s in seqs)
    arr = np.vstack([s[:L] for s in seqs])

    # Optionally compute cumulative values
    return np.cumsum(arr, axis=1) if cumulative else arr

# ---------------------------------------------------------------------------
# Figure and table output
# ---------------------------------------------------------------------------

def save_fig(fig, path: Path):
    """
    Save a figure in PDF and PNG formats.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"     figure: {path.with_suffix('.pdf').relative_to(ROOT)}")


def save_table(df: pd.DataFrame, path: Path):
    """
    Save a DataFrame as CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path.with_suffix(".csv"), index=False)
    print(f"     table:  {path.with_suffix('.csv').relative_to(ROOT)}")


def legend_below(fig, axes, algs, ncol=None, bars=False):
    """
    Add a shared legend below the figure.
    """
    from matplotlib.patches import Patch

    # Use coloured patches for bar plots and line handles otherwise
    handles = ([Patch(color=COLORS.get(a, "#999")) for a in algs] if bars else
               [plt.Line2D([], [], **{k: v for k, v in style(a).items() if k != "markersize"},
                           markersize=4) for a in algs])
    
    fig.legend(handles, [short(a) for a in algs], loc="lower center",
               ncol=ncol or min(len(algs), 6), frameon=False, bbox_to_anchor=(0.5, -0.02))


def tag(args, *parts) -> str:
    """
    Generate a cache key from experiment settings.
    """
    return "_".join(str(p).replace(" ", "").replace("(", "").replace(")", "").replace(".", "p")
                    for p in parts) + f"_r{args.runs}_k{args.rounds}_t{args.tune}"