from typing import List, Dict, Optional
from src.simulation.config import ExperimentConfig
from src.simulation.settings.paths import summary_results_paths, erank_paths
from src.simulation.settings.plots.erank import plot_erank_analysis, save_erank_analysis_csv
from src.simulation.settings.plots.metrics import plot_metrics_summary
from src.simulation.settings.plots.boxplots import plot_boxplots
from src.simulation.settings.statistics import run_statistical_analysis
from pathlib import Path

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def summarize_last_round(histories: List[Dict]) -> pd.DataFrame:
    """
    Summarize the last round of a list of experiment histories.
    Adds mean and std per metric across runs.
    """
    if not histories:
        raise ValueError("The histories list is empty")

    exclude_metrics = {"erank_round", "erank_global", "nodes_drift", "subspace_alignment"}
    
    # Last common round PER (algorithm, metric) across runs.
    # (A global minimum made every algorithm NaN if one algorithm had 0 rounds.)
    min_rounds_per_metric = {}
    for run in histories:
        for alg, values in run.items():
            if not isinstance(values, dict) or "global" not in values:
                continue
            for metric, lst in values["global"].items():
                key = (alg, metric)
                if key not in min_rounds_per_metric:
                    min_rounds_per_metric[key] = len(lst)
                else:
                    min_rounds_per_metric[key] = min(min_rounds_per_metric[key], len(lst))
    
    # Last value of each metric in every run
    last_round_metrics_per_run = []
    for run in histories:
        run_summary = {}
        for alg, values in run.items():
            if not isinstance(values, dict) or "global" not in values:
                continue
            run_summary[alg] = {}
            for metric, lst in values["global"].items():
                if metric in exclude_metrics:
                    continue
                # Prefer the final evaluation on ALL nodes; time comes from the rounds
                final = values.get("final", {})
                if metric in final:
                    run_summary[alg][metric] = final[metric]
                    continue
                last_idx = min_rounds_per_metric.get((alg, metric), 0) - 1
                if lst and last_idx >= 0 and last_idx < len(lst):
                    run_summary[alg][metric] = lst[last_idx]
                else:
                    run_summary[alg][metric] = float("nan")

        last_round_metrics_per_run.append(run_summary)
    
    algorithms = last_round_metrics_per_run[0].keys()
    metrics = last_round_metrics_per_run[0][list(algorithms)[0]].keys()
    
    rows = []
    for alg in algorithms:
        row = {"Algorithm": alg}
        for metric in metrics:
            values = [run[alg][metric] for run in last_round_metrics_per_run]
            row[f"{metric}_mean"] = round(np.mean(values), 3)
            row[f"{metric}_std"] = round(np.std(values), 3)
        lambda_regs = [run.get("lambda_reg") for run in histories if "lambda_reg" in run]
        if lambda_regs:
            row["lambda_reg"] = lambda_regs[0]
        
        rows.append(row)
    
    df_summary = pd.DataFrame(rows)
    return df_summary

def export_metrics_to_csv(
    history: Dict[str, Dict[str, List[float]]],
    metrics: List[str],
    csv_path: str = None
) -> pd.DataFrame:
    """
    Converts the training history dictionary into a multi-index table format.
    """
    data = {}

    # Filter only valid algorithms (skip entries like "Total samples" or non-dicts)
    valid_algorithms = {
        algo: metrics_dict
        for algo, metrics_dict in history.items()
        if isinstance(metrics_dict, dict) and "global" in metrics_dict
    }

    # Determine the maximum number of rounds across selected metrics
    max_rounds = max(
        len(metrics_dict.get("global", {}).get(metric, []))
        for metrics_dict in valid_algorithms.values()
        for metric in metrics
    )

    rounds = list(range(1, max_rounds + 1))

    # Populate the data dictionary
    for alg, metrics_dict in valid_algorithms.items():
        for metric in metrics:
            values = metrics_dict.get("global", {}).get(metric, [])
            padded = values + [None] * (max_rounds - len(values))
            data[(alg, metric)] = padded

    # Create a multi-index DataFrame
    df = pd.DataFrame(data, index=rounds)
    df.index.name = "Round"
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["Algorithm", "Metric"])

    # Optionally save to CSV
    if csv_path:
        df.to_csv(csv_path)

    return df

# =============================================================================
# Diagnostics
# =============================================================================
def export_erank_diagnostics(
    history: dict,
    erank_base: Path,
    run_idx: int,
    store_plots: bool,
) -> None:
    """
    Save erank CSVs and generate erank plot for a single run.
    """
    if store_plots:
        sub_csv, drift_csv = erank_paths(erank_base, run_idx)
        erank_base.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history["FedDelta"]["global"]["subspace_alignment"]).to_csv(sub_csv,   index=False)
        pd.DataFrame(history["FedDelta"]["global"]["nodes_drift"]).to_csv(drift_csv, index=False)

        save_erank_analysis_csv(
            history    = history,
            run_idx    = run_idx,
            store_path = erank_base,
        )

    plot_erank_analysis(
        history        = history,
        run_idx        = run_idx,
        store_plots    = store_plots,
        store_path     = erank_base if store_plots else None,
    )

# =============================================================================
# Post-run summary
# =============================================================================
def save_summary_results(
    all_histories: list,
    base_dir: Path,
    config: ExperimentConfig,
    tag: Optional[str] = None,
) -> None:
    """
    Last-round summary, convergence plot, boxplots and statistical tests
    for ONE (lambda, sample_cfg, mode) combination.
    """
    if not all_histories:
        return
    df_summary = summarize_last_round(all_histories)
    logger.info("EXPERIMENT SUMMARY:\n%s\n", df_summary)

    plot_dir, csv_path = summary_results_paths(base_dir, tag)

    if config.store_plots:
        plot_dir.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_csv(csv_path, index=False)

    plot_metrics_summary(
        all_histories,
        store_plots         = config.store_plots,
        store_path          = plot_dir if config.store_plots else None,
        metrics             = config.metrics,
        only_fed_algorithms = True,
        orientation         = config.orientation_plots
    )

    # Fault log (crashes, aborted rounds, re-sent batches, rejected duplicates)
    rows = [{"run": i + 1, **f} for i, h in enumerate(all_histories) for f in h.get("faults", [])]
    if rows and config.store_plots:
        pd.DataFrame(rows).to_csv(plot_dir.parent / "fault_log.csv", index=False)

    if not config.store_plots or len(all_histories) < 2:
        return

    stat_metrics = list(config.metrics)
    # Fairness across nodes (each node weighs the same) when available
    for extra in ("r2_macro", "r2_worst"):
        if extra not in stat_metrics and any(extra in h.get("FedDelta", {}).get("final", {}) for h in all_histories):
            stat_metrics.append(extra)

    plot_boxplots(all_histories, config.metrics, plot_dir.parent / "boxplots",
                  reference=config.stats_reference)
    run_statistical_analysis(
        all_histories,
        out_dir            = plot_dir.parent / "statistics",
        metrics            = stat_metrics,
        reference          = config.stats_reference,
        equivalence_margin = {"r2": config.equivalence_margin_r2},
        alpha              = config.stats_alpha,
    )
