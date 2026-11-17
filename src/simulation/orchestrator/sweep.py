import logging

from typing import Dict
from pathlib import Path
from src.simulation.core.pipeline import run_experiment
from src.simulation.settings.plots.metrics import plot_cfg_comparison
from src.simulation.settings.plots.boxplots import plot_boxplots_by_cfg
from src.simulation.settings.algorithms import build_algorithms_config
from src.simulation.settings.metrics import save_summary_results
from src.simulation.settings.paths import _build_paths
from src.simulation.config import ExperimentConfig
from src.simulation.settings.data import load_raw_data

logger = logging.getLogger(__name__)

# =============================================================================
def build_robustness_conditions(failure_rate: float, failure_interval: int):
    """
    Condition of the experiment: with client dropout if failure_rate > 0, otherwise
    without failures. Returns the list of conditions and whether dropout is injected.
    """
    if failure_rate > 0.0:
        return [{
            "name": "Random Dropout",
            "node_failure_rate": failure_rate,
            "failure_interval": failure_interval,
        }], True

    return [{
        "name": "baseline",
        "node_failure_rate": 0.0,
        "failure_interval": 0,
    }], False

def run_sweep(
    lambda_regs: list,
    algs: dict,
    dataset_name: str,
    variable_target: str,
    base_dir: Path,
    raw_data,
    cfg: dict,
    metrics: list,
    nodes_heterogeneity,
    node_epochs: int,
    node_samples_list: list,
    all_histories: list,
) -> ExperimentConfig:
    """
    Iterate over all (lambda_reg, sample_cfg) combinations.
    """
    config = None

    for lambda_reg in lambda_regs:
        # Build algorithm config once per lambda — does not depend on sample_cfg
        algs_config    = build_algorithms_config(algs)
        erank_analysis = False
        if "FedDelta" in algs_config:
            erank_analysis = algs_config["FedDelta"]["params"]["erank_diagnostics"]

        # Same ridge lambda for every linear model so the comparison is fair.
        # lambda_scope="ridge": only the closed-form ridge models (Cent, FedDelta) take it;
        # the gradient-based methods keep their own lambda_reg (their local objective scales
        # the penalty with the mini-batch, so the same value is not the same model).
        if lambda_reg is not None:
            ridge_only = cfg.get("lambda_scope", "linear") == "ridge"
            for name, info in algs_config.items():
                if "lambda_reg" not in info["params"] or "MLP" in name:
                    continue
                if ridge_only and name not in ("Cent", "FedDelta"):
                    continue
                info["params"]["lambda_reg"] = lambda_reg

        # Precompute output paths for this lambda_reg
        paths = _build_paths(base_dir, lambda_reg)

        per_cfg_histories: Dict[tuple, list] = {}
        runs_by_cfg: Dict[str, list] = {}
        
        for sample_cfg in node_samples_list:

            min_s = sample_cfg["node_min_samples"]
            max_s = sample_cfg["node_max_samples"]

            min_train = sample_cfg["min_samples_train"]

            key = f"{min_s}:{max_s}:{min_train}"

            node_batch_size = min(32, max(16, (min_train // 2 // 8) * 8))

            local_cfg = {
                **cfg,
                "node_min_samples":  min_s,
                "node_max_samples":  max_s,
                "min_samples_train": min_train,
            }

            config = ExperimentConfig(
                dataset_name        = dataset_name,
                variable_target     = variable_target,
                node_epochs         = node_epochs,
                node_batch_size     = node_batch_size,
                nodes_heterogeneity = nodes_heterogeneity,
                algs_config         = algs_config,
                metrics             = metrics,
                **{k: v for k, v in local_cfg.items() if k in ExperimentConfig.__dataclass_fields__}
            )

            # Create output directories once per (lambda_reg, sample_cfg)
            if config.store_plots:
                paths["csv_dir"].mkdir(parents=True, exist_ok=True)

            cfg_histories = []
            for run_idx in range(config.num_runs):
                # Repeated hold-out: same node partition, new train/val/test split per run
                run_data = raw_data
                if config.resplit_per_run and run_idx > 0:
                    run_data = load_raw_data(dataset_name, variable_target, cfg,
                                             split_seed = config.seed + run_idx)
                history = run_experiment(
                    run_idx        = run_idx,
                    config         = config,
                    raw_data       = run_data,
                    lambda_reg     = lambda_reg,
                    dataset_name   = dataset_name,
                    paths          = paths,
                    erank_analysis = erank_analysis
                )
                
                if history is not None:
                    if len(node_samples_list) > 1 and key not in per_cfg_histories:
                        if 'FedDelta' in history:
                            per_cfg_histories[key] = history['FedDelta']
                    cfg_histories.append(history)
                    all_histories.append(history)

            runs_by_cfg[key] = cfg_histories

            # Summary, boxplots and tests for this (lambda, sample_cfg, mode) only
            mode_tag = "dropout" if config.node_failure_rate > 0 else "random"
            save_summary_results(
                cfg_histories, base_dir, config,
                tag = f"lambda{lambda_reg}/cfg_{min_s}-{max_s}-{min_train}/{mode_tag}",
            )

        # Grouped boxplots across sample configurations
        if len(runs_by_cfg) > 1 and config.store_plots and config.num_runs > 1:
            plot_boxplots_by_cfg(
                runs_by_cfg, config.metrics,
                base_dir / "cfg_comparison" / f"lambda{lambda_reg}" / "boxplots_by_cfg",
                reference = config.stats_reference,
            )

        if len(node_samples_list) > 1:
            cfg_comparison_dir = (
                base_dir / "cfg_comparison" / f"lambda{lambda_reg}" / "cfg_comparison"
            )
            
            plot_cfg_comparison(
                per_cfg_histories = per_cfg_histories,
                metrics           = config.metrics,
                store_plots       = config.store_plots,
                store_path        = cfg_comparison_dir if config.store_plots else None,
            )

    return config