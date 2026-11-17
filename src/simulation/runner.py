import logging
import os
import pickle

from typing import Dict
from src.simulation.orchestrator.sweep import run_sweep, build_robustness_conditions
from src.simulation.orchestrator.comparison import export_lambda_comparison
from .settings.display import show_banner
from .settings.data import load_raw_data
from .settings.logging import init_logging
from .config import load_global_config, RESULTS_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# Entry point
# =============================================================================
def main():
    # Load global configuration from YAML
    cfg = load_global_config()

    # Extract and remove top-level keys that are not passed to ExperimentConfig
    datasets            = cfg.pop("datasets_list")
    node_epochs         = cfg.pop("node_epochs")
    nodes_heterogeneity = cfg.pop("nodes_heterogeneity")
    algs                = cfg.pop("algorithms", {})
    metrics             = cfg.pop("metrics")
    node_samples_list   = cfg.pop("node_samples_list")

    # All histories, exported for the experiments/ scripts (FEDDELTA_HISTORY_OUT)
    collected = {}

    for dataset_name, dataset_cfg in datasets.items():
        variable_targets = dataset_cfg["variable_target"]
        if not isinstance(variable_targets, list):
            variable_targets = [variable_targets]

        for variable_target in variable_targets:
            init_logging(dataset_name, variable_target)
            base_dir = RESULTS_DIR / dataset_name / f"target_{variable_target}"
            show_banner(dataset_name, variable_target)

            # Load raw data once — reused across all lambda/run combinations
            raw_data = load_raw_data(dataset_name, variable_target, cfg)

            robustness_conditions, is_robustness = build_robustness_conditions(
                cfg.get("node_failure_rate", 0.0),
                cfg.get("failure_interval"),
            )

            histories_dropout: Dict[str, list] = {}
            lambda_regs   = algs.get("FedDelta", {}).get("lambda_reg", [None])

            for condition in robustness_conditions:

                num_runs_override = cfg.get("robustness_runs", 1) if is_robustness else cfg.get("num_runs")

                condition_cfg = {
                    **cfg,
                    "num_runs":          num_runs_override,
                    "node_failure_rate": condition["node_failure_rate"],
                    "failure_interval":  condition.get("failure_interval"),
                }
                
                cond_name      = condition["name"]

                logger.info("=" * 70)
                logger.info(" ROBUSTNESS CONDITION: %s", cond_name)
                logger.info("=" * 70)
                
                histories = []

                # Run all (lambda, sample_cfg, run) combinations of this condition
                config = run_sweep(
                    lambda_regs                  = lambda_regs,
                    algs                         = algs,
                    dataset_name                 = dataset_name,
                    variable_target              = variable_target,
                    base_dir                     = base_dir,
                    raw_data                     = raw_data,
                    cfg                          = condition_cfg,
                    metrics                      = metrics,
                    nodes_heterogeneity          = nodes_heterogeneity,
                    node_epochs                  = node_epochs,
                    node_samples_list            = node_samples_list,
                    all_histories                = histories,
                )

                # Lambda sweep comparison
                if len(lambda_regs) > 1:
                    export_lambda_comparison(
                        histories, 
                        base_dir, 
                        dataset_name, 
                        variable_target, 
                        config
                    )

                histories_dropout[cond_name] = histories
                collected.setdefault(f"{dataset_name}|{variable_target}", {})[cond_name] = histories

    out = os.environ.get("FEDDELTA_HISTORY_OUT")
    if out:
        with open(out, "wb") as f:
            pickle.dump(collected, f)
    return collected

if __name__ == "__main__":
    main()