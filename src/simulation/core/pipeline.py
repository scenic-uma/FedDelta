import asyncio
import logging
import random
import numpy as np

from src.simulation.settings.data import setup_task
from src.simulation.settings.metrics import export_erank_diagnostics, export_metrics_to_csv
from src.simulation.settings.logging import log_experiment_setup
from src.simulation.config import ExperimentConfig
from .setup import setup_and_train

logger = logging.getLogger(__name__)

# =============================================================================
# Single run
# =============================================================================
def run_experiment(
    run_idx:        int,
    config:         ExperimentConfig,
    raw_data,
    lambda_reg:     float,
    dataset_name:   str,
    paths:          dict,
    erank_analysis: bool,
) -> dict:
    """
    Execute one full federated training run and return its history.
    """
    logger.info("=" * 70)
    logger.info(" TRAINING RUN %d of %d", run_idx + 1, config.num_runs)

    nodes, n_features, X_val_global, y_val_global, X_test_global, y_test_global = setup_task(
        raw_data, config
    )

    # Seed after node creation: runs are reproducible and paired across algorithms
    run_seed = config.seed + run_idx
    random.seed(run_seed)
    np.random.seed(run_seed)

    log_experiment_setup(
        config            = config,
        lambda_reg        = lambda_reg,
        nodes             = nodes,
        n_features        = n_features
    )

    try:
        history = asyncio.run(
            setup_and_train(
                config           = config,
                run_idx          = run_idx,
                lambda_reg       = lambda_reg,
                node_roster      = nodes,
                n_features       = n_features,
                X_test_global    = X_test_global,
                y_test_global    = y_test_global,
                X_val_global     = X_val_global,
                y_val_global     = y_val_global,
                dataset_name     = dataset_name
            )
        )

        dropout_tag = "dropout_" if config.failure_interval > 0 else "random_"

        # Export per-round metrics to CSV for later analysis
        if config.store_plots:
            export_metrics_to_csv(
                history  = history,
                metrics  = config.metrics,
                csv_path = str(paths["csv_dir"] / f"{dropout_tag}run{run_idx + 1}_rounds.csv"),
            )

        # Attach metadata to history for downstream aggregation
        history["lambda_reg"] = lambda_reg
        history["n_features"] = n_features
        history["n_nodes"] = len(nodes)
        history["run_seed"] = run_seed

        # Save erank diagnostics if enabled
        if erank_analysis:
            export_erank_diagnostics(
                history, 
                paths["erank_base"], 
                run_idx, 
                config.store_plots
            )

        return history

    except Exception as e:
        logger.error(f"Run {run_idx + 1} failed: {e}", exc_info=True)
        return None

    finally:
        logger.info("=" * 70)
        logger.info("TRAINING RUN %d FINISHED.", run_idx + 1)