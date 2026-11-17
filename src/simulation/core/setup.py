import logging

from typing import Dict
from src.simulation.settings.algorithms import get_algorithm_instance
from src.simulation.config import  ExperimentConfig
from src.federated.trainer.core import run_training
from src.simulation.core.tuning import tune_hyperparameters

logger = logging.getLogger(__name__)

async def setup_and_train(
    config:       ExperimentConfig,
    run_idx:      int,
    lambda_reg:   float,
    node_roster:  list,
    n_features:   int,
    dataset_name: str,
    X_test_global,
    y_test_global,
    X_val_global  = None,
    y_val_global  = None,
) -> Dict:
    """
    Run a single federated experiment
    """
    # --- Tune *_grid parameters on the validation set (cached across runs) ---
    tuned = await tune_hyperparameters(
        config, node_roster, n_features, X_val_global, y_val_global
    )
    algs_config = {
        name: {"class": info["class"], "params": {**info["params"], **tuned.get(name, {})}}
        for name, info in config.algs_config.items()
    }

    # --- Algorithm instantiation (tuned values override YAML) ---
    algs_instances = get_algorithm_instance(
        algs_config, 
        n_features, 
        config.verbose,
        n_clients = len(node_roster),
    )

    # --- Federated training ---
    history = await run_training(
        algorithms         = algs_instances, 
        node_roster        = node_roster,
        num_rounds         = config.num_rounds,
        node_min_samples   = config.node_min_samples,
        node_max_samples   = config.node_max_samples,
        min_samples_train  = config.min_samples_train,
        gossip_fanout      = config.gossip_fanout,
        gossip_rounds      = config.gossip_rounds,
        X_test_global      = X_test_global, 
        y_test_global      = y_test_global,
        X_val_global       = X_val_global, 
        y_val_global       = y_val_global,
        dataset_name       = dataset_name,
        node_failure_rate  = config.node_failure_rate,
        failure_interval   = config.failure_interval,
        metrics            = config.metrics,
        max_skipped_rounds = config.max_skipped_rounds,
        target_scale       = config.target_scale,
        leader_failure_rate= config.leader_failure_rate,
        leader_downtime    = config.leader_downtime,
        samples_mode       = config.samples_mode,
        stream_jitter      = config.stream_jitter,
        stream_min_batch   = config.stream_min_batch,
        election           = config.election,
        reference          = config.stats_reference,
        verbose            = config.verbose
    )

    return history 