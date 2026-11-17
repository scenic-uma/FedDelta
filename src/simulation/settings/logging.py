import logging
import sys

from src.simulation.config import RESULTS_DIR, ExperimentConfig

logger = logging.getLogger(__name__)

def setup_logging(log_path: str, level: int = logging.INFO) -> logging.Logger:
    """
    Configure logging for a single experiment.
    """

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove existing handlers to avoid log duplication
    # and to allow reconfiguration for multiple experiments
    if logger.hasHandlers():
        logger.handlers.clear()

    # Define common log format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    # File handler (writes logs to file)
    file_handler = logging.FileHandler(
        log_path,
        mode="w",           # Overwrite file for each experiment
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    # Console handler (prints logs to stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Attach handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def init_logging(dataset_name: str, variable_target: str) -> None:
    """
    Set up file logging once per dataset (idempotent for repeated calls).
    """
    log_path = RESULTS_DIR / dataset_name / f"target_{variable_target}" / "experiments.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_path))

def log_experiment_setup(
    config:         ExperimentConfig,
    lambda_reg:     float,
    nodes:          list,
    n_features:     int
) -> None:
    """
    Log a summary of the experiment configuration and dataset properties.
    """
    logger.info("=" * 50)
    logger.info(
        "EXPERIMENTAL SETUP:\n"
        " - Number of rounds: %d\n"
        " - Number of runs: %d\n"
        " - Max skipped rounds: %d\n"
        " - Batch size: %d\n"
        " - Node epochs: %d\n"
        " - Node failure rate: %s\n"
        " - Node failure interval: %d\n"
        " - Target: %s\n"
        " - Lambda: %s\n"
        " - Quorum: %d\n"
        " - Min samples: %d\n"
        " - Max samples: %d\n"
        " - Samples to train: %d\n"
        " - Nodes: %d\n"
        " - Features: %d\n"
        " - Train ratio: %.2f\n"
        " - Val ratio: %.2f",
        config.num_rounds, config.num_runs, config.max_skipped_rounds,
        config.node_batch_size, config.node_epochs, config.node_failure_rate,
        config.failure_interval, config.variable_target, lambda_reg, 
        config.quorum, config.node_min_samples, config.node_max_samples,
        config.min_samples_train, len(nodes), n_features, config.train_ratio, config.val_ratio,
    )