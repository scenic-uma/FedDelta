import logging

from src.simulation.config import ExperimentConfig
from src.simulation.data import prepare_tabular_task
from src.simulation.data.data_loader import load_tabular_dataset

logger = logging.getLogger(__name__)

# =============================================================================
def load_raw_data(
    dataset_name: str, 
    variable_target: str, 
    cfg: dict, 
    split_seed: int = None,
):
    """
    Load raw dataset once per (dataset, variable_target) combination.
    """
    
    return load_tabular_dataset(
        dataset_name      = dataset_name,
        variable_target   = variable_target,
        max_features      = cfg.get("max_features"),
        feature_selection = cfg.get("feature_selection", False),
        max_samples       = cfg.get("max_samples"),
        n_nodes           = cfg.get("num_nodes"),
        scale_target      = cfg.get("scale_target", True),
        partition         = cfg.get("partition", "natural"),
        dirichlet_alpha   = cfg.get("dirichlet_alpha", 0.5),
        global_split      = cfg.get("global_split", False),
        split_seed        = cfg.get("seed", 42) if split_seed is None else split_seed,
    )

# =============================================================================
def setup_task(raw_data, config: ExperimentConfig):
    """
    Prepare nodes and evaluation data for the current task type.
    """
        
    nodes, n_features, X_test_global, y_test_global, X_val_global, y_val_global = prepare_tabular_task(*raw_data, config)
    return nodes, n_features, X_val_global, y_val_global, X_test_global, y_test_global