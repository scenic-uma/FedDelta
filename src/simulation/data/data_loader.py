import numpy as np
import logging

from src.simulation.data.partitioning import *
from src.simulation.data.partitioning import _SPLIT_STATE, set_split_seed, set_global_split, build_tabular_client_data
from src.simulation.data.readers import *

from typing import Tuple, Any

import sys
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Node partition per dataset configuration (read and partitioned once; re-split per run)
_PARTITION_CACHE = {}

def load_tabular_dataset( 
    max_samples: int,
    dataset_name: str,
    n_nodes: int,
    variable_target: int | str = None,
    max_features: int = None,
    feature_selection = False,
    scale_target: bool = True,
    partition: str = "natural",
    dirichlet_alpha: float = 0.5,
    split_seed: int = 42,
    global_split: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, Any, Tuple[np.ndarray, np.ndarray], None]:
    """
    Loads a real dataset using a given dataset loader function.
    If the number of features exceeds `max_features`, applies feature selection.
    """
    try:
        set_partition(partition, dirichlet_alpha)
        set_split_seed(split_seed)
        # global_split=True: train/val/test drawn over the whole pool, the same for every partition
        set_global_split(global_split)

        # Node partition is fixed (seed 42); only the per-node train/val/test split uses split_seed
        key = (dataset_name, str(variable_target), n_nodes, partition, dirichlet_alpha)
        if key in _PARTITION_CACHE:
            partition_data, selected_features_path = _PARTITION_CACHE[key]
            data, n_nodes = build_tabular_client_data(partition_data)
        else:
            data, n_nodes, selected_features_path = get_dataset(
                dataset_name = dataset_name,
                variable_target = variable_target,
                n_nodes = n_nodes
            )
            _PARTITION_CACHE[key] = (_SPLIT_STATE["last_partition"], selected_features_path)
        logger.info(f"[Dataset] Train/val/test split seed = {split_seed}")

        X_train_list, X_val_list, X_test_list = [], [], []
        y_train_list, y_val_list, y_test_list = [], [], []
        
        node_data = {}

        for node_name, node_splits in data.items():

            # Train/validation/test split of the node
            X_train_node, y_train_node = node_splits["train"]
            X_val_node, y_val_node     = node_splits["val"]
            X_test_node, y_test_node  = node_splits["test"]

            # Collect split data for global scaler and test set
            X_train_list.append(X_train_node)
            y_train_list.append(y_train_node)
            
            X_val_list.append(X_val_node)
            y_val_list.append(y_val_node)
            
            X_test_list.append(X_test_node)
            y_test_list.append(y_test_node)

            node_data[node_name] = {
                "train": (X_train_node, y_train_node),
                "val":   (X_val_node, y_val_node),
                "test":  (X_test_node, y_test_node),
            }
            
        X_train_global = np.vstack(X_train_list)
        y_train_global = np.concatenate(y_train_list)

        # Standardize the target with training stats only
        # (federated equivalent: aggregate n, sum y, sum y^2 once). R2 is unchanged.
        y_mean, y_std = 0.0, 1.0
        if scale_target:
            y_mean = float(y_train_global.mean())
            y_std = float(y_train_global.std()) or 1.0
            for node in node_data:
                for split in ("train", "val", "test"):
                    X, y = node_data[node][split]
                    node_data[node][split] = (X, (y - y_mean) / y_std)
            y_val_list = [(y - y_mean) / y_std for y in y_val_list]
            y_test_list = [(y - y_mean) / y_std for y in y_test_list]
            logger.info(f"[Dataset] Target standardized (mean={y_mean:.4g}, std={y_std:.4g})")

        if feature_selection:
            logger.info(
                f"[Dataset] Applying global feature selection "
                f"(max_features={max_features})"
            )

            X_train_global, n_features, selected_features = reduce_dimensionality(
                X=X_train_global,
                y=y_train_global,
                max_features=max_features,
                max_samples=max_samples,
                file_path=selected_features_path
            )

            for node in node_data:
                for split in ["train", "val", "test"]:
                    X, y = node_data[node][split]
                    node_data[node][split] = (X[:, selected_features], y)

            X_test_list = [X_t[:, selected_features] for X_t in X_test_list]
            X_val_list = [X_v[:, selected_features] for X_v in X_val_list]

        else:
            n_features = X_train_global.shape[1]


        # Full union of the nodes' splits (equivalent to federated evaluation; nothing discarded)
        X_val_global = np.vstack(X_val_list)
        y_val_global = np.concatenate(y_val_list)

        X_test_global = np.vstack(X_test_list)
        y_test_global = np.concatenate(y_test_list)
        

        val_global = (X_val_global, y_val_global)
        test_global = (X_test_global, y_test_global)
        
        return node_data, val_global, test_global, n_features, y_std

    except Exception as e:
        # Log error if dataset loading fails
        logger.error(f"Error while loading the dataset: {e}", exc_info=True)
        sys.exit(1)

def get_dataset(dataset_name, variable_target, n_nodes):
    """
    Unified interface to load multiple real-world datasets.
    """
    dataset_name = dataset_name.lower()

    if dataset_name == 'discharges': 
        return read_discharges(variable_target, n_nodes = n_nodes)
    elif dataset_name == 'insurance': 
        return read_insurance(
            variable_target, n_nodes = n_nodes
        )
    elif dataset_name == 'anxiety': 
        return read_anxiety(
            variable_target, n_nodes = n_nodes
        )
    elif dataset_name == 'housing': 
        return read_housing(
            variable_target, n_nodes = n_nodes
        )
    elif dataset_name == 'length_of_stay': 
        return read_lengthofstay(
            target = variable_target, n_nodes = n_nodes
        )
    elif dataset_name == 'european_survey_adult': 
        return read_european_survey_adult(
            target = variable_target, n_nodes = n_nodes
        )
    else:
        raise ValueError(f"[ERROR] Dataset '{dataset_name}' is not recognized.")