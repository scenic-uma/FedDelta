from src.simulation.settings.nodes import create_nodes

import logging

logger = logging.getLogger(__name__)

def prepare_tabular_task(node_data, val_global, test_global, n_features, y_std, config):
    """
    Distribute pre-loaded data into nodes.
    """
    # MSE is reported back in original target units
    config.target_scale = y_std

    X_test_global, y_test_global = test_global
    X_val_global, y_val_global = val_global

    nodes = create_nodes(
        quorum              = config.quorum,
        n_features          = n_features,
        node_data           = node_data,
        node_batch_size     = config.node_batch_size,
        nodes_heterogeneity = config.nodes_heterogeneity,
        node_epochs         = config.node_epochs,
        verbose             = config.verbose
    )

    logger.info(
        "Dataset: %s | Nodes: %d | Features: %d",
        config.dataset_name, len(nodes), n_features,
    )
    
    return nodes, n_features, X_test_global, y_test_global, X_val_global, y_val_global