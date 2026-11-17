from src.federated.nodes.federated import FederatedNode
from typing import List, Tuple, Optional, Dict, Union

import pandas as pd
import random as _random
import logging

logger = logging.getLogger(__name__)

DataType = Union[
    Tuple[pd.DataFrame, pd.Series],           # (X, y) global tabular
    Dict[str, pd.DataFrame],                  # {client: df} tabular per client
    Dict[str, Dict[str, pd.DataFrame]]        # {client: {'train':df, ...}} time-series
]

def generate_node_capacities(
    node_ids: List[str],
    heterogeneity: str = "moderate",   # "low", "moderate", "high", "extreme"
    min_capacity: int = 50,
    max_capacity: int = 2000,
    random_seed: int = 42
) -> Dict[str, int]:
    """
    Generates realistic per-node capacities (maximum number of samples per node).
    
    Parameters:
        node_ids: List of node identifiers.
        heterogeneity: Level of difference between node sizes.
            - "low"      → Nodes have very similar capacities.
            - "moderate" → Balanced and realistic heterogeneity (recommended).
            - "high"     → High variation between nodes.
            - "extreme"  → A few very large nodes and many small ones.
        min_capacity: Minimum number of samples a node can have.
        max_capacity: Maximum number of samples a node can have.
        random_seed: Seed for reproducibility.
    
    Returns:
        Dictionary {node_id: capacity}
    """
    # Local RNG: do not reseed the global `random` module
    random = _random.Random(random_seed)
    n_nodes = len(node_ids)
    
    if n_nodes == 0:
        return {}

    # Generate capacities based on desired heterogeneity level
    if heterogeneity == "low":
        # Nodes have very similar sizes
        base = random.randint(min_capacity * 3, max_capacity // 2)
        capacities = [int(base * random.uniform(0.85, 1.15)) for _ in range(n_nodes)]
        
    elif heterogeneity == "moderate":
        # Recommended: most nodes medium-sized, some small and some large
        capacities = []
        for _ in range(n_nodes):
            r = random.random()
            if r < 0.15:
                cap = random.randint(min_capacity, min_capacity * 3)           # small nodes
            elif r < 0.85:
                cap = random.randint(min_capacity * 2, max_capacity // 2)      # medium nodes
            else:
                cap = random.randint(max_capacity // 2, max_capacity)          # large nodes
            capacities.append(cap)
            
    elif heterogeneity == "high":
        # Completely random within the full range
        capacities = [random.randint(min_capacity, max_capacity) for _ in range(n_nodes)]
        
    elif heterogeneity == "extreme":
        # Very few large nodes, majority are small
        capacities = []
        big_node_count = max(1, n_nodes // 5)
        for i in range(n_nodes):
            if i < big_node_count:
                cap = random.randint(max_capacity * 2 // 3, max_capacity)
            else:
                cap = random.randint(min_capacity, min_capacity * 4)
            capacities.append(cap)
    else:
        raise ValueError(f"Invalid heterogeneity '{heterogeneity}'. "
                        f"Valid options: low, moderate, high, extreme")

    # Shuffle to avoid order bias
    random.shuffle(capacities)
    
    # Create final dictionary
    node_capacities = {node_id: cap for node_id, cap in zip(node_ids, capacities)}

    # Logging summary
    caps = list(node_capacities.values())
    logger.info(f"[Capacities] Generated for {n_nodes} nodes | "
                f"min={min(caps)} | max={max(caps)} | mean={sum(caps)/len(caps):.0f} | "
                f"heterogeneity={heterogeneity}")

    return node_capacities

def create_nodes(
    quorum: int,
    node_batch_size: int,
    node_data: Optional[DataType] = None,
    n_features: Optional[int] = None,
    node_epochs: Optional[int] = None,
    
    nodes_heterogeneity: str = "moderate",
    verbose: bool = False,
    
) -> List[FederatedNode]:
    """
    Creates a list of federated nodes based on the input data type.
    """
    node_roster = []

    if node_data is None:
        raise ValueError("node_data cannot be None")
    
    common_params = {
        "quorum":     quorum,
        "n_features": n_features,
        "batch_size": node_batch_size,
        "epochs":     node_epochs,
        "verbose":    verbose
    }

    logger.info("[Dataset] Already partitioned per node")
    for node_name, node_data in node_data.items():
        node = FederatedNode(
            node_id = node_name,
            data = node_data,
            **common_params
        )
        node_roster.append(node)

    if nodes_heterogeneity != "None":
        node_capacities = generate_node_capacities(
            node_ids      = [str(n.id) for n in node_roster],
            heterogeneity = nodes_heterogeneity,
            min_capacity  = 1,
            max_capacity  = 100
        )

        for node in node_roster:
            node.capacity = node_capacities.get(str(node.id), 50)

        logger.info("\n".join(f" - Node {str(n.id)} → Capacity: {n.capacity:,}" for n in node_roster))

    # Inject peer list into each node — required for gossip and leader election
    for node in node_roster:
        node.peer_nodes = [n for n in node_roster if n.id != node.id]

    logger.info(f"[SYSTEM] Created {len(node_roster)} nodes")
    
    return node_roster
