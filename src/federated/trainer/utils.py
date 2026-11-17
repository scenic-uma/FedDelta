from typing import Dict, List, Optional, Tuple
from src.federated.nodes.federated import FederatedNode
from src.algorithms.federated.base import FederatedAlgorithm

import logging
import numpy as np

logger = logging.getLogger(__name__)

def log_node_status(nodes: List[FederatedNode], effective_round: int) -> None:
    """Logs the active and ready-to-train flags for every node in the cluster."""
    for node in nodes:
        logger.info(
            f"[Round {effective_round + 1} | Node {node.id}] "
            f"Active: {node.active} | Ready to train: {node.ready_to_train} | Interrupted: {node.interrupted}"
        )

def collect_diagnostics(
    federated_algs: List[Tuple[str, FederatedAlgorithm]],
    history: Dict,
) -> None:
    """
    Collects post-training diagnostics from algorithms that support them
    and stores the results in the history dict under each algorithm's global key.
 
    Supported diagnostics:
        - analyze_subspace_alignment: tracks how the update subspace evolves across rounds.
        - analyze_node_drift:         measures how far each node's local matrix drifts
                                      from the global matrix over time.
    """
    for name, alg in federated_algs:
        if hasattr(alg, "analyze_subspace_alignment"):
            history[name]["global"]["subspace_alignment"] = alg.analyze_subspace_alignment()
        if hasattr(alg, "analyze_node_drift"):
            history[name]["global"]["nodes_drift"] = alg.analyze_node_drift()

def log_round_header(round_idx: int, num_rounds: int):
    logger.info("-" * 50)
    logger.info(f" ROUND {round_idx + 1}/{num_rounds} ")
    logger.info("-" * 50)

def log_metrics(
    history: Dict[str, Dict[str, list]], 
    effective_round: int,
    absolute_round: int,
    num_participants: int
):
    """
    Log the metrics for all algorithms in the current round.
    """
    logger.info("=" * 50)
    logger.info(f"Metrics for round: Effective/Absolute: {effective_round}/{absolute_round} round | {history['Total samples'][-1]} samples | Num nodes: {num_participants}")
    logger.info("-" * 30)
    
    for name, alg_metrics in history.items():
        if name in {"Total samples", "node_status", "effective_rounds", "faults", "integrity"}:
            continue

        parts = [f"[{name}]"]

        for key, values in alg_metrics["global"].items():

            if not values:
                continue

            val = values[-1]

            if key == "exec_time":
                parts.append(f"Time: {val:.5f}s")
            elif key in ("fit_time", "agg_time", "n_participants", "n_eval_nodes", "r2_median"):
                continue
            elif key == "comm_bytes":
                parts.append(f"Comm: {val / 1e6:.2f} MB")
            elif key == "w_rel_error":
                parts.append(f"W-err: {val:.2e}")
            elif key == "r2":
                parts.append(f"R²: {val:.3f}")
            elif key == "erank_round":
                parts.append(f"Erank(update): {val:.2f}")
            elif key == "erank_global":
                parts.append(f"Erank(global): {val:.2f}")
            else:
                parts.append(f"{key.upper()}: {val:.3f}")
        
        logger.info(" | ".join(parts))

def update_history(
        history: dict, 
        algorithm_name: str, 
        global_metrics: dict,
        node_metrics: dict = None,
        mse_scale: float = 1.0
):
    """
    Appends metrics to the history dictionary for a given algorithm.

    Args:
        history (dict): Historical record of metrics for all algorithms.
        algorithm_name (str): Name of the algorithm to update.
        metrics (dict): Dictionary of metrics to append.
    """
    
    # Report MSE in original target units (target may be standardized)
    def _scale(k, v):
        return v * mse_scale if (k == "mse" and v is not None) else v

    for k, v in global_metrics.items():
        v = _scale(k, v)
        # Append the metric, defaulting to NaN if missing
        if k not in history[algorithm_name]["global"]:
            history[algorithm_name]["global"][k] = []

        history[algorithm_name]["global"][k].append(v if v is not None else float("nan"))

    # Update node metrics (list of values, appended as a list)
    if node_metrics is not None:
        for k, v in node_metrics.items():
            v = _scale(k, v)
            if k not in history[algorithm_name]["node"]:
                history[algorithm_name]["node"][k] = []
            history[algorithm_name]["node"][k].append(v if v is not None else [])

def attach_erank_metrics(alg: FederatedAlgorithm, global_metrics: Dict) -> None:
    """
    Reads effective-rank diagnostics from the algorithm's DiagnosticsCollector
    (if present) and injects them into the global metrics dict.

    """
    diagnostics = getattr(alg, "diagnostics", None)
    if diagnostics is None:
        return
 
    # last_erank_round and last_erank_global are computed lazily on access
    erank_round = diagnostics.last_erank_round
    erank_global = diagnostics.last_erank_global
 
    if erank_round is not None:
        global_metrics["erank_round"] = erank_round
    if erank_global is not None:
        global_metrics["erank_global"] = erank_global

def split_algorithms(
    algorithms: Dict[str, FederatedAlgorithm],
) -> Tuple[List, List]:
    """
    Splits the algorithms dict into two lists based on naming convention.
    Algorithms whose name starts with 'Cent' are treated as centralized baselines;
    all others are federated.
    """
    federated = [(n, a) for n, a in algorithms.items() if n.startswith("Fed")]
    centralized = [(n, a) for n, a in algorithms.items() if not n.startswith("Fed")]
    return federated, centralized

def init_history(
    algorithms: Dict[str, FederatedAlgorithm],
    metrics: List[str],
) -> Dict:
    """
    Builds the history dict skeleton: one entry per algorithm with empty lists
    for each metric. Federated algorithms also get per-node metric tracking;
    centralized ones only store global metrics.
    exec_time is excluded from node-level tracking (not meaningful per node).
    """
    history: Dict = {"Total samples": []}
    for name, _ in algorithms.items():
        history[name] = {"global": {k: [] for k in metrics}}
        # Node-level metrics only apply to federated algorithms
        if name.startswith("Fed"):
            history[name]["node"] = {k: [] for k in metrics if k != "exec_time"}
    return history

def check_participant_mismatch(round_participants: Dict[str, List]) -> None:
    """
    Checks that all federated algorithms used the same set of participants.
    Different participant sets can cause inconsistencies in the global model
    and should be investigated.
    """
    all_lists = list(round_participants.values())
    # Only meaningful when more than one algorithm ran this round
    if len(all_lists) > 1 and not all(lst == all_lists[0] for lst in all_lists[1:]):
        logger.warning("Federated algorithms used different participant sets this round.")
    return all_lists[0] if all_lists else []

def determine_ready_nodes(
        nodes: List, 
        min_samples: int, 
        max_samples: int, 
        min_samples_train: int,
        verbose: bool
) -> List:
    """
    Returns the list of nodes that are ready to train 
    """
    ready_nodes = []  # Initialize list to store ready nodes
    
    if verbose: 
        logger.info("-" * 50)
        logger.info("Getting ready nodes for training.")
        logger.info("-" * 50)
    
    for node in nodes:

        if not getattr(node, "active", True) or node.interrupted:
            continue
        
        try:
            stream = getattr(node, "stream_batch", None)
            if stream:
                # stream mode: about len(train)/num_rounds samples per round (± jitter);
                # jitter 0 -> exactly `stream`, so all training data is delivered in num_rounds
                jit = getattr(node, "stream_jitter", 0.2)
                sizes = getattr(node, "stream_sizes", None)
                if jit <= 0 and sizes is not None:
                    # deterministic schedule; an empty list -> nothing left (node deactivates)
                    n_samples = sizes.pop(0) if sizes else 0
                    if n_samples == 0 and sizes:
                        continue          # waits for its first scheduled round (stays active)
                elif jit > 0:
                    n_samples = np.random.randint(max(1, int((1 - jit) * stream)),
                                                  int((1 + jit) * stream) + 2)
                else:
                    n_samples = stream
            else:
                n_samples = np.random.randint(min_samples, max_samples + 1)
            # Ensure the node collects its local dataset
            node.collect_local_data(n_samples)  
            # Evaluate if it meets the minimum data requirement
            node.is_ready_to_train(min_samples_train)  
            # If the node is ready to train
            if node.ready_to_train:  
                ready_nodes.append(node)
        except RuntimeError as e:
            if str(e) == "No more data available":
                node.active = False
                continue
            else:
                raise
    return ready_nodes