from typing import List, Dict, Optional, Tuple
from src.federated.nodes.federated import FederatedNode
from src.algorithms.federated.base import FederatedAlgorithm
from src.federated.trainer.utils import attach_erank_metrics

import asyncio
import logging

logger = logging.getLogger(__name__)

async def run_node_training_round(
    node: FederatedNode, 
    alg: FederatedAlgorithm,
)-> bool:
    """ 
    Runs local training on a single node and sends the update to the leader.
    If the node exceeds `timeout` seconds it is treated as a straggler
    and its update is discarded for this round.
    """

    # The leader trains separately via aggregate_node_update
    if node.is_leader:
        return False
    
    try:
        return await node.attempt_node_training(alg)
    except FloatingPointError as exc:
        logger.warning(f"[Node {node.id}] update discarded: {exc}")
        return False
    except Exception as exc:
        logger.warning(f"[Node {node.id}] Local training failed: {exc}", exc_info=True)
        return False

def abort_round(alg: FederatedAlgorithm, node_roster: List[FederatedNode]) -> None:
    """Discard everything received this round; nodes keep their batches pending."""
    alg.discard_round()
    for node in node_roster:
        node.rollback(alg)


async def run_federated_round(
    name: str,
    alg: FederatedAlgorithm,
    ready_nodes: List[FederatedNode],
    node_roster: List[FederatedNode],
    leader: FederatedNode,
    eval_nodes: List[FederatedNode],
    verbose: bool,
    leader_crash: bool = False,
) -> Tuple[bool, Optional[Dict[str, float]], Optional[Dict], Optional[Dict]]:
    """
    Run one full federated round for a single algorithm.
    """

    if verbose: 
        logger.info("-" * 50)
        logger.info(f"[{name}] Starting federated round with leader {leader.id}")
        logger.info("-" * 50)

    if not ready_nodes:
        return False, {}, None
    
    # Step 1 — parallel local training (leader excluded)
    await asyncio.gather(
        *[run_node_training_round(node, alg) for node in ready_nodes],
        return_exceptions = True
    )

    # Leader crash after collecting the updates and before consolidating (C6):
    # the round is aborted atomically and every node keeps its batches.
    if leader_crash:
        abort_round(alg, node_roster)
        return False, {}, None

    # Step 2 — quorum check + aggregation + distribution
    if not leader.has_quorum(alg):
        if verbose:
            logger.warning(
                f"[{name}] Quorum not reached: "
                f"{(leader.participants)} update(s) received, {leader.quorum} required."
            )
        # Atomic round (C2): drop partial contributions; nodes re-send later (C3)
        abort_round(alg, node_roster)
        return False, {}, None

    # Round consolidated: ack to participants (their batches become committed) (C1/C3)
    node_map = {n.id: n for n in node_roster}
    for nid in leader.participants:
        if nid in node_map:
            node_map[nid].commit(alg)
    for node in node_roster:  # non-participants that trained but were not aggregated
        if node.id not in leader.participants:
            node.rollback(alg)
    alg.commit_round()

    # Step 3 — federated evaluation: every node that is up evaluates on its own test split
    #          and sends sufficient statistics; the leader rebuilds the exact metrics.
    await asyncio.gather(
        *[node.evaluate(alg, leader) for node in eval_nodes],
        return_exceptions=True
    )
    global_metrics = leader.compute_local_metrics(alg)
    global_metrics["exec_time"] = alg.round_elapsed

    # Time breakdown: slowest local training (parallel nodes) vs leader aggregation
    t_fit = max(alg.round_fit_times) if alg.round_fit_times else 0.0
    global_metrics["fit_time"] = t_fit
    global_metrics["agg_time"] = max(0.0, alg.round_elapsed - t_fit)

    # Communication (bytes, float64): uplink from participants (leader excluded),
    # downlink broadcast to every node that is up
    up, down = alg.comm_floats()
    n_up = max(0, len(leader.participants) - 1)
    global_metrics["comm_bytes"] = 8.0 * (n_up * up + len(eval_nodes) * down)
    global_metrics["n_participants"] = len(leader.participants)

    # Attach effective-rank diagnostics if the algorithm's collector has them
    attach_erank_metrics(alg, global_metrics)

    node_metrics = None

    return True, global_metrics, node_metrics