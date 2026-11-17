from typing import List, Optional, Dict
from src.federated.nodes.federated import FederatedNode

import logging
import numpy as np

logger = logging.getLogger(__name__)

def handle_node_lifecycle(
    node_roster:       List[FederatedNode],
    absolute_round:   int,
    node_failure_rate: float,
    failure_interval:  Optional[int],
    verbose:          bool = False,
) -> None:
    """
    On each designated round, randomly chooses ONE action:
      - Dropout:    deactivate a random subset of active nodes
      - Reconnect:  reactivate a random subset of interrupted nodes
    If the chosen action cannot be executed, it is a no-op.
    A minimum of 2 active nodes is always guaranteed.
    No-ops when failure_rate is 0 or the round is not a failure_interval multiple.

    If `schedule` is provided, uses pre-generated deterministic events
    instead of sampling randomly. The schedule path bypasses all stochastic
    guards — the schedule already encodes when and which nodes to affect.
    """

    # ── Guard: stochastic path only ──────────────────────────────────────────
    if failure_interval is None or failure_interval <= 0 or node_failure_rate <= 0.0:
        return

    if (absolute_round) % failure_interval != 0:
        if verbose: 
            logger.debug(f"[LIFECYCLE] Round {absolute_round} — not a lifecycle round "
                f"(interval={failure_interval}), skipping.")
        return

    # ── Inventory ────────────────────────────────────────────────────────────
    active_nodes      = [n for n in node_roster if n.active]
    interrupted_nodes = [n for n in node_roster if n.interrupted and not n.active]

    if verbose: 
        logger.info(f"[LIFECYCLE | ROUND {absolute_round}] "
            f"Active: {len(active_nodes)} | Interrupted: {len(interrupted_nodes)} | "
             f"Total: {len(node_roster)}")

    # ── Random action (no-op if not executable) ───────────────────────────────
    action = np.random.choice(["dropout", "reconnect"])
    
    if verbose: 
        logger.info(f"[LIFECYCLE | ROUND {absolute_round}] Chosen action: '{action}'")
        logger.info("-" * 50)

    # ── Execute action ────────────────────────────────────────────────────────
    if action == "dropout":
        max_droppable = max(0, len(active_nodes) - 2)
        if max_droppable == 0:
            if verbose: 
                logger.info(f"[LIFECYCLE | ROUND {absolute_round}] DROPOUT — "
                    f"no droppable nodes (min 2 active enforced), skipping."
                )
        else:
            num_dropped = min(max(1, round(len(active_nodes) * node_failure_rate)), max_droppable)

            if verbose: 
                logger.info(f"[LIFECYCLE | ROUND {absolute_round}] DROPOUT — "
                    f"rate={node_failure_rate} | "
                    f"eligible={len(active_nodes)} | "
                    f"max_droppable={max_droppable} | "
                    f"will_drop={num_dropped}"
                )

            dropped     = np.random.choice(active_nodes, size=num_dropped, replace=False)
            dropped_ids = ", ".join(str(n.id) for n in dropped)
            for node in dropped:
                node.interrupt()
                if verbose: 
                    logger.info(f"[LIFECYCLE | ROUND {absolute_round}] "
                            f"Node '{node.id}' → interrupted")
            
            if verbose: 
                logger.info(f"[LIFECYCLE | ROUND {absolute_round + 1}] "
                        f"Dropped {num_dropped} node(s): {dropped_ids}")

    elif action == "reconnect":
        if not interrupted_nodes:
            if verbose: 
                logger.info(f"[LIFECYCLE | ROUND {absolute_round}] RECONNECT — "
                        f"no interrupted nodes, skipping.")
        else:
            num_reconnected = min(max(1, round(len(interrupted_nodes) * node_failure_rate)),
                                  len(interrupted_nodes))

            if verbose:
                logger.info(f"[LIFECYCLE | ROUND {absolute_round}] RECONNECT — "
                            f"rate={node_failure_rate} | "
                            f"interrupted={len(interrupted_nodes)} | "
                            f"will_reconnect={num_reconnected}")

            reconnected     = np.random.choice(interrupted_nodes, size=num_reconnected, replace=False)
            reconnected_ids = ", ".join(str(n.id) for n in reconnected)
            for node in reconnected:
                node.reconnect()
                if verbose:
                    logger.info(f"[LIFECYCLE | ROUND {absolute_round}] "
                                f"Node '{node.id}' → reconnected")
            if verbose:
                logger.info(f"[LIFECYCLE | ROUND {absolute_round}] "
                            f"Reconnected {num_reconnected} node(s): {reconnected_ids}")