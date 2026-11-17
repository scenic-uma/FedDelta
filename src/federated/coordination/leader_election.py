import logging
from typing import List, Optional

from src.federated.nodes.federated import FederatedNode
from src.federated.nodes.leader import LeaderNode
from src.federated.coordination.gossip_protocol import run_gossip_rounds, self_proclamation

logger = logging.getLogger(__name__)

async def choose_leader(
    ready_nodes: List[FederatedNode],
    fanout: int = 3,
    rounds: int = 3,
    verbose: bool = False
) -> FederatedNode:
    """
    Perform a gossip-based leader election among ready nodes.

    Phase 1 — Gossip rounds: nodes exchange known bids with random peers
               until views converge.
    Phase 2 — Self-proclamation: each node independently checks if it won;
               the winner broadcasts notifications to all others in parallel.
    """
    if not ready_nodes:
        logger.warning("No ready nodes available for gossip election.")
        return None

    if verbose:
        logger.info("-" * 50)
        logger.info(f"[Gossip] Starting election among {len(ready_nodes)} ready nodes.")
        logger.info("-" * 50)

    # Phase 1: propagate bids across the network
    run_gossip_rounds(
        ready_nodes = ready_nodes, 
        fanout      = fanout, 
        rounds      = rounds, 
        verbose     = verbose
    )

    # Phase 2: each node decides and the winner notifies in parallel
    leader_node = await self_proclamation(ready_nodes = ready_nodes, verbose = verbose)

    return leader_node

def demote_leader(node: Optional[LeaderNode]) -> None:
    """
    Demotes a leader node back to a normal federated node state.
    """
    if node is None:
        return

    node.is_leader = False
    node.reset_round()
