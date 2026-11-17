import random
import logging
import asyncio
from typing import List, Optional

from src.federated.nodes.federated import FederatedNode

logger = logging.getLogger(__name__)

def promote_to_leader(node: FederatedNode) -> FederatedNode:
    """
    Promote an existing node to leader role.
    """
    node.setup_leader()
    node.is_leader = True
    return node

def run_gossip_rounds(
    ready_nodes: List[FederatedNode],
    fanout: int = 3,
    rounds: int = 3,
    verbose: bool = False
) -> None:
    """
    Run gossip rounds among ready nodes.
    Each node exchanges its known_bids with random peers,
    converging toward a globally consistent view without a central coordinator.
    """

    # Each node starts knowing only its own bid
    for node in ready_nodes:
        node.known_bids = {node.id: node.compute_bid()}

    # Gossip rounds
    for r in range(1, rounds + 1):
        if verbose:
            logger.info("-" * 50)
            logger.info(f"[Gossip] Round {r}/{rounds}")
            logger.info("-" * 50)
        
        for node in ready_nodes:
            # Select up to `fanout` random peers (excluding self)
            peers = random.sample(
                [n for n in ready_nodes if n.id != node.id],
                k=min(fanout, len(ready_nodes) - 1)
            )
            for peer in peers:
                # Node pushes its known bids to the peer.
                # The peer merges them into its own local state (see receive_bids).
                node.send_bid(peer)

async def self_proclamation(
    ready_nodes: List[FederatedNode],
    verbose: bool = False
) -> Optional[FederatedNode]:
    '''
    Nodes self-evaluate leadership eligibility concurrently.
    First successful self-proclamation wins.
    '''
    logger.info("-" * 50)
    
    tasks = {
        asyncio.create_task(node.self_proclaim_leader()): node
        for node in ready_nodes
    }

    leader_node = None

    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            node = tasks.pop(task)

            try:
                did_proclaim = task.result()
            except Exception as e:
                logger.warning(f"[Gossip] Node {node.id} failed: {e}")
                continue

            if did_proclaim and leader_node is None:
                # First node to self-proclaim — promote immediately
                leader_node = promote_to_leader(node)

                if verbose:
                    logger.info("-" * 50)
                    logger.info(f"[Gossip] Node {leader_node.id} promoted to leader")

                # Cancel all remaining tasks — election is over
                for remaining in pending:
                    remaining.cancel()

                # Wait for cancellations to complete cleanly
                await asyncio.gather(*pending, return_exceptions=True)
                tasks.clear()
                break

    if not leader_node:
        # Fallback: gossip didn't converge, pick highest known bid
        logger.warning("[Gossip] No node self-proclaimed. Gossip-based leader election failed")
        return None

    # Conflict resolution: with partial gossip views several nodes may self-proclaim and
    # their notifications race. Only the winner's announcement is valid: every ready node
    # adopts it and any other self-proclaimed node steps down, so all updates of the
    # round reach the same leader (otherwise they would be aggregated but never committed).
    for node in ready_nodes:
        node.leader = leader_node
        node.leaderNotification = True
        node.is_leader = node is leader_node

    return leader_node