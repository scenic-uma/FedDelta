from src.federated.nodes.node import BaseNode
from src.algorithms.federated.base import FederatedAlgorithm
from src.federated.nodes.leader import LeaderNode
from typing import Tuple, Dict, List

import asyncio

from src.federated.trainer.evaluation import node_stats
import numpy as np

class FederatedNode(BaseNode, LeaderNode):
    def __init__(
        self,
        node_id,
        quorum,
        peer_nodes: List['FederatedNode'] = [],
        data = None,
        variable_target = None,
        n_features = None,
        batch_size = None,
        epochs = None,
        verbose = False
    ):
        """
        Initialize a node in the federated linear regression system.
        """
        super().__init__(
            node_id         = node_id,
            n_features      = n_features,
            verbose         = verbose,
            data            = data,
            batch_size      = batch_size,
            epochs          = epochs,
            variable_target = variable_target
        )
        
        # Current leader node in the federated system (if any)
        self.leader = None 
        self.is_leader = False
        
        # Flag to indicate if the node has received a leader notification
        self.leaderNotification = False

        # Flag indicating whether the node has sent its local update in the current training round
        self.contribution_sent = False

        # Internal state for gossip-based leader election
        self.known_bids: Dict[str, float] = {}

        # Number of nodes involved in the global update
        self.quorum = quorum

        # known cluster members
        self.peer_nodes: List['FederatedNode'] = peer_nodes

    def send_bid(self, target: 'FederatedNode') -> None:
        """
        Send this node's known bids to a peer during gossip exchange.
        """
        if self.verbose:
            self.logger.info(f"[Node {self.id}] sending known bids to Node {target.id}")
        target.receive_bids(self.known_bids.copy())

    def receive_bids(self, incoming_bids: Dict[str, float]) -> None:
        """
        Receive and merge a peer's known bids into this node's local state.
        Incoming bids only add new entries — existing values are not overwritten
        since bids are computed once and are immutable.
        """
        for node_id, bid in incoming_bids.items():
            if node_id not in self.known_bids:
                self.known_bids[node_id] = bid
        if self.verbose:
            self.logger.info(
                f"[Node {self.id}] received bids, now knows {list(self.known_bids.keys())} nodes."
            )

    async def self_proclaim_leader(self) -> bool:
        """
        Check if this node sees itself as the winner in its local bid view.
        If so, broadcast a leader notification to all peers in parallel.
        Returns True if this node self-proclaimed, False otherwise.
        """
        if not self.known_bids:
            return False

        # A node must know at least one peer to be eligible for leadership
        if len(self.known_bids) < len(self.peer_nodes) / 3:
            if self.verbose:
                self.logger.info(
                    f"[Node {self.id}] cannot self-proclaim — "
                    f"only knows itself, no peers discovered yet."
                )
            return False

        # Check if this node is the highest bidder in its own local view
        local_winner_id = max(self.known_bids, key=self.known_bids.get)
        local_winner_bid = self.known_bids[local_winner_id]

        if local_winner_id != self.id:
            if self.verbose:
                self.logger.info(
                    f"[Node {self.id}] does not self-proclaim — "
                    f"local view points to Node {local_winner_id} "
                    f"(bid={local_winner_bid:.4f})"
                )
            return False

        if self.verbose:
            self.logger.info(
                f"[Node {self.id}] self-proclaims as leader "
                f"(bid={self.known_bids[self.id]:.4f}, "
                f"known_nodes={len(self.known_bids)})"
            )

        # Notify all peers in parallel
        await asyncio.gather(*[
            asyncio.to_thread(peer.receive_notification, self.id)
            for peer in self.peer_nodes if peer.id != self.id
        ])

        # ── The elected node must also set itself as its own leader ──
        self.leader = self
        self.is_leader = True

        return True

    def receive_notification(self, leader_id: 'FederatedNode') -> None:
        """
        Receives a notification from the leader node.
        """
        # Update the leader reference to the notifying leader node
        self.leader = next((n for n in self.peer_nodes if n.id == leader_id), None)

        # Log the reception of the notification if verbose mode is on
        if self.verbose:
            self.logger.info(f"[Node {self.id}] Received leadership notification from Node {self.leader.id}")

        # Mark that a leader notification has been received
        self.is_leader = False
        self.leaderNotification = True

    async def send_local_updates(self, local_update: Tuple[np.ndarray, int], alg: 'FederatedAlgorithm') -> None:
        """
        Sends the local updates to the leader node for aggregation.
        """
        if self.leaderNotification and self.ready_to_train:
            if self.verbose:
                self.logger.info(f"[Node {self.id}] is sending its local updates to leader {self.leader.id}")
    
            contribution = await self.leader.aggregate_node_update(
                alg, local_update, self.id, batch_ids=self.sent.get(alg, [])
            )

            if contribution:
                self.contribution_sent = True

    def incorporate_global_update(self) -> None:
        """
        Incorporate the global model update received from the leader into the local algorithm.
        """
        if self.verbose:
            self.logger.info(f"[Node {self.id}] updates its model using the global update received from the leader {self.leader.id}")

    async def evaluate(
        self,
        alg: FederatedAlgorithm,
        leader: 'FederatedNode' = None,
    ) -> Dict:
        """
        Evaluates the global model on each node's local test set.
        """
        
        X_test, y_test = self.data["test"]

        # Local evaluation -> only sufficient statistics leave the node (see trainer/evaluation.py)
        node_metrics = node_stats(alg, X_test, y_test)
        n_samples = X_test.shape[0]

        # Send to the leader of this round (non-ready nodes may hold a stale/None leader)
        if leader is not None:
            self.leader = leader
        await self.send_local_metrics(alg, node_metrics, n_samples)
    
    async def send_local_metrics(self, alg: 'FederatedAlgorithm', local_metrics: Dict, n_samples: int) -> None:
        """
        Sends the local metrics to the leader node for aggregation.
        """
        if self.verbose:
            self.logger.info(f"[Node {self.id}] sends its local metrics to leader {self.leader.id}")
        await self.leader.aggregate_local_metrics(alg, local_metrics, n_samples, self.id)
