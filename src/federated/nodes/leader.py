import numpy as np
from typing import Dict, Tuple
import asyncio
import inspect

from src.algorithms.federated.base import FederatedAlgorithm

class LeaderNode:
    def setup_leader(self):
        """
        Initialize a leader node in the federated linear regression system.
        This node extends FederatedNode and acts as the leader coordinating aggregation.
        """
        # Number of participant nodes contributing updates
        self.participants = []
        # Create an asyncio lock to ensure mutual exclusion when accessing shared resources
        self.lock = asyncio.Lock()

    def reset_round(self) -> None:
        """Reset participant tracking for the next round."""
        self.participants = []

    def has_quorum(self, alg: FederatedAlgorithm) -> bool:
        """
        Check quorum and trigger aggregation and distribution if met.
        """
        if len(self.participants) < self.quorum:
            return False
        
        global_update = self.process_global_update(alg)
        self.distribute_global_update(alg, global_update)
        
        return True

    async def aggregate_node_update(
        self, 
        alg: FederatedAlgorithm, 
        node_update: Tuple[np.ndarray, int], 
        node_id: int,
        batch_ids: list = None,
    ) -> bool:
        async with self.lock:
            # Exactly-once (C1): reject a second update from the same node in this round
            # or any batch the leader has already committed in a previous round.
            batch_ids = list(batch_ids or [])
            if node_id in self.participants or any((node_id, b) in alg.committed_ids for b in batch_ids):
                alg.rejected_duplicates += 1
                self.logger.warning(f"[Leader {self.id}] duplicate contribution from Node {node_id} rejected.")
                return False

            # If no participants yet, leader trains locally first (on its own pending batches)
            if not self.participants:
                try:
                    local_update = self.train_local(alg)

                except Exception as e:
                    self.logger.error(f"[Leader {self.id}] local training failed: {e}")
                    return False
                
                try:
                    if local_update is not None:
                        if "features" in inspect.signature(alg.aggregate_update).parameters:
                            await alg.aggregate_update(local_update, self.features)
                        else:
                            await alg.aggregate_update(local_update)
                        self.participants.append(self.id)
                        alg.round_ids.extend((self.id, b) for b in self.sent.get(alg, []))
                    if self.verbose:
                        self.logger.info(
                            f"[Leader {self.id}] aggregated its local update"
                        )
                
                except Exception as e:
                    if self.verbose:
                        self.logger.error(
                            f"[Leader {self.id}] aggregating local contribution failed: {e}"
                        )
                    return False

            # Aggregate node update
            try:
                if "features" in inspect.signature(alg.aggregate_update).parameters:
                    await alg.aggregate_update(node_update, self.features)
                else:
                    await alg.aggregate_update(node_update)
                
                self.participants.append(node_id)
                alg.round_ids.extend((node_id, b) for b in batch_ids)
            except Exception as e:
                if self.verbose:
                    self.logger.error(
                        f"[Leader {self.id}] failed to aggregate update "
                        f"from Node {node_id}: {e}"
                    )
                return False

            if self.verbose:
                self.logger.info(f"[Leader {self.id}] aggregated update from Node {node_id}")

            return True
    
    def process_global_update(self, alg: FederatedAlgorithm) -> Tuple[np.ndarray, int]:
        """
        Processes the aggregated updates from all participant nodes to compute the global update.
        """
        # Log the start of global aggregation if verbose mode is enabled
        if self.verbose:
            self.logger.info("-" * 50)
            self.logger.info(f"[Leader {self.id}] Aggregating contributions from {len(self.participants)} nodes.")
            self.logger.info("-" * 50)

        # Compute the global update from aggregated local updates
        global_update = alg.complete_aggregation_round()

        # Return the computed global update
        return global_update
    
    def distribute_global_update(self, alg: FederatedAlgorithm, global_update: Tuple[np.ndarray, int]) -> None:
        """
        Distributes the updated global model parameters from the leader to all participant nodes.
        """
        # Log the start of the global model distribution
        if self.verbose:
            self.logger.info("-" * 50)
            self.logger.info(f"[Leader {self.id}] Distributing global model update to all participating nodes.")
            self.logger.info("-" * 50)

        # Send the global update to all nodes except the leader itself
        for node in self.peer_nodes:
            if self.id != node.id:
                if self.verbose:
                    self.logger.info(f"[Leader {self.id}] sends global update to Node {node.id}.")
                node.incorporate_global_update()

    async def aggregate_local_metrics(
        self,
        alg: FederatedAlgorithm,
        local_metrics: Dict[str, float],
        n_samples: int,
        node_id: str,
    ) -> None:
        
        """
        Incremental weighted aggregation of metrics per training round.
        """
        await alg.aggregate_local_metrics(local_metrics, n_samples)

        if self.verbose:
            self.logger.info(
                f"[Leader {self.id}] aggregated metrics from Node {node_id}."
            )
    
    def compute_local_metrics(self, alg: FederatedAlgorithm) -> Dict[str, float]:
        """
        Computes final weighted average for this round and resets accumulators.
        """

        return alg.compute_local_metrics()


        