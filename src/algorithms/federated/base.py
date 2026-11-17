# src/algorithms/federated/base.py
from ..base import BaseAlgorithm
from abc import abstractmethod
from typing import Dict, Tuple, Any
import numpy as np
import asyncio
import time
import logging

from src.federated.trainer.evaluation import pooled_metrics

logger = logging.getLogger(__name__)

class FederatedAlgorithm(BaseAlgorithm):
    """
    Base class for all federated algorithms.
    Adds async safety and aggregation logic.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lock = asyncio.Lock()
        self.round_total_samples    = 0
        self.weighted_sum           = None
        self.round_fit_times        = []
        self.round_elapsed          = 0.0
        self.round_node_stats       = []   # per-node evaluation statistics (current round)

        # Fault tolerance (exactly-once, atomic rounds)
        self.committed_ids          = set()  # (node_id, batch_id) already consolidated
        self.round_ids              = []     # (node_id, batch_id) received in the current round
        self.rejected_duplicates    = 0
        self.tau_weighted_sum       = None

    async def aggregate_update(self, update: Tuple[Any, int]):
        """
        Receives (local_model_weights_list, n_samples) and accumulates for Fed strategy.
        """
        # Acquire async lock to safely update shared variables
        async with self.lock:
            # Track each node's fit duration for round time calculation
            self.round_fit_times.append(update[4])
            start = time.perf_counter()

            self.round_total_samples, self.weighted_sum, self.tau_weighted_sum = self.aggregation_method.accumulate(
                update, 
                total_samples = self.round_total_samples, 
                weighted_sum = self.weighted_sum,
                tau_weighted_sum = self.tau_weighted_sum
            )

            self.round_elapsed += time.perf_counter() - start

    @abstractmethod
    def complete_aggregation_round(self) -> np.ndarray:
        """
        Finalize the global model after all client updates have been aggregated.
        Called once per communication round.
        Returns the new global model.
        """
        pass

    def n_params(self) -> int:
        w = self.global_model
        ws = w.get_weights() if hasattr(w, "get_weights") else [np.asarray(w)]
        return int(sum(np.size(x) for x in ws))

    def comm_floats(self):
        """(uplink, downlink) floats per client and round. Default: the model parameters."""
        p = self.n_params()
        return p, p

    def commit_round(self) -> None:
        """Round consolidated: its batches can never be counted again."""
        self.committed_ids.update(self.round_ids)
        self.round_ids = []

    def discard_round(self) -> None:
        """
        Atomic rounds (C2): a round that does not complete (no quorum, leader crash)
        leaves no trace. Partial contributions are dropped; nodes keep their batches
        and re-send them in a later round.
        """
        self.weighted_sum = None
        self.tau_weighted_sum = None
        self.round_total_samples = 0
        self.round_ids = []
        self.round_fit_times = []
        self.round_elapsed = 0.0

    def reset_elapsed_metrics(self) -> None:
        """
        Resets elapsed-time metrics for a completed round.
        """
        self.round_fit_times = []
        self.round_elapsed = 0.0

    async def aggregate_local_metrics(
        self,
        local_metrics: Dict[str, float],
        n_samples: int,
    ) -> None:
        """
        Leader side: store one node's sufficient statistics (n, sse, sum_y, sum_y2, r2_local).
        """
        async with self.lock:
            self.round_node_stats.append(dict(local_metrics))

    def compute_local_metrics(self) -> Dict[str, float]:
        """
        Exact pooled MSE/R² and fairness metrics over the evaluating nodes.
        Resets the accumulator for the next round.
        """
        metrics = pooled_metrics(self.round_node_stats)
        self.round_node_stats = []
        self.round_total_samples = 0
        return metrics