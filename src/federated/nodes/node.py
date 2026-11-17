import numpy as np
import logging
import random
import inspect
import math

from src.algorithms.federated.base import FederatedAlgorithm

from src.federated.nodes.utils import compute_theoretical_flops, node_theoretical_time

"""
Definition of a generic federated node.
Each node stores its local data (X, y) and a unique identifier.
Includes a buffer in case there are not enough samples to train during a given round.
Does not contain training logic — that resides in the FederatedAlgorithm class.
"""

class BaseNode:
    """
    Generic federated node that solely manages local data.

    Attributes:
        id (Optional[int]): Unique identifier of the node.
        X (np.ndarray): Local feature matrix.
        y (np.ndarray): Local label vector.

        global_counter (int): Counts global update rounds.
        buffer_X (List[np.ndarray]): Buffer storing incoming mini-batches of features.
        buffer_y (List[np.ndarray]): Buffer storing incoming mini-batches of labels.
        dim (int): Dimensionality of the feature space (excluding intercept).
        global_model (np.ndarray): Current global model parameters including intercept.
        logger (logging.Logger): Logger instance for this class.
        verbose (bool): Flag to enable verbose logging.
        ready_to_train (bool): Indicates whether the node has enough data to start training.
        min_samples (int): Minimum sample size to trigger training.
        max_samples (int): Maximum sample size for data batches.
    """

    def __init__(
        self,
        node_id: int | None = None,
        # ── Data sources
        data: np.ndarray = None, 
        # ── Configuration
        n_features: int = None,
        batch_size: int = None,
        epochs: int = 1,
        variable_target: int = None,
        verbose: bool = False
    ):
        """
        Initializes a federated learning node.
        There are two main data modes: Time-series (sliding window)and Tabular
        """
        self.id = node_id
        self.logger = logging.getLogger(self.__class__.__name__)
        self.verbose = verbose

        # Training control
        self.active = True
        self.ready_to_train = False

        # Hyperparameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_features = n_features
        self.variable_target = variable_target
        self.features = None
        self.data = data

        # Model & buffers
        self.global_model = np.zeros(n_features + 1)
        self.buffer_X = []
        self.buffer_y = []
        self.current_index = 0

        # Batch ledger (fault tolerance): every collected batch has an id and stays
        # on the node until the leader confirms (commits) it, independently per algorithm.
        self.batches = {}        # batch_id -> (X, y)
        self.batch_seq = 0
        self.committed = {}      # alg -> set of committed batch ids
        self.sent = {}           # alg -> batch ids sent in the current round
        self.last_committed = {} # alg -> batch ids committed in the last successful round
        self.recover_at = None   # round at which a crashed node comes back (crash-recovery)
        self.ever_sent = {}      # alg -> batch ids sent at least once
        self.resent_last = {}    # alg -> how many batches of the last send were re-sends
        self.double_commits = {} # alg -> batches committed again after being committed (must be 0)

        # Data handling
        self.capacity = None 
        self.interrupted = False
        
        self.X_full, self.y_full = data['train']
    
    def __repr__(self):
        return f"Node(id={self.id}, Capacity={self.capacity})"
    
    def deactivate(self) -> None:
        """
        Deactivates the node when no more data is available.
        Clears local data and buffers to free memory.
        """
        self.active = False
        self.ready_to_train = False
        self.X_local = None
        self.y_local = None
        self.buffer_X = []
        self.buffer_y = []
        self.interrupted = False

        if self.verbose:
            self.logger.info(f"[Node {self.id}] Deactivated — no more data available.")

    def reconnect(self) -> None:
        """
        Simulates a node reconnecting after an external interruption.
        Only reactivates nodes that were interrupted, not exhausted ones.
        """
        if not self.active and not self.interrupted:
            self.logger.warning(f"[Node {self.id}] Cannot reconnect — node is exhausted.")
            return
    
        self.active = True
        self.interrupted = False

        if self.verbose:
            self.logger.info(f"[Node {self.id}] Reconnected — back in rotation.")

    def interrupt(self) -> None:
        """
        Simulates an external interruption — node failure or forced disconnection.
        The node will not participate in any future round.
        """
        self.active = False
        self.interrupted = True

        if self.verbose:
            self.logger.info(f"[Node {self.id}] Interrupted — simulated failure.")
    
    def compute_bid(self) -> float:
        """
        Simulate a bid for leader election based on node capacity and load.
        
        The bid combines three factors:
        - log(capacity): smooths large capacity differences so high-capacity 
            nodes don't dominate unconditionally.
        1 + n_samples: rewards nodes with more local data — more samples
            means better representation, so higher bid.
        - noise: small random factor to break ties and add diversity,
            preventing the same node from always winning.
        """
        # Number of local training samples (proxy for current load)
        n_samples = 0 if self.X_local is None else self.X_local.shape[0]

        # Random jitter ±15% to avoid deterministic leader lock-in
        noise = random.uniform(0.85, 1.15)

        # Both capacity and samples push the bid up; log keeps capacity scaled
        bid = (math.log1p(self.capacity) * (1.0 + n_samples)) * noise

        if self.verbose:
            self.logger.info(
                f"[Node {self.id}] Bid={bid:.6f} "
                f"(capacity={self.capacity}, samples={n_samples}, noise={noise:.3f})"
            )
        return bid

    def collect_local_data(self, n_samples: int) -> None:
        """
        Fetches a new data batch from the provider and scales the features.
        """
        try:

            n_total = len(self.X_full)

            if self.verbose:
                self.logger.info(
                    f"[Node {self.id}] Total = {n_total}, "
                    f"current_index = {self.current_index}, "
                    f"remaining = {n_total - self.current_index}, "
                    f"requested = {n_samples}"
                )

            start = self.current_index
            end = min(start + n_samples, n_total)

            if end <= start:
                self.deactivate()
                return None, None

            self.X_local = self.X_full[start:end]
            self.y_local = self.y_full[start:end].reshape(-1,1)

            # Register the new batch in the ledger (views, no copy)
            self.batches[self.batch_seq] = (self.X_local, self.y_local)
            self.batch_seq += 1

            self.current_index = end
            # The node is deactivated on the NEXT request, when nothing is left (end <= start).
            # Deactivating here would clear the batch just collected, which was never sent.
        
        except RuntimeError as e:
                # If the provider has no more data, the node will not participate in this round
                self.logger.info(f"[Node {self.id}] Provider exhausted: {e}")
                self.deactivate()

    def is_ready_to_train(self, min_samples_to_train: int) -> bool:
        """
        Determines whether the node has enough data to train a model.
        If not, stores the current data in the buffer for future rounds.
        """
        # Count new samples just received in this round
        n_new = 0 if self.X_local is None else len(self.X_local)
        # Count samples previously accumulated in the buffer
        buffer_samples = sum(x.shape[0] for x in self.buffer_X)
        # Total available samples (new + from buffer)
        total_samples = n_new + buffer_samples
        # Enough total samples to proceed with training
        if total_samples >= min_samples_to_train:
            if self.buffer_X and self.X_local is not None:
                # Combine buffered and current data
                X_buf = np.vstack(self.buffer_X)
                y_buf = np.vstack(self.buffer_y)
                self.X_local = np.vstack([X_buf, self.X_local])
                self.y_local = np.vstack([y_buf, self.y_local])
                
            # Clear the buffer after using it
            self.buffer_X = []
            self.buffer_y = []
            self.ready_to_train = True

            if self.verbose:
                self.logger.info(f"[Node {self.id}] Ready to train with {self.X_local.shape[0]} samples.")
        else:
            if self.X_local is not None:
                # Case 2: Not enough data yet — store the new data in the buffer
                self.buffer_X.append(self.X_local.copy())
                self.buffer_y.append(self.y_local.copy())

                # Clear current round data to avoid accidental use
                self.X_local = None
                self.y_local = None
            
            self.ready_to_train = False

            if self.verbose:
                self.logger.info(f"[Node {self.id}] Not ready to train yet - Buffering: total {total_samples} samples (new + previous)")

    # ------------------------------------------------------------------
    # Batch ledger: exactly-once contributions
    # ------------------------------------------------------------------
    def pending_ids(self, alg) -> list:
        """
        Batches not yet committed for this algorithm (includes batches from failed rounds).
        """
        done = self.committed.get(alg, set())
        return [b for b in sorted(self.batches) if b not in done]

    def data_for(self, alg):
        """
        All uncommitted data for this algorithm, plus the batch ids it contains.
        """
        ids = self.pending_ids(alg)
        if not ids:
            return None, None, []
        X = np.vstack([self.batches[b][0] for b in ids])
        y = np.vstack([self.batches[b][1] for b in ids])
        return X, y, ids

    def commit(self, alg) -> list:
        """
        Leader acknowledged the round: mark the sent batches as committed.
        """
        ids = self.sent.pop(alg, [])
        done = self.committed.setdefault(alg, set())
        twice = done.intersection(ids)          # exactly-once check (C1): should never happen
        if twice:
            self.double_commits[alg] = self.double_commits.get(alg, 0) + len(twice)
        done.update(ids)
        self.last_committed[alg] = ids
        return ids

    def rollback(self, alg) -> None:
        """
        Round failed: keep the batches pending so they are re-sent later.
        """
        self.sent.pop(alg, None)

    def committed_data(self, alg):
        """
        Data committed for `alg` in the last successful round (used by centralized baselines).
        """
        ids = self.last_committed.pop(alg, [])
        if not ids:
            return None, None
        X = np.vstack([self.batches[b][0] for b in ids])
        y = np.vstack([self.batches[b][1] for b in ids])
        return X, y

    async def attempt_node_training(self, alg) -> bool:
        """
        Attempt local training on a single node within the algorithm's timeout.
        A node that misses the deadline keeps its batches pending (they are not lost).
        """
        X_pending, _, _ = self.data_for(alg)
        if X_pending is None:
            return False  # nothing new to contribute for this algorithm
        n = X_pending.shape[0]

        # Estimate the number of floating-point operations for this node
        flops = compute_theoretical_flops(
            alg        = alg,
            n          = n,
            epochs     = getattr(self, "epochs", None),
            batch_size = getattr(self, "batch_size", None)
        )

        # Convert FLOPs to wall-clock time estimate based on node capacity
        theoretical_time = node_theoretical_time(flops, capacity = self.capacity)
        meets_deadline   = theoretical_time <= alg.timeout

        # Skip participation if estimated time exceeds the allowed timeout
        if not meets_deadline:
            return False

        # Run local training
        update = self.train_local(alg)

        if update and self.verbose:
            self.logger.info(
            f"[Node {self.id}] Trained on {n} samples in "
            f"~{theoretical_time:.6f}s (max = {alg.timeout:.2f}s)"
        )
            
        if update is None:
            return False

        # Send the local update (with its batch ids) to the leader
        await self.send_local_updates(update, alg)
    
        return True

    def train_local(self, alg) -> tuple[np.ndarray, np.ndarray]:
        """
        Local computation on the node:
        """
        # All batches not yet committed for this algorithm (ids travel with the update)
        X, y, ids = self.data_for(alg)
        if X is None:
            return None
        self.sent[alg] = ids
        ever = self.ever_sent.setdefault(alg, set())
        self.resent_last[alg] = len(ever.intersection(ids))
        ever.update(ids)

        # Inspect alg.fit() signature to determine which interface to use
        params = inspect.signature(alg.fit).parameters
        if 'node_id' in params:
            # Stateful clients (e.g. SCAFFOLD control variates)
            local_update = alg.fit(X, y, self.epochs, self.batch_size, node_id=self.id)
        elif 'epochs' in params:
            local_update = alg.fit(X, y, self.epochs, self.batch_size)
        elif "features" in params:
            local_update = alg.fit(X, y, self.features)
        else: 
            local_update = alg.fit(X, y)
        
        if local_update and self.verbose and self.is_leader:
            self.logger.info(
                    f"[Leader {self.id}] Trained on {len(X)} samples ")
        
        return local_update
    
    def compute_global_model(self, alg: FederatedAlgorithm) -> None:
        """
        Computes the current estimate of the global beta vector using the given algorithm instance.
        """
        global_model = alg.compute_global_model(self.node_capacity_flops)
        
        if self.verbose:
            self.logger.info(f"[Node {self.id}] computes the global model")

        return global_model
    
    def get_global_model(self, alg: FederatedAlgorithm) -> np.ndarray:
        """
        Call the 'get_global_model' method from the given algorithm instance
        and return the global model
        """
        return alg.get_global_model()
