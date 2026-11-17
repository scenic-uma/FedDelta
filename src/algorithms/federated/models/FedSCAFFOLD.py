"""
SCAFFOLD (Karimireddy et al., ICML 2020) for ridge regression.

Local step:
    w <- w - eta * (g_i(w) - c_i + c)
Client control variate (option II):
    c_i_new = c_i - c + (x - w) / (K * eta)
Server updates (Algorithm 1):
    x <- x + server_lr * (1 / |S_t|) * sum_i (w_i - x)
    c <- c + (1 / N) * sum_i (c_i_new - c_i)
where S_t = clients aggregated this round and N = total number of clients.

Fault tolerance: a client's new c_i is applied only when its round is committed;
if the round is aborted (no quorum, leader crash) the client keeps its previous c_i.
"""
import math
import time
import logging
from typing import Dict, Optional

import numpy as np

from ..base import FederatedAlgorithm

logger = logging.getLogger(__name__)


class FedSCAFFOLD(FederatedAlgorithm):
    def __init__(
        self,
        n_features: int,
        timeout: float,
        initial_weights: np.ndarray,
        n_clients: Optional[int] = None,
        eta0: float = 0.01,
        server_lr: float = 1.0,
        lambda_reg: float = 0.01,
        penalty: str = "constant",
        weighting: str = "uniform",
        epochs_as_updates: bool = True,
        verbose: bool = False,
    ):
        super().__init__(initial_weights=initial_weights, penalty=penalty,
                         lambda_reg=lambda_reg, verbose=verbose)
        if eta0 <= 0:
            raise ValueError("eta0 must be > 0")
        if n_clients is not None and n_clients < 1:
            raise ValueError("n_clients must be >= 1")
        if weighting not in ("uniform", "samples"):
            raise ValueError("weighting must be 'uniform' or 'samples' (FedAvg-style)")

        self.n_features = n_features
        self.n_clients = n_clients          # N, injected at runtime = number of nodes
        self.timeout = timeout
        self.eta0 = eta0
        self.server_lr = server_lr
        self.weighting = weighting
        self.epochs_as_updates = epochs_as_updates

        d = n_features + 1
        self.c = np.zeros((d, 1))                    # server control variate
        self.c_i: Dict[str, np.ndarray] = {}         # client control variates (committed)
        self._reset_round()

    def _reset_round(self):
        self._pending_ci: Dict[str, np.ndarray] = {}  # new c_i computed this round
        self._round_ci: Dict[str, np.ndarray] = {}    # new c_i of aggregated clients
        self.sum_dy = None
        self.sum_dc = None
        self.sum_w = 0.0
        self.round_total_samples = 0
        self.n_round_clients = 0

    def flops_info(self, n: int, epochs: int, batch_size: int) -> dict:
        d = self.n_features + 1
        steps = epochs if self.epochs_as_updates else epochs * max(1, n // batch_size)
        return {"total_flops": float(steps * (2 * batch_size * d + 4 * d))}

    def fit(self, X_i, y_i, epochs, batch_size, node_id: Optional[str] = None):
        if node_id is None:
            raise ValueError("SCAFFOLD requires a unique node_id for each client")

        X = self.add_intercept(np.asarray(X_i, dtype=float))
        y = np.asarray(y_i, dtype=float).reshape(-1, 1)
        n = X.shape[0]
        if n == 0:
            raise ValueError("Client dataset cannot be empty")

        b = max(1, min(batch_size or n, n))
        K = max(1, int(epochs if self.epochs_as_updates else epochs * math.ceil(n / b)))

        x = np.asarray(self.global_model, dtype=float).reshape(-1, 1)
        c_i = self.c_i.get(node_id, np.zeros_like(self.c)).copy()
        w = x.copy()

        start = time.perf_counter()
        idx = np.random.permutation(n)
        pos = 0
        with np.errstate(all="ignore"):
            for _ in range(K):
                if pos + b > n:
                    idx, pos = np.random.permutation(n), 0
                batch = idx[pos:pos + b]
                pos += b
                g = self.compute_gradient(X[batch], y[batch], w, lambda_reg=self.lambda_reg)
                w = w - self.eta0 * (g - c_i + self.c)

            # Option II control variate
            c_i_new = c_i - self.c + (x - w) / (K * self.eta0)
        elapsed = time.perf_counter() - start

        if not (np.all(np.isfinite(w)) and np.all(np.isfinite(c_i_new))):
            raise FloatingPointError("SCAFFOLD diverged (lower eta0)")

        # Applied only if the round is committed (see complete_aggregation_round)
        self._pending_ci[node_id] = c_i_new
        return (w - x, c_i_new - c_i, n, K, elapsed, node_id)

    async def aggregate_update(self, update):
        async with self.lock:
            dy, dc, n_i, _, elapsed, node_id = update
            self.round_fit_times.append(elapsed)
            start = time.perf_counter()

            if node_id in self._pending_ci:
                self._round_ci[node_id] = self._pending_ci.pop(node_id)

            weight = 1.0 if self.weighting == "uniform" else float(n_i)
            self.sum_dy = dy * weight if self.sum_dy is None else self.sum_dy + dy * weight
            self.sum_dc = dc.copy() if self.sum_dc is None else self.sum_dc + dc
            self.sum_w += weight
            self.round_total_samples += n_i
            self.n_round_clients += 1
            self.round_elapsed += time.perf_counter() - start

    def complete_aggregation_round(self):
        t_fit = max(self.round_fit_times) if self.round_fit_times else 0.0
        start = time.perf_counter()

        if self.n_round_clients > 0:
            x = np.asarray(self.global_model, dtype=float).reshape(-1, 1)
            x = x + self.server_lr * self.sum_dy / self.sum_w

            # Commit the control variates of the aggregated clients
            self.c_i.update(self._round_ci)
            N = self.n_clients or max(1, len(self.c_i))
            self.c = self.c + self.sum_dc / N
            self.global_model = x.ravel()

        self.round_elapsed += (time.perf_counter() - start) + t_fit
        self._reset_round()
        return self.global_model

    def comm_floats(self):
        """Model + control variate in both directions."""
        p = self.n_features + 1
        return 2 * p, 2 * p

    def discard_round(self) -> None:
        """Aborted round: clients keep their previous control variates."""
        self._reset_round()
        super().discard_round()
