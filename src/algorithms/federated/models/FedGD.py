from ..base import FederatedAlgorithm
from typing import Tuple

from src.algorithms.federated.aggregation.base import BaseAggregation

import logging
import numpy as np
import time
import math

logger = logging.getLogger(__name__)

class FedGD(FederatedAlgorithm):
    """
    Federated Gradient Descent (FedGD) strategy for linear regression.
    """
    def __init__(
        self,
        n_features: int,
        timeout: float,
        aggregation_method: BaseAggregation,
        initial_weights: np.ndarray,
        eta0: float = 0.01,
        tol: float =1e-4,
        penalty: str = None,
        lambda_reg: float = 0.001,
        mu = 0.0,
        early_stopping: bool = False,
        epochs_as_updates: bool = True,
        verbose = False
    ):
        super().__init__(
            initial_weights = initial_weights,
            penalty = penalty,
            lambda_reg = lambda_reg,
            verbose = verbose
        )

        self.n_features = n_features
        # Learning rate for SGD updates
        self.eta0 = eta0
        # Tolerance for early stopping in SGD updates
        self.tol = tol
        # Accumulator for weighted local updates
        self.weighted_sum = None
        self.tau_weighted_sum = None
        # Injected aggregation strategy
        self.aggregation_method = aggregation_method
        self.mu = mu
        self.timeout = timeout
        # Local early stopping on a held-out split (noisy with few samples)
        self.early_stopping = early_stopping
        # If True, node_epochs is a target number of updates (equalizes SGD/MGD/BGD)
        self.epochs_as_updates = epochs_as_updates

    def flops_info(self, n: int, epochs: int, batch_size: int) -> dict:
        """
        Returns a breakdown of theoretical FLOPs for one local training round.
        """
        d_eff = self.n_features + 1  # intercept included

        updates_per_epoch    = max(1, n // batch_size)
        grad_cost_per_update = 2 * batch_size * d_eff  # forward + backward

        return {
            "updates_per_epoch":    updates_per_epoch,
            "grad_cost_per_update": grad_cost_per_update,
            "total_flops": float(epochs * updates_per_epoch * grad_cost_per_update)
        }

    def fit(
        self, 
        X_i, 
        y_i,
        epochs,
        batch_size
    ) -> Tuple[np.ndarray, int]:
        """
        Performs local training using Gradient Descent (GD) to minimize MSE loss.
        """
        # Add intercept term (bias)
        X_i = self.add_intercept(X_i)

        # node_epochs is a target number of updates: epochs needed to reach it with mini-batches
        if self.epochs_as_updates:
            epochs = math.ceil(epochs / math.ceil(X_i.shape[0] / batch_size))

        # Initialize local weights as a column vector copy of the global model
        w_global = np.asarray(self.global_model, dtype=float).reshape(-1, 1)
        
        start = time.perf_counter()

        # Perform local training using gradient descent (overflow -> handled below)
        with np.errstate(all="ignore"):
            w_i, tau_i, n_i = self.gradient_descent(
                X = X_i,
                y = y_i,
                w_init = w_global.copy(),
                eta0 = self.eta0,
                epochs = epochs,
                batch_size = batch_size,
                mu = self.mu,
                lambda_reg = self.lambda_reg,
                tol = self.tol,
                early_stopping = self.early_stopping
            )

        elapsed = time.perf_counter() - start

        # Diverged node (learning rate too high): discard its update
        if not np.all(np.isfinite(w_i)):
            raise FloatingPointError("FedGD local update diverged (lower eta0)")

        return (w_i, w_global, n_i, tau_i, elapsed)
    
    def complete_aggregation_round(self) -> np.ndarray:

        t_fit = max(self.round_fit_times) if self.round_fit_times else 0.0
        start = time.perf_counter()
        
        self.global_model  = self.aggregation_method.process_global_update(
            global_model = self.global_model,
            total_samples = self.round_total_samples,
            weighted_sum = self.weighted_sum,
            tau_weighted_sum = self.tau_weighted_sum
        )

        self.round_elapsed += (time.perf_counter() - start) + t_fit

        self.weighted_sum = None
        self.tau_weighted_sum = None
        self.round_total_samples = 0 

        return self.global_model