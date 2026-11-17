"""
Federated MLP for tabular regression built on scikit-learn's MLPRegressor
(same model as the CentMLP baseline). No custom network code:
  - each node loads the global weights into a local MLPRegressor,
  - trains `local_epochs` passes with partial_fit (sklearn's own Adam/SGD),
  - returns coefs_ + intercepts_; FedAvg / FedNova average them.
FedProx is not supported (MLPRegressor has no proximal term).
"""
import copy
import math
import time
import logging
import warnings
from typing import List, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from ..base import FederatedAlgorithm
from src.algorithms.federated.aggregation.base import BaseAggregation

logger = logging.getLogger(__name__)


class MLPWeights:
    """Global weights [W1..WL, b1..bL] with get/set_weights, as expected by the aggregators."""
    def __init__(self, coefs: List[np.ndarray], intercepts: List[np.ndarray]):
        self.n_layers = len(coefs)
        self.set_weights(list(coefs) + list(intercepts))

    def get_weights(self) -> List[np.ndarray]:
        return [w.copy() for w in self.weights]

    def set_weights(self, weights: List[np.ndarray]) -> None:
        self.weights = [np.asarray(w, dtype=np.float64).copy() for w in weights]

    @property
    def coefs(self):
        return self.weights[:self.n_layers]

    @property
    def intercepts(self):
        return self.weights[self.n_layers:]


class FedMLP(FederatedAlgorithm):
    def __init__(
        self,
        n_features: int,
        timeout: float,
        aggregation_method: BaseAggregation,
        hidden_layers: List[int] = (64, 32),
        activation: str = "relu",
        optimizer: str = "adam",
        eta0: float = 0.001,
        lambda_reg: float = 1e-4,
        local_epochs: int = 5,
        seed: int = 42,
        verbose: bool = False,
    ):
        # Template model: one dummy partial_fit only creates sklearn's internal state,
        # then the weights are replaced by sklearn's own Glorot initialisation.
        self.template = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers), activation=activation,
            solver=optimizer, learning_rate_init=eta0, alpha=lambda_reg,
            shuffle=True, random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.template.partial_fit(np.zeros((2, n_features)), np.zeros(2))

        rng = np.random.RandomState(seed)
        sizes = [n_features, *list(hidden_layers), 1]
        factor = 2.0 if activation == "logistic" else 6.0
        coefs, intercepts = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            bound = np.sqrt(factor / (a + b))
            coefs.append(rng.uniform(-bound, bound, (a, b)))
            intercepts.append(rng.uniform(-bound, bound, b))

        super().__init__(initial_weights=MLPWeights(coefs, intercepts),
                         lambda_reg=lambda_reg, verbose=verbose)

        self.n_features = n_features
        self.sizes = sizes
        self.eta0 = eta0
        self.local_epochs = local_epochs
        self.timeout = timeout
        self.aggregation_method = aggregation_method
        self.weighted_sum = None
        self.tau_weighted_sum = None

    def _model_with(self, weights: MLPWeights) -> MLPRegressor:
        """Fresh copy of the template loaded with the given weights and a new optimizer."""
        model = copy.deepcopy(self.template)
        model.coefs_ = [w.copy() for w in weights.coefs]
        model.intercepts_ = [b.copy() for b in weights.intercepts]
        model.learning_rate_init = self.eta0
        if hasattr(model, "_optimizer"):
            del model._optimizer  # local optimizer state restarts every round (standard FedAvg)
        return model

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self._model_with(self.global_model).predict(np.asarray(X_test, dtype=float))

    def flops_info(self, n: int, epochs: int, batch_size: int) -> dict:
        n_params = sum(a * b + b for a, b in zip(self.sizes[:-1], self.sizes[1:]))
        return {"total_flops": float(6 * n_params * n * max(1, self.local_epochs))}

    def fit(self, X_i, y_i, epochs, batch_size) -> Tuple:
        """Local training: `local_epochs` passes of sklearn partial_fit from the global weights."""
        X = np.asarray(X_i, dtype=float)
        y = np.asarray(y_i, dtype=float).ravel()
        w_global = self.global_model.get_weights()

        model = self._model_with(self.global_model)
        model.batch_size = max(1, min(batch_size or len(y), len(y)))

        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for _ in range(self.local_epochs):
                model.partial_fit(X, y)
        elapsed = time.perf_counter() - start

        W = [w.copy() for w in model.coefs_] + [b.copy() for b in model.intercepts_]
        if not all(np.all(np.isfinite(w)) for w in W):
            raise FloatingPointError("FedMLP diverged (lower eta0)")

        # Local steps (for FedNova normalisation)
        tau = self.local_epochs * math.ceil(len(y) / model.batch_size)
        return (W, w_global, len(y), tau, elapsed)

    def complete_aggregation_round(self):
        t_fit = max(self.round_fit_times) if self.round_fit_times else 0.0
        start = time.perf_counter()
        self.global_model = self.aggregation_method.process_global_update(
            global_model     = self.global_model,
            total_samples    = self.round_total_samples,
            weighted_sum     = self.weighted_sum,
            tau_weighted_sum = self.tau_weighted_sum,
        )
        self.round_elapsed += (time.perf_counter() - start) + t_fit
        self.weighted_sum = None
        self.tau_weighted_sum = None
        self.round_total_samples = 0
        return self.global_model
