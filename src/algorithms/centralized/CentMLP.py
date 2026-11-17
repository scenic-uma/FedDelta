import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from ..base import BaseAlgorithm

class CentMLP(BaseAlgorithm):
    """
    Centralized MLP baseline (same architecture as FedMLP).
    Warm-started: each round runs `epochs` passes over all data seen so far.
    """
    def __init__(self, hidden_layers=(64, 32), activation: str = "relu",
                 eta0: float = 0.001, lambda_reg: float = 1e-4, epochs: int = 5,
                 batch_size: int = 64, seed: int = 42, verbose: bool = False):
        super().__init__(lambda_reg=lambda_reg, verbose=verbose)
        self.global_model = MLPRegressor(
            hidden_layer_sizes=tuple(hidden_layers), activation=activation,
            solver="adam", learning_rate_init=eta0, alpha=lambda_reg,
            batch_size=batch_size, max_iter=epochs, warm_start=True,
            shuffle=True, random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.global_model.fit(X, np.asarray(y, dtype=float).ravel())

    def evaluate(self, X_test, y_test) -> dict:
        X = np.asarray(X_test, dtype=float)
        y_pred = self.global_model.predict(X)
        y_test = np.asarray(y_test, dtype=float).ravel()
        return {"mse": mean_squared_error(y_test, y_pred), "r2": r2_score(y_test, y_pred)}
