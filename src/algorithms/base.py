# federated/algorithm.py
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from typing import Any

import logging
import numpy as np
import math
import asyncio

class BaseAlgorithm(ABC):

    def __init__(
            self, 
            initial_weights = None, 
            penalty: str = None, 
            lambda_reg: float = 0.0,
            verbose = False
    ):
        """
        Store the initial global model and the regularization settings.
        """
        self.global_model = initial_weights
        self.lock = asyncio.Lock() 
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Any:
        """
        Train the model (local or global).
        """
        pass

    def add_intercept(self,X: np.ndarray) -> np.ndarray:
        """
        Adds an intercept (bias) column to a 2D feature matrix.
        """
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def compute_gradient(self,
            X: np.ndarray, 
            y: np.ndarray, 
            w: np.ndarray, 
            mu: float = 0.0, 
            w_global: np.ndarray = None,
            lambda_reg: float = 0.0
    ) -> np.ndarray:
        """
        Compute the gradient of the MSE loss, optionally including the FedProx proximal term.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            y: Target values, shape (n_samples, 1)
            w: Current weights, shape (n_features, 1)
            mu: Proximal regularization parameter (default: 0.0, no proximal term)
            w_global: Global model weights, shape (n_features, 1) (default: None)
            l2: L2 regularization parameter (default: 0.0)
        
        Returns:
            Gradient of the loss w.r.t. w, shape (n_features, 1)
        """
        # Compute the predicted values
        y_pred = X.dot(w)
        # Compute the prediction error
        error = y_pred - y
        # Compute the gradient of the MSE loss with respect to weights
        grad = (2 / X.shape[0]) * X.T.dot(error)
        
        # Add proximal term if mu > 0 and w_global is provided
        if mu > 0.0:
            grad += mu * (w - w_global)

        # L2 term, scaled by 1/n so the local objective matches ridge: ||Xw-y||^2 + lambda*||w||^2
        if self.penalty is not None and str(self.penalty).lower() in ("constant", "dynamic"):
            grad += (2 * lambda_reg / X.shape[0]) * w
        
        return grad
    
    def gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w_init: np.ndarray,
        eta0: float,
        epochs: int,
        batch_size: int,
        mu: float = 0.0,
        lambda_reg: float = 0.001,
        tol: float = 1e-4,
        n_iter_no_change: int = 5,
        validation_fraction: float = 0.1,
        early_stopping: bool = False,
    ) -> tuple[np.ndarray, int, int]:
        """
        Local mini-batch gradient descent with a constant learning rate.
        FedAvg: mu = 0; FedProx: mu > 0 (proximal term towards the initial global weights).
        With early_stopping, training stops when the loss on a held-out split of the node's
        data does not improve for n_iter_no_change updates, and the best weights are returned.
        Returns the weights, the number of local updates and the number of training samples.
        """
        if early_stopping and X.shape[0] >= 10:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_fraction, shuffle=True)
        else:
            early_stopping = False
            X_train, y_train = X, y

        w = w_init.copy()
        w_global = w_init.copy()
        best_w, best_loss, no_improve = w.copy(), np.inf, 0
        n, tau_i = X_train.shape[0], 0

        for _ in range(epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, batch_size):
                b = idx[start:start + batch_size]
                grad = self.compute_gradient(X_train[b], y_train[b], w, mu, w_global, lambda_reg=lambda_reg)
                w = self.update_weights(w, grad, eta0)
                tau_i += 1

                if early_stopping:
                    val_loss = np.mean((X_val @ w - y_val) ** 2)
                    if val_loss < best_loss - tol:
                        best_loss, best_w, no_improve = val_loss, w.copy(), 0
                    else:
                        no_improve += 1
                    if no_improve >= n_iter_no_change:
                        return best_w, tau_i, n

        return (best_w if early_stopping else w), tau_i, n
    
    def update_weights(self, w: np.ndarray, grad: np.ndarray, lr: float) -> np.ndarray:
        """
        Update the weight vector by taking a step opposite to the gradient.
        """
        new_w = w - lr * grad
        return new_w
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Required by BaseAlgorithm: makes predictions using the current global model.
        This is the method used for evaluation (MSE, R²).
        """

        X_int = self.add_intercept(X_test)

        return (X_int @ self.global_model.reshape(-1, 1)).ravel()

    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluates the global model using MSE and R² on a test dataset.
        """
        
        y_pred = self.predict(X_test)

        # Diverged model: report NaN instead of crashing the run
        if not np.all(np.isfinite(y_pred)):
            return {"mse": float("nan"), "r2": float("nan"), "y_pred": y_pred, "y_true": y_test}

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "mse": mse,
            "r2": r2,
            "y_pred": y_pred,
            "y_true": y_test
        }
