
import numpy as np
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from ..base import BaseAlgorithm

class Cent(BaseAlgorithm):
    """
    Centralized linear regression baseline.
    """

    def __init__(
        self,
        penalty: str = None,
        lambda_reg: float = 0.01,
        cv_folds: int = 5,
        verbose = False
    ):
        super().__init__(
            penalty = penalty,
            lambda_reg = lambda_reg,
            verbose = verbose
        )

        # Tolerance for early stopping in SGD updates
        self.cv_folds = cv_folds

    def fit(self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """
        Fits a linear regression model (OLS or Ridge) to solve Aβ = B.

        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        model = None

        penalty_type = self.penalty.lower()

        # Column of ones for the intercept
        X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

        if penalty_type == "none":
            model = LinearRegression(fit_intercept = True)
            model.fit(X, y)

        elif penalty_type == "constant":
            model = Ridge(alpha = self.lambda_reg, fit_intercept=True)
            model.fit(X, y)

        elif penalty_type == "dynamic":

            # Build alpha range around center_lambda
            lower = max(self.lambda_reg / 10, 1e-8)
            upper = self.lambda_reg * 10 + 1e-8
            lambdas_reg = np.logspace(np.log10(lower), np.log10(upper), 50)

            model = RidgeCV(
                alphas = lambdas_reg,
                cv = self.cv_folds,
                scoring = "neg_mean_squared_error",
                fit_intercept=True)
            
            model.fit(X, y)

        elif penalty_type == "ols":

            U, S, Vt = np.linalg.svd(X_aug, full_matrices=False)
            S_reg_inv = np.diag([s/(s**2 + self.lambda_reg) for s in S])
            beta = Vt.T @ S_reg_inv @ U.T @ y

            model = beta
        else:
            raise ValueError(f"Unknown penalty type '{self.penalty}'. Choose 'none', 'constant' or 'dynamic'.")
        
        self.global_model = model

    def evaluate(self, X_test, y_test):
        """
        Evaluates model to obtain metrics: MSE and R².
        """

        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float).ravel()

        # Check if input is 3D (time-series data) and flatten it for linear models
        if X_test.ndim == 3:
            X_test = X_test.reshape(X_test.shape[0], -1)

        if isinstance(self.global_model, np.ndarray):
            # OLS manual
            X_aug = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
            y_pred = X_aug @ self.global_model
        else:
            # sklearn
            y_pred = self.global_model.predict(X_test)

        # Compute and return performance metrics
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }