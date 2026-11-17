
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from ..base import BaseAlgorithm

class CentRF(BaseAlgorithm):
    """
    Centralized Random Forest regression baseline..
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        cv_folds: int = 5,
        optimal: bool = False,
        verbose = False
    ):
        super().__init__(
            verbose = verbose
        )

        if isinstance(max_depth, str) and max_depth.lower() == "none":
            max_depth = None

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.cv_folds = cv_folds
        self.optimal = optimal

    def fit(
            self,
            X: np.ndarray, 
            y: np.ndarray
    ) -> np.ndarray:
        """
        Fits a Random Forest regression model.

        """
        # Ensure X and y are numpy arrays of type float
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()

        # Initialize the Random Forest model with fixed hyperparameters
        model = RandomForestRegressor(
            n_estimators = self.n_estimators,
            max_depth = self.max_depth,
            random_state=42,
            n_jobs=-1
        )

        if self.optimal: 
            # Define the hyperparameter grid for tuning
            param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            }

            grid = GridSearchCV(
                estimator = model,
                param_grid = param_grid,
                cv=self.cv_folds,
                scoring = "neg_mean_squared_error",
                n_jobs = -1,
                refit = True  # refit the best model on all the data
            )

            # Perform hyperparameter search
            grid.fit(X, y)
            # Save the best model found
            self.global_model = grid.best_estimator_

        else:
            # Fit the model directly with fixed hyperparameters
            model.fit(X, y)
            self.global_model = model

    def evaluate(self, X_test, y_test):
        """
        Evaluates model to obtain metrics: MSE and R².
        """

        X_test = np.asarray(X_test, dtype=float)
        y_test = np.asarray(y_test, dtype=float).ravel()

        y_pred = self.global_model.predict(X_test)

        # Compute and return performance metrics
        return {
            "mse": mean_squared_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred)
        }