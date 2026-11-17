import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from typing import Optional, List
from src.feature_selection import FeatureSelector

class RandomForestSelector(FeatureSelector):
    """
    Feature selector using Random Forest feature importance.
    
    This implementation uses a Random Forest regressor to compute feature importance
    scores and selects the top-k features based on these scores. It can also use
    cross-validation to ensure stable feature selection.
    """
    
    def __init__(
        self, 
        n_features: Optional[int] = None,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = 42,
        cv_folds: int = 5,
        use_cv_importance: bool = True,
        importance_threshold: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize the Random Forest feature selector.
        
        Args:
            n_features: Number of features to select. If None, uses importance_threshold.
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees. None means unlimited.
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at leaf node.
            random_state: Random state for reproducibility.
            cv_folds: Number of cross-validation folds for stability.
            use_cv_importance: Whether to use cross-validated importance scores.
            importance_threshold: Minimum importance threshold for feature selection.
            verbose: Whether to enable verbose logging.
        """
        super().__init__(n_features=n_features, verbose=verbose)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.use_cv_importance = use_cv_importance
        self.importance_threshold = importance_threshold
        
        # Initialize the Random Forest model
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForestSelector':
        """
        Fit the Random Forest feature selector to the data.
        
        Args:
            X: Input features DataFrame of shape (n_samples, n_features)
            y: Target variable Series of shape (n_samples,)
            
        Returns:
            self: Returns self to allow method chaining
        """
        # Validate input data
        X, y = self._validate_input(X, y)
        
        # Store original feature names
        self.original_features_ = list(X.columns)
        n_original_features = len(self.original_features_)
        
        if self.verbose:
            self.logger.info(f"Fitting Random Forest on {X.shape[0]} samples with {n_original_features} features")
        
        if self.use_cv_importance:
            # Use cross-validation to get more stable importance scores
            self.feature_scores_ = self._compute_cv_importance(X, y)
        else:
            # Use single model importance
            self.rf_model.fit(X, y)
            self.feature_scores_ = self.rf_model.feature_importances_
        
        # Select features based on importance scores
        self._select_features()
        
        # Mark as fitted
        self.is_fitted_ = True
        
        # Log results
        self._log_selection_results(n_original_features, len(self.selected_features_))
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by selecting the important features.
        
        Args:
            X: Input features DataFrame of shape (n_samples, n_features)
            
        Returns:
            DataFrame with selected features of shape (n_samples, k_features)
        """
        if not self.is_fitted_:
            raise ValueError("RandomForestSelector must be fitted before transform")
        
        # Validate input
        X, _ = self._validate_input(X)
        
        # Check if all required features are present
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in transform: {missing_features}")
        
        # Return DataFrame with selected features
        return X[self.selected_features_].copy()
    
    def _compute_cv_importance(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Compute feature importance using cross-validation for stability.
        
        Args:
            X: Input features DataFrame
            y: Target variable Series
            
        Returns:
            Array of average feature importance scores across CV folds
        """
        from sklearn.model_selection import KFold
        
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        importance_scores = []
        
        if self.verbose:
            self.logger.info(f"Computing feature importance using {self.cv_folds}-fold cross-validation")
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create a new model for each fold to avoid contamination
            fold_rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state + fold,  # Different seed per fold
                n_jobs=-1
            )
            
            fold_rf.fit(X_train, y_train)
            importance_scores.append(fold_rf.feature_importances_)
            
            if self.verbose:
                val_score = fold_rf.score(X_val, y_val)
                self.logger.debug(f"Fold {fold + 1} R² score: {val_score:.4f}")
        
        # Return mean importance across folds
        return np.mean(importance_scores, axis=0)
    
    def _select_features(self) -> None:
        """
        Select features based on importance scores and selection criteria.
        """
        feature_importance_pairs = list(zip(self.original_features_, self.feature_scores_))
        
        # Sort by importance (descending)
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        if self.n_features is not None:
            # Select top-k features
            n_select = min(self.n_features, len(self.original_features_))
            selected_pairs = feature_importance_pairs[:n_select]
        elif self.importance_threshold is not None:
            # Select features above threshold
            selected_pairs = [(feat, imp) for feat, imp in feature_importance_pairs 
                            if imp >= self.importance_threshold]
            
            if len(selected_pairs) == 0:
                # If no features meet threshold, select the most important one
                selected_pairs = [feature_importance_pairs[0]]
                if self.verbose:
                    self.logger.warning(f"No features above threshold {self.importance_threshold}, selecting top feature")
        else:
            # Default: select features with above-average importance
            mean_importance = np.mean(self.feature_scores_)
            selected_pairs = [(feat, imp) for feat, imp in feature_importance_pairs 
                            if imp >= mean_importance]
            
            if len(selected_pairs) == 0:
                # Fallback: select top 50% of features
                n_select = max(1, len(self.original_features_) // 2)
                selected_pairs = feature_importance_pairs[:n_select]
                if self.verbose:
                    self.logger.warning(f"No features above average importance, selecting top {n_select} features")
        
        # Extract selected feature names
        self.selected_features_ = [feat for feat, _ in selected_pairs]
        
        if self.verbose:
            self.logger.info(f"Selected features with importance scores:")
            for feat, imp in selected_pairs[:10]:  # Show top 10
                self.logger.info(f"  {feat}: {imp:.4f}")
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Get the performance of the Random Forest model on the given data.
        
        Args:
            X: Input features DataFrame
            y: Target variable Series
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_fitted_:
            raise ValueError("RandomForestSelector must be fitted before getting performance")
        
        # Use the selected features
        X_selected = self.transform(X)
        
        # Fit a new model on selected features
        eval_rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Get cross-validation scores
        cv_scores = cross_val_score(eval_rf, X_selected, y, cv=self.cv_folds, scoring='r2')
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_selected_features': len(self.selected_features_),
            'n_original_features': len(self.original_features_),
            'feature_reduction_ratio': len(self.selected_features_) / len(self.original_features_)
        } 