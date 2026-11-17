import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from .feature_selection import FeatureSelector

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

class XGBoostSelector(FeatureSelector):
    """
    Feature selector using XGBoost feature importance.
    
    This implementation uses XGBoost regressor to compute feature importance
    scores using multiple importance types (gain, weight, cover) and selects 
    the most important features. It supports early stopping and cross-validation
    for robust feature selection.
    """
    
    def __init__(
        self,
        n_features: Optional[int] = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: Optional[int] = 42,
        importance_type: str = 'gain',
        cv_folds: int = 5,
        use_cv_importance: bool = True,
        early_stopping_rounds: Optional[int] = 10,
        importance_threshold: Optional[float] = None,
        verbose: bool = False
    ):
        """
        Initialize the XGBoost feature selector.
        
        Args:
            n_features: Number of features to select. If None, uses importance_threshold.
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of trees.
            learning_rate: Boosting learning rate.
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns when constructing each tree.
            reg_alpha: L1 regularization term on weights.
            reg_lambda: L2 regularization term on weights.
            random_state: Random state for reproducibility.
            importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover').
            cv_folds: Number of cross-validation folds for stability.
            use_cv_importance: Whether to use cross-validated importance scores.
            early_stopping_rounds: Early stopping rounds for training.
            importance_threshold: Minimum importance threshold for feature selection.
            verbose: Whether to enable verbose logging.
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        super().__init__(n_features=n_features, verbose=verbose)
        
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample  
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.importance_type = importance_type
        self.cv_folds = cv_folds
        self.use_cv_importance = use_cv_importance
        self.early_stopping_rounds = early_stopping_rounds
        self.importance_threshold = importance_threshold
        
        # Validate importance type
        valid_importance_types = ['gain', 'weight', 'cover', 'total_gain', 'total_cover']
        if self.importance_type not in valid_importance_types:
            raise ValueError(f"importance_type must be one of {valid_importance_types}")
        
        # Initialize the XGBoost model parameters
        self.xgb_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'verbosity': 1 if verbose else 0
        }
        
        self.xgb_model = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'XGBoostSelector':
        """
        Fit the XGBoost feature selector to the data.
        
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
            self.logger.info(f"Fitting XGBoost on {X.shape[0]} samples with {n_original_features} features")
        
        if self.use_cv_importance:
            # Use cross-validation to get more stable importance scores
            self.feature_scores_ = self._compute_cv_importance(X, y)
        else:
            # Use single model importance
            self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
            
            # Split data for early stopping if specified
            if self.early_stopping_rounds is not None:
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=self.random_state
                )
                
                self.xgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                self.xgb_model.fit(X, y)
            
            # Get feature importance
            importance_dict = self.xgb_model.get_booster().get_score(importance_type=self.importance_type)
            
            # Convert to array format (features that didn't appear get 0 importance)
            self.feature_scores_ = np.array([
                importance_dict.get(f'f{i}', 0.0) for i in range(len(self.original_features_))
            ])
        
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
            raise ValueError("XGBoostSelector must be fitted before transform")
        
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
            
            # Create XGBoost parameters for this fold
            fold_params = self.xgb_params.copy()
            fold_params['random_state'] = self.random_state + fold if self.random_state else None
            
            # Create and train model
            fold_xgb = xgb.XGBRegressor(**fold_params)
            
            if self.early_stopping_rounds is not None:
                # Further split training data for early stopping
                from sklearn.model_selection import train_test_split
                X_tr, X_es, y_tr, y_es = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=fold
                )
                
                fold_xgb.fit(
                    X_tr, y_tr,
                    eval_set=[(X_es, y_es)],
                    verbose=False
                )
            else:
                fold_xgb.fit(X_train, y_train)
            
            # Get importance scores for this fold
            importance_dict = fold_xgb.get_booster().get_score(importance_type=self.importance_type)
            fold_importance = np.array([
                importance_dict.get(f'f{i}', 0.0) for i in range(len(self.original_features_))
            ])
            
            importance_scores.append(fold_importance)
            
            if self.verbose:
                val_score = fold_xgb.score(X_val, y_val)
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
            self.logger.info(f"Selected features with {self.importance_type} importance scores:")
            for feat, imp in selected_pairs[:10]:  # Show top 10
                self.logger.info(f"  {feat}: {imp:.4f}")
    
    def get_model_performance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Get the performance of the XGBoost model on the given data.
        
        Args:
            X: Input features DataFrame
            y: Target variable Series
            
        Returns:
            Dictionary with performance metrics
        """
        if not self.is_fitted_:
            raise ValueError("XGBoostSelector must be fitted before getting performance")
        
        from sklearn.model_selection import cross_val_score
        
        # Use the selected features
        X_selected = self.transform(X)
        
        # Create evaluation model with same parameters
        eval_xgb = xgb.XGBRegressor(**self.xgb_params)
        
        # Get cross-validation scores
        cv_scores = cross_val_score(eval_xgb, X_selected, y, cv=self.cv_folds, scoring='r2')
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'n_selected_features': len(self.selected_features_),
            'n_original_features': len(self.original_features_),
            'feature_reduction_ratio': len(self.selected_features_) / len(self.original_features_),
            'importance_type': self.importance_type
        }
    
    def get_feature_interactions(self, top_k: int = 10) -> Optional[List[tuple]]:
        """
        Get top feature interactions discovered by XGBoost.
        
        Args:
            top_k: Number of top interactions to return
            
        Returns:
            List of feature interaction tuples, or None if model not available
        """
        if not self.is_fitted_ or self.xgb_model is None:
            return None
        
        try:
            # Get interaction scores from the booster
            interactions = self.xgb_model.get_booster().get_score(importance_type='gain', fmap='')
            
            # Filter for interaction terms (contain 'f' followed by multiple feature indices)
            interaction_pairs = []
            for feature_name, score in interactions.items():
                if '|' in feature_name or 'interaction' in feature_name.lower():
                    interaction_pairs.append((feature_name, score))
            
            # Sort by importance and return top k
            interaction_pairs.sort(key=lambda x: x[1], reverse=True)
            return interaction_pairs[:top_k]
        
        except Exception as e:
            if self.verbose:
                self.logger.warning(f"Could not extract feature interactions: {e}")
            return None 