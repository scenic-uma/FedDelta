from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging
from typing import Union, List, Optional, Tuple

class FeatureSelector(ABC):
    """
    Abstract base class for feature selection algorithms.
    
    This interface defines the contract for all feature selection implementations.
    Each concrete implementation should inherit from this class and implement
    the required abstract methods.
    
    The main functionality is to reduce the dimensionality of a dataset by
    selecting the most important features while eliminating redundant ones.
    """
    
    def __init__(self, n_features: Optional[int] = None, verbose: bool = False):
        """
        Initialize the feature selector.
        
        Args:
            n_features: Maximum number of features to select. If None, will be determined
                       by the specific algorithm implementation.
            verbose: Whether to enable verbose logging.
        """
        self.n_features = n_features
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.selected_features_ = None
        self.feature_scores_ = None
        self.is_fitted_ = False
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector':
        """
        Fit the feature selector to the data.
        
        Args:
            X: Input features DataFrame of shape (n_samples, n_features)
            y: Target variable Series of shape (n_samples,)
            
        Returns:
            self: Returns self to allow method chaining
        """
        pass
    
    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset by selecting the important features.
        
        Args:
            X: Input features DataFrame of shape (n_samples, n_features)
            
        Returns:
            DataFrame with selected features of shape (n_samples, k_features)
            where k_features <= n_features
        """
        pass
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the selector and transform the data in one step.
        
        Args:
            X: Input features DataFrame of shape (n_samples, n_features)
            y: Target variable Series of shape (n_samples,)
            
        Returns:
            DataFrame with selected features of shape (n_samples, k_features)
        """
        return self.fit(X, y).transform(X)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the selected features.
        
        Returns:
            List of selected feature names
            
        Raises:
            ValueError: If the selector hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted before getting feature names")
        return self.selected_features_
    
    def get_feature_scores(self) -> Optional[np.ndarray]:
        """
        Get the importance scores of all features.
        
        Returns:
            Array of feature importance scores, or None if not available
            
        Raises:
            ValueError: If the selector hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted before getting feature scores")
        return self.feature_scores_
    
    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]:
        """
        Get a mask or indices of the selected features.
        
        Args:
            indices: If True, return feature indices. If False, return boolean mask.
            
        Returns:
            Boolean mask or indices of selected features
            
        Raises:
            ValueError: If the selector hasn't been fitted yet
        """
        if not self.is_fitted_:
            raise ValueError("Feature selector must be fitted before getting support")
            
        if indices:
            return [i for i, feature in enumerate(self.original_features_) 
                   if feature in self.selected_features_]
        else:
            return np.array([feature in self.selected_features_ 
                           for feature in self.original_features_])
    
    def _validate_input(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Validate input data format and consistency.
        
        Args:
            X: Input features DataFrame
            y: Target variable Series (optional)
            
        Returns:
            Validated X and y
            
        Raises:
            ValueError: If input data is invalid
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if X.empty:
            raise ValueError("X cannot be empty")
        
        if y is not None:
            if not isinstance(y, pd.Series):
                raise ValueError("y must be a pandas Series")
            
            if len(X) != len(y):
                raise ValueError(f"X and y must have the same length. Got {len(X)} and {len(y)}")
        
        return X, y
    
    def _log_selection_results(self, original_features: int, selected_features: int):
        """
        Log the results of feature selection.
        
        Args:
            original_features: Number of original features
            selected_features: Number of selected features
        """
        if self.verbose:
            self.logger.info(f"Feature selection completed: {selected_features}/{original_features} features selected")
            if hasattr(self, 'feature_scores_') and self.feature_scores_ is not None:
                top_features = np.argsort(self.feature_scores_)[-5:][::-1]
                self.logger.info(f"Top 5 features by importance: {[self.original_features_[i] for i in top_features]}")
