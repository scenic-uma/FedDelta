# Feature Selection Module

This module provides a comprehensive interface for feature selection in the federated learning infrastructure. It implements multiple algorithms to reduce dataset dimensionality by eliminating redundant features while preserving important ones.

## Overview

The feature selection module follows the infrastructure's design patterns and provides:

- **Abstract Interface**: `FeatureSelector` base class defining the contract for all implementations
- **Multiple Algorithms**: Random Forest and XGBoost-based feature selection
- **Comprehensive Testing**: Extensive test suite validating redundant feature elimination and important feature preservation
- **Real-world Validation**: Tested on both synthetic and real datasets

## Core Interface
All feature selectors inherit from the `FeatureSelector` abstract base class:

```python
from src.feature_selection import FeatureSelector

class FeatureSelector(ABC):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FeatureSelector'
    def transform(self, X: pd.DataFrame) -> pd.DataFrame
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame
    def get_feature_names(self) -> List[str]
    def get_feature_scores(self) -> Optional[np.ndarray]
    def get_support(self, indices: bool = False) -> Union[np.ndarray, List[int]]
```

### Key Design Principles

1. **DataFrame-First**: All methods work with pandas DataFrames to preserve feature names
2. **Scikit-learn Compatible**: Follows fit/transform pattern for easy integration
3. **Comprehensive Validation**: Input validation and error handling throughout
4. **Flexible Selection**: Support for both fixed number of features and threshold-based selection
5. **Cross-validation Support**: Stable feature importance computation across CV folds

## Implementations
### RandomForestSelector

Uses Random Forest feature importance to select the most relevant features.

```python
from src.feature_selection import RandomForestSelector

selector = RandomForestSelector(
    n_features=10,              # Number of features to select
    n_estimators=100,           # Number of trees in forest
    use_cv_importance=True,     # Use cross-validation for stability
    cv_folds=5,                 # Number of CV folds
    random_state=42,            # For reproducibility
    verbose=True                # Enable logging
)

X_selected = selector.fit_transform(X, y)
```

**Key Features:**
- Cross-validated importance scores for stability
- Configurable Random Forest parameters
- Handles both fixed number and threshold-based selection
- Performance evaluation capabilities

### XGBoostSelector

Uses XGBoost feature importance with multiple importance types.

```python
from src.feature_selection import XGBoostSelector

selector = XGBoostSelector(
    n_features=10,              # Number of features to select
    importance_type='gain',     # 'gain', 'weight', 'cover', etc.
    n_estimators=100,           # Number of boosting rounds
    use_cv_importance=True,     # Use cross-validation
    early_stopping_rounds=10,   # Early stopping for efficiency
    random_state=42,
    verbose=True
)

X_selected = selector.fit_transform(X, y)
```

**Key Features:**
- Multiple importance types (gain, weight, cover)
- Early stopping support
- Cross-validated importance computation
- Feature interaction analysis capabilities

## Usage Examples

### Basic Usage

```python
import pandas as pd
from sklearn.datasets import load_diabetes
from src.feature_selection import RandomForestSelector

logger = logging.getLogger(__name__)

# Load data
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# Select top 5 features
selector = RandomForestSelector(n_features=5, random_state=42)
X_selected = selector.fit_transform(X, y)

logger.info(f"Selected features: {selector.get_feature_names()}")
logger.info(f"Original shape: {X.shape}, Selected shape: {X_selected.shape}")
```

### Advanced Usage with Performance Evaluation

```python
# Create selector with custom parameters
selector = RandomForestSelector(
    n_features=6,
    n_estimators=200,
    max_depth=10,
    use_cv_importance=True,
    cv_folds=10,
    verbose=True
)

# Fit and transform
X_selected = selector.fit_transform(X, y)

# Get detailed information
feature_names = selector.get_feature_names()
feature_scores = selector.get_feature_scores()
support_mask = selector.get_support()

# Evaluate performance
performance = selector.get_model_performance(X, y)
logger.info(f"CV R²: {performance['cv_r2_mean']:.3f} ± {performance['cv_r2_std']:.3f}")
logger.info(f"Feature reduction: {performance['feature_reduction_ratio']:.1%}")
```

### Threshold-based Selection

```python
# Select features above importance threshold
selector = RandomForestSelector(
    importance_threshold=0.05,  # Select features with importance > 0.05
    random_state=42
)

X_selected = selector.fit_transform(X, y)
logger.info(f"Selected {len(selector.get_feature_names())} features above threshold")
```

## Testing

The module includes comprehensive tests covering:

### Test Categories

1. **Basic Functionality Tests**
   - Initialization and parameter validation
   - Fit/transform workflow
   - Error handling and edge cases

2. **Redundant Feature Elimination Tests**
   - Synthetic datasets with known redundant features
   - Verification that redundant features are eliminated
   - Preservation of informative features

3. **Real Dataset Tests**
   - Performance on diabetes dataset
   - Cross-validation stability
   - Feature importance ranking validation

4. **Algorithm Comparison Tests**
   - Consistency between Random Forest and XGBoost
   - Feature overlap analysis
   - Performance comparison

### Running Tests

```bash
# Run all feature selection tests
python -m pytest tests/src/feature_selection/test_feature_selection.py -v

# Run specific test categories
python -m pytest tests/src/feature_selection/test_feature_selection.py::TestRandomForestSelector -v
python -m pytest tests/src/feature_selection/test_feature_selection.py::TestXGBoostSelector -v
```

## Validation Results

The feature selection implementations have been validated to:

### ✅ Eliminate Redundant Features
- Successfully identifies and removes features that are linear combinations of others
- Eliminates highly correlated features (r > 0.95)
- Reduces noise features while preserving signal

### ✅ Preserve Important Features
- Maintains features that contribute significantly to target prediction
- Preserves at least 80% of original model performance
- Selects features with above-average importance scores

### ✅ Handle Real-world Data
- Tested on diabetes dataset (442 samples, 10 features)
- Achieves comparable or better performance with fewer features
- Stable feature selection across cross-validation folds

### ✅ Algorithm Consistency
- Random Forest and XGBoost show reasonable feature overlap (>10%)
- Both algorithms successfully identify informative features
- Performance maintained across different selection strategies

## Dependencies

### Required
- pandas >= 1.3.0
- numpy >= 1.20.0
- scikit-learn >= 1.0.0

### Optional
- xgboost >= 1.5.0 (for XGBoostSelector)

## Integration with Federated Learning

The feature selection module integrates seamlessly with the federated learning infrastructure:

1. **Preprocessing Step**: Apply feature selection before federated training
2. **Consistent Features**: Ensure all nodes use the same selected features
3. **Reduced Communication**: Fewer features mean smaller model updates
4. **Improved Performance**: Better signal-to-noise ratio in federated models

### Example Integration

```python
# Apply feature selection before federated training
selector = RandomForestSelector(n_features=10, random_state=42)
X_selected = selector.fit_transform(X_train, y_train)

# Use selected features in federated algorithm
selected_features = selector.get_feature_names()
X_federated = X_full[selected_features]

# Proceed with federated training using X_federated
```

## Performance Characteristics

### Computational Complexity
- **Random Forest**: O(n_trees × n_samples × log(n_samples) × n_features)
- **XGBoost**: O(n_rounds × n_samples × n_features)
- **Memory**: Linear in dataset size, efficient for large datasets

### Scalability
- Supports datasets with thousands of features
- Cross-validation provides stability for small datasets
- Efficient implementation using scikit-learn and XGBoost optimizations

## Future Enhancements

Potential areas for extension:

1. **Additional Algorithms**: LASSO, Mutual Information, Recursive Feature Elimination
2. **Ensemble Selection**: Combine multiple feature selection methods
3. **Streaming Selection**: Online feature selection for streaming data
4. **Federated Selection**: Distributed feature selection across federated nodes
5. **Interpretability**: Enhanced feature importance visualization and analysis

## Contributing

When adding new feature selection algorithms:

1. Inherit from `FeatureSelector` base class
2. Implement all abstract methods
3. Add comprehensive tests following existing patterns
4. Update documentation and examples
5. Ensure compatibility with pandas DataFrames
6. Follow the project's logging and error handling conventions
