from .feature_selection import FeatureSelector
from .random_forest_selector import RandomForestSelector

# XGBoost import with optional dependency handling
try:
    from .xgboost_selector import XGBoostSelector
    __all__ = [
        "FeatureSelector",
        "RandomForestSelector", 
        "XGBoostSelector"
    ]
except ImportError:
    # XGBoost not available, only export RF selector
    __all__ = [
        "FeatureSelector",
        "RandomForestSelector"
    ]
    XGBoostSelector = None

__version__ = "1.0.0"
