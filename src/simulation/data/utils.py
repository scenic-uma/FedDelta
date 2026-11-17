from src.feature_selection.feature_selection_utils import apply_feature_selection

import logging
import pandas as pd

logger = logging.getLogger(__name__)

def read_dataset(infra_dir, filename, pickle = False):
    """
    Load a dataset from the given filename and return a DataFrame.
    """
    path = infra_dir / filename
    if pickle:
        return pd.read_pickle(path)
    return pd.read_parquet(path)

def reduce_dimensionality(X, y, max_features, max_samples):
    """
    Checks if a feature selection for 'max_features' is already stored in JSON.
    If so, loads and applies it. Otherwise, performs feature selection, saves
    the selected features, and applies them.
    """
    key = str(max_features)

    data = {}

    if key in data:
        selected_features = data[key]
        logger.info(f"[Dataset] Loaded precomputed feature selection for {max_features} features.")
    else:
        # Use a subset for faster FS if dataset is too large
        if X.shape[0] > max_samples:
            X_init_df, y_init_df = X.iloc[:max_samples], y.iloc[:max_samples]
        else:
            X_init_df, y_init_df = X, y

        # Perform feature selection
        selected_features, _ = apply_feature_selection(X_init_df, y_init_df, max_features)

        # Store in JSON
        data[key] = selected_features

    # Reduce the dataset using selected features
    X_reduced = X[selected_features]
    new_d = len(selected_features)

    return X_reduced, new_d, selected_features