from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import our feature selectors
from src.feature_selection.random_forest_selector import RandomForestSelector
try:
    from src.feature_selection.xgboost_selector import XGBoostSelector
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.info("XGBoost not available, will only demonstrate Random Forest selector")

def evaluate_feature_selection(X_original, y, X_selected, selector_name):
    """Evaluate the performance of feature selection."""
    # Split data for evaluation
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(
        X_original, y, test_size=0.2, random_state=42
    )
    X_train_sel, X_test_sel, _, _ = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Train models on original and selected features
    model_orig = LinearRegression()
    model_sel = LinearRegression()
    
    model_orig.fit(X_train_orig, y_train)
    model_sel.fit(X_train_sel, y_train)
    
    # Evaluate performance
    r2_orig = r2_score(y_test, model_orig.predict(X_test_orig))
    r2_sel = r2_score(y_test, model_sel.predict(X_test_sel))
    
    logger.info(f"=== {selector_name} Results ===")
    logger.info(f"Original features: {X_original.shape[1]}")
    logger.info(f"Selected features: {X_selected.shape[1]}")
    logger.info(f"Reduction ratio: {X_selected.shape[1] / X_original.shape[1]:.2%} \n")
    logger.info(f"R² with all features: {r2_orig:.4f}")
    logger.info(f"R² with selected features: {r2_sel:.4f}")
    logger.info(f"Performance retention: {r2_sel / r2_orig:.2%}\n")
   
    return r2_orig, r2_sel

def apply_feature_selection(X, y, n_features):
    """Apply RandomForest and (optionally) XGBoost selectors,
    prints results and returns the selected feature indices per selector.
    """
    
    rf_selector = RandomForestSelector(
        n_features=n_features,
        random_state=42,
        verbose=True,
        use_cv_importance=True
    )
    X_rf_selected = rf_selector.fit_transform(X, y)
    selected_features_rf = rf_selector.get_feature_names()
    
    # Evaluate performance
    evaluate_feature_selection(X, y, X_rf_selected, "Random Forest")
    
    # Feature importance
    feature_scores = rf_selector.get_feature_scores()
    feature_importance_pairs = list(zip(rf_selector.original_features_, feature_scores))
    feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Feature Selection Interface Demonstration")
    logger.info("=" * 50)
    logger.info("\2. Testing Random Forest Feature Selector...")
    logger.info(f"Selected features (RF): {selected_features_rf}\n")
    logger.info("Top 5 most important features:")
    for i, (feature, score) in enumerate(feature_importance_pairs[:5]):
        logger.info(f"  {i+1}. {feature}: {score:.4f}")
    
    # Optionally XGBoost selector
    selected_features_xgb = None
    if XGBOOST_AVAILABLE:
        xgb_selector = XGBoostSelector(
            n_features=n_features,
            importance_type='gain',
            random_state=42,
            verbose=True,
            use_cv_importance=True
        )
        X_xgb_selected = xgb_selector.fit_transform(X, y)
        selected_features_xgb = xgb_selector.get_feature_names()
        evaluate_feature_selection(X, y, X_xgb_selected, "XGBoost")
        
        # Compare selectors
        rf_set = set(selected_features_rf)
        xgb_set = set(selected_features_xgb)
        overlap = rf_set.intersection(xgb_set)
        
        logger.info("3. Testing XGBoost Feature Selector...")
        logger.info(f"Selected features (XGB): {selected_features_xgb}\n")
        logger.info(f"4. Comparing Feature Selectors...")
        logger.info(f"Feature overlap: {len(overlap)}/{n_features} ({len(overlap)/n_features:.1%})")
        logger.info(f"Common features: {sorted(overlap)}")
        logger.info(f"RF only: {sorted(rf_set - xgb_set)}")
        logger.info(f"XGBoost only: {sorted(xgb_set - rf_set)}")

    # Return selected features (RF first, then XGB if available)
    return selected_features_rf, selected_features_xgb
