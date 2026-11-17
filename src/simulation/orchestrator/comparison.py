
from pathlib import Path
from src.simulation.config import ExperimentConfig
from src.simulation.settings.plots.hyperparameters import plot_hyperparameter_comparison

# =============================================================================
# Comparison plots
# =============================================================================
def export_lambda_comparison(
    all_histories: list,
    base_dir: Path,
    dataset_name: str,
    variable_target: str,
    config: ExperimentConfig,
    parameter: str = "lambda_reg",
) -> None:
    """
    Plot and save lambda sweep comparison across runs.
    """
    lambda_stem = f"{dataset_name}_{variable_target}_lambdas"
    lambda_dir  = (
        base_dir / "comparison_lambdas" / lambda_stem
    )
    if config.store_plots:
        lambda_dir.parent.mkdir(parents=True, exist_ok=True)

    plot_hyperparameter_comparison(
        all_histories,
        parameter   = parameter,
        store_plots = config.store_plots,
        store_path  = lambda_dir if config.store_plots else None,
        show_std    = False,
        store_csv   = True,
    )