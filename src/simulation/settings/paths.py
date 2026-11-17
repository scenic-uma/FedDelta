from pathlib import Path
from typing import Optional

# =============================================================================
# Path helpers
# =============================================================================
def _build_paths(base_dir: Path, lambda_reg) -> dict:
    """
    Precompute all output paths for a given a lambda_reg.
    """
    csv_dir = base_dir / "per_round_metrics"
    if lambda_reg is not None:
        csv_dir = csv_dir / f"lambda{lambda_reg}"

    erank_base = (
        base_dir / "erank" / f"lambda{lambda_reg}"
    )

    return {"csv_dir": csv_dir, "erank_base": erank_base}

def erank_paths(erank_base: Path, run_idx: int):
    """
    Return (subspace_csv, drift_csv) paths for a given run.
    """
    sub_csv   = Path(f"{erank_base}_subspace_diag_{run_idx + 1}.csv")
    drift_csv = Path(f"{erank_base}_subspace_node_drift_{run_idx + 1}.csv")
    return sub_csv, drift_csv

def summary_results_paths(base_dir: Path, tag: Optional[str] = None):
    """
    Return (plot_dir, csv_path). `tag` separates (lambda, sample_cfg, mode) combinations.
    """
    stem     = "summary_results"
    root     = base_dir / "summary_results"
    if tag:
        root = root / tag
    plot_dir = root / stem
    csv_path = plot_dir.parent / f"{stem}.csv"
    return plot_dir, csv_path