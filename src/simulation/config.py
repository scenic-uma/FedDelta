import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Union, Optional
from pathlib import Path
import yaml

# ---------------------------------------------------------------------------
# Base directories
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"
# Override with FEDDELTA_RESULTS_DIR (used by the experiments/ scripts)
RESULTS_DIR = Path(os.environ.get("FEDDELTA_RESULTS_DIR", ROOT_DIR / "results" / "default"))

"""Create required directories if they do not exist. Call explicitly at startup."""
for directory in [DATASET_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class ExperimentConfig:
    """
    Parameters required to run a single federated learning experiment.

    Covers dataset behaviour, federated setup, sampling rules, feature
    configuration, gossip-based communication settings, and optional
    centralised training parameters.
    """

    # Experiment identity
    dataset_name:        str
    variable_target:     Union[int, str, List[Union[int, str]]]

    # Local training             
    node_epochs:         int          
    node_batch_size:     int
    nodes_heterogeneity: str

    # General options
    orientation_plots:   str # "horizontal" | "vertical"
    node_failure_rate:   float  
    failure_interval:    int   
    train_ratio:         float = 0.70
    val_ratio:           float = 0.15
    
    # Federated parameters
    num_nodes:           int  = 15            
    num_rounds:          int  = 30 
    num_runs:            int  = 1        
    quorum:              int  = 6 

    # Per-node sampling
    node_min_samples:    int  = 250       
    node_max_samples:    int  = 450       
    min_samples_train:   int  = 200      

    # Features
    max_features:        int  = 15  
    feature_selection:   bool = False    

    # Gossip
    gossip_fanout:       int  = 3          
    gossip_rounds:       int  = 3    
    max_skipped_rounds:  int = 3        

    # Preprocessing / reproducibility
    scale_target:        bool  = True
    target_scale:        float = 1.0        # set at load time (y std)
    seed:                int   = 42
    resplit_per_run:     bool  = True   # new train/val/test split per run (node partition fixed)

    # Fault tolerance: leader crash after collecting updates (round aborted)
    leader_failure_rate: float = 0.0
    leader_downtime:     int   = 2          # rounds a crashed node stays down

    # Data partition across nodes: natural (groups, e.g. hospital) | random (IID) | dirichlet
    partition:           str   = "natural"
    dirichlet_alpha:     float = 0.5
    # True: train/val/test split over the whole pool, independent of the partition (same
    # train and test samples for every partition); False: 70/15/15 split of each node
    global_split:        bool  = False
    # Samples per node and round: fixed (node_min/max_samples) | stream (node data / num_rounds)
    samples_mode:        str   = "fixed"
    # stream mode: relative jitter of the samples per round (0.2 = ±20%). 0 -> exactly
    # ceil(len(train)/num_rounds) per round, so every node delivers all its training data
    stream_jitter:       float = 0.2
    # stream mode with jitter 0: each node splits its training data into equal batches of at
    # least this size, one per round (at most num_rounds batches; a node with fewer samples
    # sends them all at once). Avoids tiny final batches.
    stream_min_batch:    int   = 1
    # Leader election: gossip (default) | random | fixed (always the first available node)
    election:            str   = "gossip"
    # Robustness conditions when node_failure_rate > 0: all (full/partial/dropout) | dropout
    robustness_conditions: str = "all"

    # Statistics
    stats_reference:     str   = "FedDelta"
    stats_alpha:         float = 0.05
    equivalence_margin_r2: float = 0.01

    # Hyperparameter tuning (keys ending in _grid). False -> use YAML values as they are
    tune_hyperparameters: bool = True
    tuning_rounds:       Optional[int] = None   # default: num_rounds
    tuned_params:        Dict[str, Dict] = field(default_factory=dict)

    # Algorithms
    algs_config:         Dict[str, Dict[str, Any]] = field(default_factory = dict)

    # Output
    verbose:             bool = False            
    store_plots:         bool = False
    metrics:             List[str] = field(default_factory=lambda: ["mse", "r2", "exec_time"])
    
def load_global_config() -> dict:
    """
    Load the global YAML configuration file used by all experiments.
    """
    # Override with: FEDDELTA_CONFIG=my_config.yaml python experiments/exp_default.py
    path = Path(os.environ.get("FEDDELTA_CONFIG", ROOT_DIR / "config_experiment.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)