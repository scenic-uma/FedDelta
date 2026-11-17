from colorama import init
from .io import lazy_load_class

import numpy as np
import logging
import inspect

logger = logging.getLogger(__name__)

init(autoreset=True)

# Mapping algorithm names to their corresponding implementation classes
ALGORITHM_IMPORT_PATHS = {
    "FedGD": "src.algorithms.federated.models.FedGD:FedGD",
    "FedDelta": "src.algorithms.federated.models.FedDelta:FedDelta",
    "FedSCAFFOLD": "src.algorithms.federated.models.FedSCAFFOLD:FedSCAFFOLD",
    "FedMLP": "src.algorithms.federated.models.FedMLP:FedMLP",
    "Cent": "src.algorithms.centralized.Cent:Cent",
    "CentRF": "src.algorithms.centralized.CentRF:CentRF",
    "CentMLP": "src.algorithms.centralized.CentMLP:CentMLP",
}

# Mapping algorithm names to their corresponding implementation classes
AGGREGATION_IMPORT_PATHS = {
    "FedAvg": "src.algorithms.federated.aggregation.FedAvg:FedAvg",
    "FedNova": "src.algorithms.federated.aggregation.FedNova:FedNova",
    "FedProx": "src.algorithms.federated.aggregation.FedProx:FedProx",
    "FedAvgM": "src.algorithms.federated.aggregation.FedOpt:FedAvgM",
    "FedAdagrad": "src.algorithms.federated.aggregation.FedOpt:FedAdagrad",
    "FedAdam": "src.algorithms.federated.aggregation.FedOpt:FedAdam",
    "FedYogi": "src.algorithms.federated.aggregation.FedOpt:FedYogi",
}

def _resolve_aggregation_method(agg_name: str, **server_kwargs):
    """Instantiates an aggregation class from its string name (server_* YAML keys -> kwargs)."""
    load = lazy_load_class(AGGREGATION_IMPORT_PATHS[agg_name])
    cls = load()
    valid = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in server_kwargs.items() if k in valid})

def get_selected_algorithms(algs):
    """
    Return only the algorithms that are enabled (use=True) and whose category matches the given task type.
    """
    return {
        name: params
        for name, params in algs.items()
        if params.get("use", False)  # Keep only enabled algorithms
    }

def build_algorithms_config(alg_params):
    """
    Returns a dictionary mapping algorithm names to:
      {
        "class": lazy_loaded_class,
        "params": filtered_params
      }
    Only includes algorithms enabled.
    """
    filtered_algos = get_selected_algorithms(alg_params)

    result = {}

    for name, params in filtered_algos.items():

        clean_name = name.split()[0]

        if clean_name not in ALGORITHM_IMPORT_PATHS:
            continue

        # Lazy load
        cls_loader = lazy_load_class(ALGORITHM_IMPORT_PATHS[clean_name])

        # Remove fields not needed downstream
        filtered_params = {
            k: v for k, v in params.items()
            if k not in ("use", "category")
        }

        result[name] = {
            "class": cls_loader,
            "params": filtered_params
        }

    return result

def get_algorithm_instance(
    alg_config: dict,
    n_features: int,
    verbose: bool = False,
    n_clients: int = None,
) -> dict:
    """
    Instantiates each algorithm in alg_config with the appropriate arguments.
    - dynamic_args provide runtime values (weights, verbosity, etc.)
    - params from config override dynamic_args, except for protected keys.
    - aggregation_method strings are resolved to class instances before construction.
    """
    # Keys that must always come from dynamic_args, never from YAML params
    PROTECTED_KEYS = {"initial_weights", "verbose"}

    instances = {}

    for name, info in alg_config.items():
        cls    = info["class"]()
        params = dict(info["params"])

        # Resolve aggregation_method string → instance
        if isinstance(params.get("aggregation_method"), str):
            # e.g. server_lr -> lr, server_beta1 -> beta1 (FedOpt family)
            server_kwargs = {k[len("server_"):]: v for k, v in params.items() if k.startswith("server_")}
            params["aggregation_method"] = _resolve_aggregation_method(
                params["aggregation_method"], **server_kwargs
            )

        # Base runtime arguments
        dynamic_args = {
            "verbose":         verbose,
            "n_features":      n_features,
            "initial_weights": np.zeros(n_features + 1),
            **({"n_clients": n_clients} if n_clients else {}),
        }

        # Merge: YAML params take priority, except for protected keys
        merged = {
            **dynamic_args,
            **{k: v for k, v in params.items() if k not in PROTECTED_KEYS},
        }

        # Keep only args accepted by the constructor
        valid_args    = inspect.signature(cls.__init__).parameters.keys()
        filtered_args = {k: v for k, v in merged.items() if k in valid_args}

        instances[name] = cls(**filtered_args)

    return instances
