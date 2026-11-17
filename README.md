# FedDelta

Code of the experiments of the paper on **FedDelta**, a serverless federated learning method for
ridge regression on data that arrive incrementally.

Each node sends the sufficient statistics of its new data, `A_i = X_iᵀX_i` and `B_i = X_iᵀy_i`.
In every round a leader, elected among the nodes with a gossip protocol, adds them to the global
statistics and updates the inverse of the regularized matrix with the Sherman–Morrison–Woodbury
identity, so there is no fixed server. A quorum-based protocol with atomic rounds and exactly-once
contributions handles node and leader failures. The global model equals the centralized ridge
solution on the data committed so far, up to round-off.

The repository runs every node in a single process (a simulation of the federation) and compares
FedDelta with centralized and federated baselines.

## Methods

| Name in the code | Name in the paper | Description |
|---|---|---|
| `FedDelta` | FedDelta | Incremental ridge regression with sufficient statistics and SMW updates |
| `Cent` | Cent | Centralized ridge regression, retrained on all the data received so far |
| `CentRF` | CentRF | Centralized random forest |
| `CentMLP` | CentMLP | Centralized multi-layer perceptron |
| `FedGD (FedAvg)` | FedAvg | Local mini-batch gradient descent + weighted averaging |
| `FedGD (FedProx)` | FedProx | FedAvg with a proximal term (`mu`) |
| `FedSCAFFOLD` | SCAFFOLD | Local updates corrected with control variates |
| `FedGD (FedAvgM)` | FedAvgM | FedAvg with server momentum |
| `FedMLP (FedAvg)` | FedAvg-MLP | FedAvg applied to a multi-layer perceptron |
| `FedGD (FedNova)` | — | Implemented but not reported: with the same number of local updates in every node it reduces to FedAvg |

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/scenic-uma/feddelta.git
cd feddelta
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

## Datasets

The datasets are not included in the repository. Download them and save them as Parquet files
under `dataset/tabular/`, with these names:

| Dataset | Source | File |
|---|---|---|
| Housing | [California Housing Prices (Kaggle)](https://www.kaggle.com/datasets/camnugent/california-housing-prices/data) | `dataset/tabular/housing.parquet` |
| SPARCS | [Hospital Inpatient Discharges (SPARCS De-Identified) 2015, New York State](https://health.data.ny.gov/Health/Hospital-Inpatient-Discharges-SPARCS-De-Identified/82xm-y6g8/about_data) | `dataset/tabular/Hospital_Inpatient_Discharges__SPARCS_De-Identified___2015.parquet` |
| EESE | [European Health Interview Survey for Spain 2020](https://www.sanidad.gob.es/estadEstudios/estadisticas/EncuestaEuropea/Enc_Eur_Salud_en_Esp_2020.htm) | `dataset/tabular/EESEadulto/df_eese_regression.parquet` |
| LOS | [Hospital Length of Stay (Microsoft)](https://microsoft.github.io/r-server-hospital-length-of-stay/input_data.html) | `dataset/tabular/LengthOfStay/df_lengthofstay.parquet` |

Columns used by the loaders (`src/simulation/data/readers.py`):

- **Housing**: target `median_house_value`; split at random into nodes.
- **SPARCS**: target `Total Charges`; one node per hospital (`Facility Id`), using the largest hospitals.
- **EESE**: target `G21`; one node per autonomous community (`CCAA`).
- **LOS**: target `lengthofstay`; one node per facility (`facid`).

The `dataset/` and `results/` folders are listed in `.gitignore`.

## Reproducing the experiments

Each script in `experiments/` runs one experiment of the paper. All parameters are set at the top of
its `main()` function, and the common settings (algorithms, hyperparameters, grids) are read from
`config_experiment.yaml`. Run them from the root of the repository:

| Script | Experiment | Main output in `results/` |
|---|---|---|
| `get_dataset_summary.py` | Summary of the datasets (samples, features, nodes, conditioning) | `datasets/table.csv` |
| `exp_exactness.py` | Exactness and numerical stability (RQ1), real and synthetic data | `exactness/table_real.csv`, `exactness/table_synthetic.csv` |
| `exp_convergence.py` | Convergence and accuracy (RQ2), with offline hyperparameter selection | `convergence/table.csv`, `convergence/lambda.csv` |
| `plot_convergence.py` | Convergence figure (run after `exp_convergence.py`) | `convergence/figures/convergence.pdf` |
| `exp_heterogeneity.py` | Random, natural and Dirichlet partitions (RQ3) | `heterogeneity/table.csv` |
| `exp_fault_tolerance.py` | Client dropout and leader crashes (RQ4) | `fault_tolerance/table.csv` |
| `exp_scalability_clients.py` | Cost with 10 to 200 nodes (RQ5) | `scalability_clients/table.csv` |

For example:

```bash
python experiments/exp_exactness.py
python experiments/exp_convergence.py
python experiments/plot_convergence.py
```

Every configuration of an experiment runs in its own process. For each one, the scripts save:

- `results/<experiment>/configs/<tag>.yaml`: the exact configuration used;
- `results/<experiment>/logs/<tag>.log`: the log of the run;
- `results/<experiment>/raw/<tag>.pkl`: the histories of all runs, reused on later calls (set `FORCE = True` in the script to recompute);
- `results/<experiment>/framework/<tag>/`: per-round metrics, summaries and statistical tests.

A single configuration can also be run directly from `config_experiment.yaml`:

```bash
python experiments/exp_default.py
```

## Repository structure

```
config_experiment.yaml       Default configuration (datasets, nodes, rounds, algorithms, grids)
experiments/                 One script per experiment (see above) and shared utilities (common.py)
src/
├── algorithms/
│   ├── base.py              Common interface, gradients and local mini-batch gradient descent
│   ├── centralized/         Cent, CentRF, CentMLP
│   └── federated/
│       ├── models/          FedDelta, FedGD, FedSCAFFOLD, FedMLP
│       └── aggregation/     FedAvg, FedProx, FedNova, FedAvgM
├── federated/
│   ├── coordination/        Gossip leader election
│   ├── nodes/               Nodes, leader, batches and commit of the contributions
│   └── trainer/             Rounds, quorum, evaluation, failure injection and data-integrity checks
├── simulation/
│   ├── data/                Dataset loaders, partition into nodes and train/val/test split
│   ├── core/                One run of an experiment and hyperparameter selection
│   ├── orchestrator/        Loops over configurations and runs
│   ├── settings/            Configuration, metrics, statistics and plots
│   └── runner.py            Entry point used by experiments/exp_default.py
└── feature_selection/       Optional feature selection (feature_selection: True in the YAML)
```

## Licence

MIT. See [LICENCE.md](LICENCE.md).
