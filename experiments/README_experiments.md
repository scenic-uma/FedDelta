# Experiments

Each `exp_<name>.py` runs one experiment of the paper and writes its tables (and figures, where
applicable) to `results/<name>/`. Algorithm hyperparameters come from `config_experiment.yaml`;
each script only overrides what the experiment varies. The scripts take no command-line
options: all parameters (datasets, runs, rounds, failure levels, etc.) are set at the top of
`main()`.

Finished runs are cached in `results/<name>/raw/`, so a script can be re-run to redo tables and
figures without retraining. Set `FORCE = True` in the script to recompute.

```bash
python experiments/exp_default.py             # one run with config_experiment.yaml
python experiments/exp_exactness.py
python experiments/exp_convergence.py
python experiments/plot_convergence.py        # figure from the exp_convergence.py results
python experiments/exp_heterogeneity.py
python experiments/exp_fault_tolerance.py
python experiments/exp_scalability_clients.py
python experiments/get_dataset_summary.py
```

| Script | What it evaluates | Outputs in `results/<name>/` |
|---|---|---|
| `exp_exactness.py` | FedDelta vs. the exact ridge solution, round by round, on the real datasets and on synthetic data with condition numbers 1e2–1e10 | `table_real`, `table_synthetic` |
| `exp_convergence.py` | Accuracy and convergence of all methods, with offline tuning of the hyperparameters on the global validation set | `table`, `lambda`, per-round metrics in `framework/` |
| `plot_convergence.py` | Convergence figure from the per-round metrics of `exp_convergence.py` (or `exp_default.py`, `SOURCE = "default"`) | `figures/convergence.pdf`, `figures/last_round.csv` |
| `exp_heterogeneity.py` | Random, natural, Dirichlet(1) and Dirichlet(0.1) partitions | `table`, `fig_r2`, `fig_worst` |
| `exp_fault_tolerance.py` | Client dropout and leader crashes | `table` |
| `exp_scalability_clients.py` | Cost per round vs. number of clients K (SPARCS, random split) | `table`, `fig_scalability` (if `SAVE_FIGURES = True`) |
| `get_dataset_summary.py` | Size, dimension and correlation statistics of each dataset after preprocessing | `results/datasets/table` |
| `exp_default.py` | One run of `config_experiment.yaml` (or `FEDDELTA_CONFIG=<file>`) | `results/default/<dataset>/target_<y>/...` |

Tables are saved as `.csv`, figures as `.pdf` with a `.png` preview.

Each experiment folder also contains `configs/` (the configuration of every run), `logs/`, `raw/`
(cache) and `framework/` (per-round outputs of the simulator).

Notes
- Housing has no natural groups: it uses the random partition everywhere, and in
  `exp_heterogeneity.py` only random and Dirichlet.
- The partition is fixed. Each run changes the seed, and with it the samples per round, the
  arrival order, the failures and the leader.
