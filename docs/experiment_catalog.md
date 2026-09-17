# Experiment catalog

The default study comprises Adaptime's 90 dataset/frequency/term tasks with
Chronos-2. Every task compares the same candidate family and official test grid.

| Output method | Question |
|---|---|
| `vanilla_univariate` | How does independent forecasting perform? |
| `vanilla_multivariate` | Does jointly supplying target histories help? |
| `self_augmentation` | Do past-only square-root magnitude and sign channels help? |
| `top_k_1`, `top_k_5`, `top_k_10`, `top_k_15`, `top_k_20` | Do normalized-lookback neighbors, supplied with observed futures, help? |
| `selected_task` | Does validation select a useful input choice for the entire task? |
| `selected_per_variate` | Does independent selection improve variate-specific choices? |
| `scope_selector` | Does a binary task-level bootstrap choice improve vanilla scope? |
| `scope_mix` | Does a frozen validation mixture of vanilla scopes help? |
| `top5_mix` | Does a soft univariate/top-5 covariate mixture help? |
| `top1-horizon-mix` | Does mixing vanilla with a scaled historical neighbor continuation help? |

`src/scripts/run_selectime.py` runs the full pipeline or a named stage.
`src/scripts/select_per_task.py` and `src/scripts/select_per_variate.py`
independently consume the same completed validation predictions.
`scripts/submit_experiment.sh` submits Chronos-2. `scripts/submit_ts_icl.sh`
submits TS-ICL without multivariate or scope controls. `scripts/submit_chronos_bolt.sh`
submits only vanilla versus horizon mix. Each accepts `dgx|selena` and Hydra
overrides and composes the same sequential allocation.

Chronos-2 reports 14 methods, TS-ICL 11, and Chronos-Bolt two per task. The
intermediate direct horizon is excluded from the original selectors and report.
Mixture weights are fitted once per task on validation, excluding alternative
fallback rows, with the Beta(1,1) half-tie window-MSSE win rule. No test loss
enters calibration.

The retained retrieval ablation is `datastore_scope=same_series`; the default
`all` includes other variates/items within the same dataset/frequency task.
An optional datastore cap is explicit and disabled by default. K prefixes share
complete maximum-K support, so changing K does not change retrieval eligibility.

The first K<=15 Chronos-2 comparison is analyzed in the results recap.
The expanded K=20 grid, binary controls, mixtures and other backbones remain
unevaluated until complete current-contract artifacts are analyzed. Prior Adaptime outputs are not
Selectime evidence and are not imported.
