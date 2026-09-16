# Experiment catalog

The default study comprises Adaptime's 90 dataset/frequency/term tasks with
Chronos-2. Every task compares the same candidate family and official test grid.

| Output method | Question |
|---|---|
| `vanilla_univariate` | How does independent forecasting perform? |
| `vanilla_multivariate` | Does jointly supplying target histories help? |
| `self_augmentation` | Do past-only square-root magnitude and sign channels help? |
| `top_k_1`, `top_k_5`, `top_k_10`, `top_k_15` | Do normalized-lookback neighbors, supplied with observed futures, help? |
| `selected_task` | Does validation select a useful input choice for the entire task? |
| `selected_per_variate` | Does independent selection improve variate-specific choices? |

`src/scripts/run_selectime.py` runs the full pipeline or a named stage.
`src/scripts/select_per_task.py` and `src/scripts/select_per_variate.py`
independently consume the same completed validation predictions.
`scripts/submit_experiment.sh` submits the sequential experiment.

The retained retrieval ablation is `datastore_scope=same_series`; the default
`all` includes other variates/items within the same dataset/frequency task.
An optional datastore cap is explicit and disabled by default. K prefixes share
complete maximum-K support, so changing K does not change retrieval eligibility.

All hypotheses remain untested in Selectime until its current-contract
artifacts have been generated and analyzed. Prior Adaptime outputs are not
Selectime evidence and are not imported.
