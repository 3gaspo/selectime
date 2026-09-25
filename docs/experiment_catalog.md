# Experiment catalog

The default study comprises Adaptime's 90 dataset/frequency/term tasks with
Chronos-2. Every task compares the same candidate family and official test grid.

| Output method | Question |
|---|---|
| `vanilla_univariate` | How does independent forecasting perform? |
| `vanilla_multivariate` | Does jointly supplying target histories help? |
| `scope_selector` | Does one task-level bootstrap choice between univariate and multivariate help? |
| `scope_selector_per_variate` | Does making that hard choice independently per item/variate help? |
| `scope_mix` | Does one task-level win-frequency mixture of vanilla scopes help? |
| `scope_mix_per_variate` | Do independent item/variate scope-mixture weights help? |
| `scope_ridge` | Does a closed-form MSSE-weighted scope coefficient with fixed `alpha=1` help? |
| `top_k_1_mix`, `top_k_5_mix`, `top_k_10_mix`, `top_k_15_mix`, `top_k_20_mix` | Does mixing univariate with a normalized-lookback retrieved-covariate forecast help? |

`src/scripts/run_selectime.py` runs the full pipeline or a named stage.
`scripts/submit_experiment.sh` submits Chronos-2. `scripts/submit_ts_icl.sh`
submits TS-ICL without multivariate or scope controls. `scripts/submit_chronos_bolt.sh`
submits vanilla only. Each accepts `dgx|selena` and Hydra
overrides and composes the same sequential allocation.

Chronos-2 reports 12 methods, TS-ICL six, and Chronos-Bolt one per task. Raw
top-K predictions are intermediate mixture inputs. Win-frequency weights are
fitted once per task or item/variate on validation, excluding alternative
fallback rows, with the Beta(1,1) half-tie window-MSSE rule. `scope_ridge`
fits its one correction coefficient on those same aligned validation windows
with fixed `alpha=1`; no support records an unfitted vanilla fallback. No test
loss enters calibration.

The retained retrieval ablation is `datastore_scope=same_series`; the default
`all` includes other variates/items within the same dataset/frequency task.
An optional datastore cap is explicit and disabled by default. K prefixes share
complete maximum-K support, so changing K does not change retrieval eligibility.

The first K<=15 Chronos-2 comparison is analyzed in the results recap.
The revised K=20 mixtures, scope controls and other backbones remain
unevaluated until complete current-contract artifacts are analyzed. Prior Adaptime outputs are not
Selectime evidence and are not imported.
