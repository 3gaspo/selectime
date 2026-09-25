# Results recap

The expanded comparison covers 90 matched TIME tasks with seed 0. Chronos-2
reports 14 methods: eight input candidates, the task and per-variate selectors,
two scope controls, the top-5 mixture, and the direct-horizon mixture.
Chronos-Bolt reports univariate forecasts and the direct-horizon mixture. Each
method has finite MASE on the same 67,502 Seasonal-defined metric cells.

| Method | Mean task MASE | Change vs Chronos-2 univariate |
|---|---:|---:|
| Chronos-2 scope mixture | 1.058165 | -1.03% |
| Chronos-2 multivariate | 1.058981 | -0.96% |
| Chronos-2 per-variate selector | 1.060312 | -0.83% |
| Chronos-2 task selector | 1.063977 | -0.49% |
| Chronos-2 top-5 mixture | 1.064982 | -0.40% |
| Chronos-2 scope selector | 1.065183 | -0.38% |
| Chronos-2 univariate | 1.069230 | 0.00% |
| Chronos-2 self-augmentation | 1.071291 | +0.19% |
| Chronos-2 retrieved K=1 | 1.078047 | +0.82% |
| Chronos-2 retrieved K=5 | 1.082413 | +1.23% |
| Chronos-2 retrieved K=20 | 1.086914 | +1.65% |
| Chronos-2 retrieved K=10 | 1.087506 | +1.71% |
| Chronos-2 retrieved K=15 | 1.089185 | +1.87% |
| Chronos-Bolt univariate | 1.167905 | +9.23% |

The frozen scope mixture has the lowest raw and Seasonal-scaled mean, but it is
only 0.08% below native multivariate Chronos-2. Task effects are heterogeneous,
one seed establishes no statistical significance, and the small aggregate
margin does not establish a practically meaningful difference. Task and
per-variate selection remain modest improvements over univariate input. The
standalone retrieved-covariate candidates are all worse than univariate, while
their summed recorded query-loop time increases from 433 seconds at K=1 to
5,279 seconds at K=20.

The task selector keeps univariate input on 39 tasks and multivariate input on
31; self-augmentation is selected on nine tasks and retrieval on eleven. The
3,882 per-variate decisions use univariate input 51.4% of the time,
multivariate input 22.3%, self-augmentation 11.4%, and retrieval 14.9%.
Standalone retrieved candidates fall back to univariate on 15.25% of metric
cells. Task/per-variate selector fallback rates are 0.009% and 0.652%.

The direct-horizon mixture is numerically unsafe as evaluated. One
SG_Carpark/15T/short task reaches MASE 313,637 for Chronos-2 and 382,465 for
Bolt, dominating the 90-task means of 3,485.98 and 4,250.80. Excluding only
that task gives 1.1471 and 1.2139, which still do not beat the corresponding
vanilla forecasts. The compact synchronization does not contain the raw
neighbor and prediction arrays needed to identify the exact offending row.

TS-ICL produced no completed forecast result. The initial attempt stopped
because `tsicl` was absent; recovery job 3506722 proved that `tsicl==0.2.1`
loads, then stopped on an all-missing 27-step SG_Weather validation history.
The current validation contract does not send all-missing histories to a
backbone, excludes them from validation losses, and keeps univariate input when
no usable validation rows remain. Preparation and retrieval remain reusable,
but validation forecasts, selections and all downstream TS-ICL stages have not
yet run under that contract.

The [executive summary](../latex/executive_summary.pdf) contains the complete
scientific interpretation and plots. The local analysis bundle retains the
full task table, summary table, selection table, outlier table, plots and
evidence ledger under `outputs/analysis/expanded_results_20260919/`.
