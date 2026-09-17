# Results recap

The initial Chronos-2 comparison, with K in {1,5,10,15}, covers 90 TIME tasks, seven Chronos-2 input
candidates and both validation selectors, with seed 0. All 810 evaluations and
the comparison manifest are completed. Every method has finite MASE on the
same 67,502 Seasonal-defined metric cells.

| Method | Arithmetic mean task MASE |
|---|---:|
| Univariate vanilla | 1.069230 |
| Native multivariate vanilla | 1.058981 |
| Self-augmentation | 1.071291 |
| Top-K, K=1 | 1.078040 |
| Top-K, K=5 | 1.082419 |
| Top-K, K=10 | 1.087484 |
| Top-K, K=15 | 1.089183 |
| Task selector | 1.064426 |
| Per-variate selector | 1.060098 |

The task and per-variate selectors reduce this mean by 0.45% and 0.85%
relative to univariate vanilla. They win/lose/tie on 36/15/39 and 55/27/8
tasks, respectively. Neither beats native multivariate vanilla on this mean;
its reduction versus univariate is 0.96%. The mean of task Seasonal-scaled
MASE slightly favors per-variate selection (0.702976 versus 0.703148 for
multivariate), illustrating that the aggregation choice affects the ranking.
No statistical significance or repeat robustness is established.

Task selection chose univariate on 39 tasks, multivariate on 32,
self-augmentation on 10, and retrieval on nine. Every retrieval candidate
used univariate fallback on 15.25% of metric cells. Selected-task and
selected-per-variate fallback rates are 0.009% and 0.621%.

Recorded candidate query-loop time totals are 234.31 seconds for univariate,
245.46 for multivariate, 732.31 for self-augmentation, and 429.44, 1404.40,
2708.87 and 3952.03 for increasing K. Retrieval and datastore preparation are
separate. Selector independent latency is unmeasured. No runtime advantage
is claimed for selection.

Evidence is the completed comparison report and its 810 input manifests.
Lightweight publication contains summaries and population metric-cell
dispersion; raw arrays and the shared Seasonal payload were not independently
inspected. Population dispersion does not estimate across-run uncertainty.
The next evidence is repeat robustness and measured independent selector cost.

The expanded K=20 grid, binary scope controls, soft mixtures and Bolt/TS-ICL
backbones are implemented but unevaluated. The initial maximum-K=15 results
do not establish their accuracy or validate maximum-K=20 eligibility.
