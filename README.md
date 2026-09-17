# Selectime

Selectime compares frozen foundation-model input choices on the TIME benchmark and
selects them using a held-out validation period. It performs no parameter
training. The official test windows and Seasonal Naive metric grid remain fixed.

The candidate family is vanilla univariate, native vanilla multivariate,
self-augmentation, and top-K retrieved covariates for `K in {1,5,10,15,20}` with Chronos-2.
TS-ICL retains univariate, augmentation and top-K candidates but excludes native
multivariate and scope controls. Chronos-Bolt compares only vanilla and
`top1-horizon-mix`. Canonical backbone aliases are `chronos2`, `ts_icl`,
and `chronos_bolt`.
Self-augmentation supplies `sqrt(abs(x))` and `sign(x)` as past-only covariates.
Top-K supplies retrieved histories and their already observed futures as past
and future covariates. Backbone candidates use deterministic median point forecasts; mixtures
combine these point forecasts with frozen validation weights.

The two original selectors are implemented: one selects a single candidate per
dataset/frequency/term task; the other selects independently per dataset item
and target variate. Both use the same validation forecasts and Adaptime's paired
moving-date-block-bootstrap one-standard-error rule. They prefer univariate
vanilla, multivariate vanilla, self-augmentation, then smaller K among candidates
within one standard error of the observed best validation score.

## Data and protocol

Provide prepared saved-Arrow TIME data through `TIME_DATASET` and the local
`weights/chronos2/`, `weights/chronos-bolt-base/`, or
`weights/tsicl/tsicl-v1.ckpt` checkpoint through `TIME_WEIGHTS`. Optional `.env` settings
configure these input roots. Prepared Arrow targets are consumed directly;
CSV loading, exclusions and missing-value policies are not reapplied.

`src/timebench/config/datasets.yaml` contains the inherited official TIME
horizons/test lengths and Adaptime's task-specific alignment periods, retrieval
lookbacks, datastore strides and validation strides. The selected configuration
path and applied sections are logged and recorded in manifests. Explicit Hydra
settings override the task protocol. The default scope preserves Adaptime's
90 tasks and excludes `Coastal_T_S/5T`, `current_velocity/20T`, `azure2019_D/5T`,
and `azure2019_I/5T` in every stage and report.

During validation, neighbor futures must finish before the validation period
starts. After selection, the datastore expands to all eligible history before
the official test starts. Previously observed test values remain outside the
datastore, while backbone contexts use the history observed at each query.
Calendar ticks use sampling-step units, including frequency multipliers.
Datastore dates are aligned to each query's phase and follow the task's
frequency-dependent stride. There is no fitting interval.

The validation interval immediately precedes test. Its default length is
`H + (floor(test_length/H)-1)*validation_stride`, retaining Adaptime's planned
validation-date count without reserving adaptation-training dates. Histories
shorter than that requested span truncate the interval at the beginning of the
series. Forecasts use all available past context up to the configured backbone context: 8192 for Chronos-2, 2048 by default
for TS-ICL and Chronos-Bolt. Explicit TS-ICL contexts may extend to 4096. Retrieval
requires the complete configured retrieval lookback.

Lookback distance is Euclidean after separate instance normalization of each
query and neighbor. Search requires at least 80% finite feature overlap and a
complete maximum-K neighbor list for every positive K. Each smaller K consumes
an ordered prefix. Neighbor futures must be finite. After retrieval, both
neighbor pasts and futures are rescaled from neighbor lookback statistics to
query lookback statistics, retaining the query's original units; no future
statistics enter this transformation. Short retrieved pasts are left-padded
with missing values to align with the model context.

`datastore_scope=all` retrieves within the selected dataset/frequency task,
across items and variates. `datastore_scope=same_series` restricts it to the exact
query item/variate. `max_datastore_windows=null` disables the optional cap;
when set, `all` divides the budget evenly across variates and retains recent
aligned dates, whereas `same_series` applies the cap per item/variate.

Selection minimizes per-date mean-variate MSSE (mean squared error divided by
the seasonal squared-difference scale from the observed prefix). Per-variate
selection uses that variate's date losses. All candidates share scored target
steps and date support. With no usable validation dates, selection chooses
univariate vanilla. A single date uses the observed minimum because a block
bootstrap cannot estimate its uncertainty. Retrieval-ineligible and non-finite
candidate cells use the canonical univariate forecast; masks and counts persist.
Canonical univariate forecasts must be finite on required support.

## Binary scope selection and soft mixtures

`scope_selector` applies the same task-level bootstrap rule to only univariate
and multivariate vanilla. `scope_mix` blends those two forecasts. `top5_mix`
blends univariate with future-included top-5 covariate forecasts.
`top1-horizon-mix` blends univariate with the nearest neighbor's directly
retrieved horizon, rescaled to query units using lookback statistics. That
horizon is an observed historical continuation, not a backbone forecast or
covariate input. It requires only one neighbor, independently of maximum-K
covariate eligibility. Its raw `top1_horizon` predictions are intermediate
artifacts, not an extra reported method or selector candidate.

Each soft mixture freezes one scalar weight per task from paired eligible
validation item/variate/window MSSE wins. With `w` wins (ties count one half)
and `n` trials, the alternative weight is `(1+w)/(2+n)`. No usable trials gives
pure univariate vanilla. Test predictions are `(1-p)*vanilla+p*alternative`.
Retrieval/non-finite candidate fallback rows do not estimate mixture weights.
The four controls are reported separately and never enter the two original
selectors. Scope controls exist only for Chronos-2; top-5 mix also exists for
TS-ICL; horizon mix exists for all three backbones.

## Running

The maintainer prepares the declared environment on the execution host. Run
from the project root with `PYTHONPATH=src`. Checkpoints and model execution are
offline. A completed shared or project-owned Seasonal Naive grid is required
before test forecasting and evaluation. Its optional producer is:

```bash
bash scripts/submit_seasonal_naive.sh dgx shared
```

The full Chronos-2 experiment runs all stages in one sequential allocation:

```bash
bash scripts/submit_experiment.sh dgx
bash scripts/submit_ts_icl.sh dgx
bash scripts/submit_chronos_bolt.sh dgx
```

Use `selena` instead of `dgx` for the matching scheduler front. TS-ICL uses
`tsicl==0.2.1` in the execution-host environment; the maintainer prepares it
there before submission. The new launchers compose the same sequential stages.

A narrow remote smoke run uses `SG_Weather/D`, short:

```bash
EXPERIMENT_MODE=test bash scripts/submit_experiment.sh dgx
```

Direct execution and the two separately callable selectors use Hydra:

```bash
PYTHONPATH=src uv run --no-sync python src/scripts/run_selectime.py
PYTHONPATH=src uv run --no-sync python src/scripts/select_per_task.py
PYTHONPATH=src uv run --no-sync python src/scripts/select_per_variate.py
```

The standalone selector scripts require completed preparation and validation
forecasts. Example overrides are `datasets=[SG_Weather/D]`, `terms=[short]`,
`datastore_scope=same_series`, and `max_datastore_windows=10000`.
`validation_length=0` explicitly disables validation and selects univariate.

Stages run in order:
`prepare,extract_validation,predict_validation,select_task,select_per_variate,extract_test,predict_test,assemble,evaluate,report`.
`stage=<name>` runs one stage directly. The cluster launcher accepts the
comma-separated `STAGES` recovery override. Exact completed stages are reused;
interrupted tasks restart from their beginning. Conflict controls are
`TIME_RUN_CONFLICT_POLICY=overwrite_exact|overwrite_path|new`,
`TIME_SKIP_COMPLETED`, and `TIME_FORCE_RERUN`.

Each forecasting stage runs canonical univariate predictions first, finalizes
their manifests after a successful scheduler step, then runs the remaining
candidates. Direct forecasting defaults to `prediction_group=all`; the scheduler
uses `vanilla` and `remaining` groups to respect producer completion.

Reports default to the current exact scientific configuration and its selected
repeat. `report_config_policy=error|distinct|latest|average` and
`report_repeat_policy=selected|latest|distinct|average` expose the inherited
selection controls. Set `report_current_config=false` to include other completed
configurations, optionally filtered with dotted keys in `report_config_filters`.
Configuration averaging first averages exact repeats within each configuration,
then equally weights the configurations. It averages declared task statistics,
rather than pooling metric cells. Distinct labels include differing nested
dependency settings. The report manifest records policies and every input;
selection frequencies describe choices observed in those selected inputs.
Producer dependencies always require the exact expected configuration.

Run-pinning and interruption recovery remain available:

```bash
PYTHONPATH=src uv run --no-sync python src/scripts/select_result_run.py /path/to/run_n
PYTHONPATH=src uv run --no-sync python src/scripts/interrupt_result_launch.py outputs/selectime --launch-id <launch-id>
```

## Source and artifacts

```text
src/timebench/data/           Arrow window readers, intervals and aligned indices
src/timebench/model_loading/  native Chronos-2, Chronos-Bolt and TS-ICL adapters
src/timebench/proposal/       retrieval, covariate transformations, bootstrap selection and validation mixtures
src/timebench/pipeline/       stage orchestration, manifests and run recovery
src/timebench/evaluation/     narrowed TIME adapter, metrics, shared grid and saver
src/timebench/results/        comparison and selection summaries
src/timebench/conf/           Hydra experiment configuration
src/timebench/config/         official TIME grid and per-task retrieval protocol
src/scripts/                 pipeline, task/per-variate selector and Seasonal entry points
src/slurm/                   common scheduler workflow and runtime implementations
src/tests/                   lightweight scientific regression contracts
scripts/                     concise experiment and Seasonal submission launchers
*.slurm                      root-level scheduler fronts
```

Artifacts live exclusively in the owning project's `outputs/selectime/<backbone>/` on
each execution surface:
`data/shared/`, `retrieval/{validation,test}/`,
`predictions/{validation,test}/<candidate>/`, `selections/{task,per_variate}/`,
`evaluations/<method>/`, and `reports/comparison/`, each followed by
`<dataset>/<frequency>/<term>/run_n/`. Every stage uses schema-1 plain-configuration
manifests. Raw predictions are float32 `.npy` files; evaluations retain standard
TIME files with mean, population variance, standard deviation and finite/grid
counts. Reports include matched Seasonal scaling, fallback rates, validation
choices and their selection frequencies. Runtime logs belong in `logs/`.

Candidate forecasting, retrieval and datastore preprocessing timings are
reported separately. Selected test predictions reuse candidate artifacts;
their independent inference latency is unmeasured and remains undefined.
The two selection results therefore do not claim a summed candidate latency.

`sync_code_to_selena.sh`, `sync_results_to_dgx.sh`, `clear_selena_artifacts.sh`,
and `publish_job.sh` retain project-scoped cluster operations. Lightweight result
transfer includes reports and compact metadata; detailed transfer adds metric
arrays and masks; full transfer includes numeric prediction/retrieval payloads.

## Research documentation

- [Architecture](docs/architecture.md)
- [Experiment catalog](docs/experiment_catalog.md)
- [Method overview source](latex/method_overview.tex)
- [Method overview PDF](latex/method_overview.pdf)
- [Experiment guideline source](latex/experiment_guideline.tex)
- [Experiment guideline PDF](latex/experiment_guideline.pdf)
- [Evidence status PDF](latex/executive_summary.pdf)
- [Results recap](docs/results_recap.md)

Selectime inherits maintained TIME utilities from Improved TIME and adapts
retrieval and validation-selection ideas from Adaptime. It includes no Ridge,
adaptation-training mixture fitting, rolling fitting, TS-RAG ARM, or standalone foundation
benchmarking workflow. The first analyzed comparison used K up to 15. Its results do not establish
performance for the expanded K=20 candidates, controls or new backbones. The inherited code is Apache-2.0 licensed.
