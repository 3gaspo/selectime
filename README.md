# Selectime

Selectime compares frozen foundation-model input choices on the TIME benchmark and
selects them using a held-out validation period. It performs no parameter
training. The official test windows and Seasonal Naive metric grid remain fixed.

Chronos-2 reports univariate and native multivariate baselines, hard scope
selection, validation-weighted scope mixtures, a fixed-alpha scope ridge, and
one univariate/top-K mixture for each `K in {1,5,10,15,20}`. Raw top-K forecasts
are intermediate mixture inputs, not standalone reported methods. TS-ICL keeps
the top-K mixtures but has no native multivariate or scope methods. Chronos-Bolt
reports only vanilla. Canonical backbone aliases are `chronos2`, `ts_icl`, and
`chronos_bolt`.

Top-K supplies retrieved histories and their already observed futures as past
and future covariates. Backbone forecasts are deterministic medians. Scope
selection chooses univariate or multivariate either once per task or once per
item/variate using Adaptime's paired moving-date-block-bootstrap
one-standard-error rule.

## Data and protocol

Provide prepared saved-Arrow TIME data through `TIME_DATASET` and the local
`weights/chronos2/`, `weights/chronos-bolt-base/`, or
`weights/tsicl/tsicl-v1.ckpt` checkpoint through `TIME_WEIGHTS`. Optional `.env` settings
configure these input roots. Prepared Arrow targets are consumed directly;
CSV loading, exclusions and missing-value policies are not reapplied.

`src/timebench/config/datasets.yaml` contains the inherited official TIME
horizons/test lengths and Adaptime's task-specific alignment periods, retrieval
lookbacks and datastore strides. Validation always uses the official horizon
`H` as its stride. The selected configuration
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
frequency-dependent stride. There is no separate fitting interval: scope ridge
fits only on the same pre-test validation windows used by every other control.

Validation dates step backward by `H` from the first official test date, with
the same phase as the test dates and at most as many dates as the test grid.
`validation_length` limits that count and defaults to `test_length`; histories
that do not reach all requested dates report fewer available dates. Forecasts
use all available past context up to the configured backbone
context: 8192 for Chronos-2, 2048 by default for TS-ICL and Chronos-Bolt.
Explicit TS-ICL contexts may extend to 4096. Retrieval requires the complete
configured retrieval lookback.

Validation rows are usable only when both the available target history and the
forecast future contain a finite value. Unusable rows are not sent to any
backbone and contribute no selection or mixture loss. Requested, available and
usable date/row counts are recorded. When none are
usable, selectors and heuristic mixtures keep the canonical univariate default;
scope ridge records an unfitted vanilla fallback. This filtering is
validation-only and does not change the official Seasonal-defined test support.

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

Scope selection minimizes per-date mean-variate MSSE (mean squared error divided
by the seasonal squared-difference scale from the observed prefix). The
per-variate selector uses that variate's date losses. A single date uses the
observed minimum because a block bootstrap cannot estimate uncertainty.
Retrieval-ineligible and non-finite candidate cells use the canonical
univariate forecast; masks and counts persist. Canonical univariate forecasts
must be finite on required support.

## Scope controls and top-K mixtures

`scope_selector` chooses one vanilla scope for a complete task;
`scope_selector_per_variate` makes the same hard choice independently per
item/variate. `scope_mix` and `scope_mix_per_variate` blend the two scopes at
those respective granularities. Each `top_k_<K>_mix` blends univariate with its
future-included top-K covariate forecast.

Heuristic mixtures freeze weights from paired eligible validation-window MSSE
wins. With `w` wins (ties count one half) and `n` trials, the alternative weight
is `(1+w)/(2+n)`. No usable trials gives pure univariate vanilla. The task-level
`scope_ridge` instead fits
`univariate + p*(multivariate-univariate)` by an MSSE-weighted closed form with
`scope_ridge_alpha=1.0`; `p` is not clipped. With no usable fitting row it
records an unfitted `p=0` vanilla fallback. Retrieval fallback rows never fit a
mixture. Scope controls exist only for Chronos-2; top-K mixtures also exist for
TS-ICL.

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

Direct execution uses Hydra:

```bash
PYTHONPATH=src uv run --no-sync python src/scripts/run_selectime.py
```

Example overrides are `datasets=[SG_Weather/D]`, `terms=[short]`,
`datastore_scope=same_series`, and `max_datastore_windows=10000`.
`validation_length=0` explicitly disables validation and selects univariate.

Stages run in order:
`prepare,extract_validation,predict_validation,calibrate,extract_test,predict_test,assemble,evaluate,report`.
`stage=<name>` runs one stage directly. The cluster launcher accepts the
comma-separated `STAGES` recovery override. Exact completed stages are reused;
interrupted tasks restart from their beginning. A task whose artifacts were
fully written before a later scheduler failure remains `computed`; a recovery
launch finalizes that same task without recomputing it. Reports and downstream
stages still accept only `completed` producers. Conflict controls are
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
src/scripts/                 pipeline, lifecycle and Seasonal entry points
src/slurm/                   common scheduler workflow and runtime implementations
src/tests/                   lightweight scientific regression contracts
scripts/                     concise experiment and Seasonal submission launchers
*.slurm                      root-level scheduler fronts
```

Artifacts live exclusively in the owning project's `outputs/selectime/<backbone>/` on
each execution surface. Selena uses
`/scratch/users/<nni>/codes/selectime/outputs/`; DGX/local execution uses the
checkout's `outputs/`. Those are defaults; explicit `OUTPUTS_ROOT` and
`LOGS_ROOT` values take precedence. The shared Seasonal producer deliberately
uses the common Seasonal artifact root and its `logs/` child, while Selectime
consumes that grid through `TIME_SEASONAL_TASKS_ROOT`.
The layout is:
`data/{validation,test}/shared/`,
`retrieval/{validation,test}/covariate/`,
`predictions/{validation,test}/<candidate>/`, `selections/<control>/`,
and `evaluations/<method>/`, each followed by
`<dataset>/<frequency>/<term>/run_n/`. Every stage uses schema-1 plain-configuration
manifests. Raw predictions are float32 `.npy` files; evaluations retain standard
TIME files with mean, population variance, standard deviation and finite/grid
counts. Reports live separately under `outputs/reports/selectime/`. Validation
and test caches have independent stage identities; unchanged univariate rows
are copied by `(item, channel, origin)` from equivalent completed runs, and
manifests record reused versus newly inferred rows. Reports include matched Seasonal scaling, fallback rates, validation
choices and their selection frequencies. Runtime logs belong in `logs/`.

Candidate forecasting, retrieval and datastore preprocessing timings are
reported separately. Assembled selector and mixture outputs reuse candidate
artifacts; their independent inference latency is unmeasured and remains
undefined.

`sync_code_to_selena.sh`, `sync_results_to_dgx.sh`, `clear_selena_artifacts.sh`,
and `publish_job.sh` retain project-scoped cluster operations. Lightweight result
transfer includes reports and compact metadata; detailed transfer adds metric
arrays and masks; full transfer includes numeric prediction/retrieval payloads.
Each allocation logs visible accelerators, GPU and host memory, and explicit
cgroup availability before scientific stages. Every learned or CPU-only stage
also records the device it actually selected.

## Research documentation

- [Architecture](docs/architecture.md)
- [Experiment catalog](docs/experiment_catalog.md)
- [Method overview source](latex/method_overview.tex)
- [Method overview PDF](latex/method_overview.pdf)
- [Experiment guideline source](latex/experiment_guideline.tex)
- [Experiment guideline PDF](latex/experiment_guideline.pdf)
- [Expanded comparison executive summary](latex/executive_summary.pdf)
- [Results recap](docs/results_recap.md)

Selectime inherits maintained TIME utilities from Improved TIME and adapts
retrieval and validation-selection ideas from Adaptime. Ridge appears only as
the closed-form fixed-alpha `scope_ridge`; the project includes no adaptation
training, rolling fitting, TS-RAG ARM, or standalone foundation
benchmarking workflow. The completed expanded comparison covers all 90 tasks for
Chronos-2 and Chronos-Bolt. The Chronos-2 scope mixture has the lowest mean task
MASE (1.058165), 1.03% below univariate Chronos-2 and 0.08% below native
multivariate input. Historical standalone retrieved-covariate candidates did
not improve the aggregate mean and are no longer reported directly. TS-ICL has
no forecast result: after its environment was
repaired, the historical recovery stopped on an all-missing early validation
history. The current validation contract excludes such rows and therefore
requires fresh validation predictions and downstream selection. See the results recap and executive
summary for the complete evidence boundaries. The inherited code is Apache-2.0 licensed.
