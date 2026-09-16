# Architecture

Selectime is a direct Improved TIME child. Its scientific proposal is the
selection of Chronos-2 inputs, implemented under `src/timebench/proposal/`.
This package contains normalized distance, eligible top-K search, past-only
transformations, neighbor scaling, and the paired bootstrap selector. It has
no dependency on scheduler, Hydra, manifests or reporting.

`data/windows.py` owns the observed-prefix reader and the chronological
datastore/validation/test indices. `model_loading/chronos2.py` owns the native
Chronos-2 tensor/dictionary adapter. `pipeline/workflow.py` composes these
owners and allocates each task before work. The standard stages are:

```text
saved Arrow + official TIME settings + task retrieval settings
  -> prepare indices for validation and official test
  -> validation retrieval -> candidate validation forecasts
  -> independent task and per-variate bootstrap selections
  -> expanded pre-test retrieval -> all candidate test forecasts
  -> assemble both frozen selections from those candidate predictions
  -> inherited TIME metrics on the same Seasonal grid -> comparison report
```

Multivariate forecasting runs once per item/date and is projected into the
canonical item/channel/date order. This makes all candidates and both
selectors comparable on the same metric cells. Univariate candidates read
one target variate and only their declared covariates.

Validation and test retrieval own distinct neighbor artifacts because their
datastore cutoffs differ. Each is searched once at maximum K, and every smaller
K reads a prefix of that list. Datastore and query representations are
memory-mapped. Trajectories remain in Arrow until needed for a forecast; no
datastore backbone forecasts, coefficient vectors, or fitted models exist.

The shared `runs.py` lifecycle retains exact plain-configuration reuse,
task-boundary recovery, manifest histories and explicit conflict controls.
Scheduler stages finalize ready artifacts only after their owning `srun`
succeeds; interrupted stages remain incomplete. Reports record the exact
evaluation, prediction and Seasonal manifests consumed.
Forecast stages finalize canonical vanilla in its own successful scheduler step
before starting candidates that consume it. Reports retain both independent
configuration/repeat policies, dotted configuration filters and nested labels;
averages reduce repeats before configurations. Recovery and run-pinning entry
points remain under `src/scripts/`.
