# Architecture

Selectime is a direct Improved TIME child. Its scientific proposal is the
selection and soft mixing of foundation-model inputs, implemented under `src/timebench/proposal/`.
This package contains normalized distance, eligible top-K search, past-only
neighbor scaling, the paired bootstrap scope selectors, Beta-smoothed
win-frequency mixtures, and the closed-form scope ridge. It has
no dependency on scheduler, Hydra, manifests or reporting.

`data/windows.py` owns the observed-prefix reader and the chronological
datastore/validation/test indices. `model_loading/` owns the native Chronos-2, Chronos-Bolt and TS-ICL
adapters, dispatched by one exact canonical alias. `pipeline/workflow.py` composes these
owners and allocates each task before work. The standard stages are:

```text
saved Arrow + official TIME settings + task retrieval settings
  -> prepare indices for validation and official test
  -> validation retrieval -> candidate validation forecasts
  -> task/per-variate scope selection and mixture/ridge calibration
  -> expanded pre-test retrieval -> all candidate test forecasts
  -> assemble frozen scope controls and top-K mixtures from raw predictions
  -> inherited TIME metrics on the same Seasonal grid -> comparison report
```

Multivariate forecasting runs once per item/date and is projected into the
canonical item/channel/date order. This makes both scopes and all mixtures
comparable on the same metric cells. Univariate candidates read
one target variate and only their declared covariates.

Validation and test retrieval own distinct neighbor artifacts because their
datastore cutoffs differ. Maximum-K covariate search retains ordered prefixes.
Datastore and query representations are memory-mapped. Trajectories remain in
Arrow until needed for a forecast; there are no datastore backbone forecasts.
Validation calibrations store hard scope choices, frozen win-frequency weights,
or the fixed-alpha closed-form scope-ridge coefficient. All current paths include the backbone
under `outputs/selectime/<backbone>/`, preventing cross-backbone report or
manifest selection. Prior flat-layout results remain historical evidence.

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
