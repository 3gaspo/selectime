"""Fresh-process timing of one uncached Selectime test batch per fitted method."""
from collections import defaultdict
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from time import perf_counter
import numpy as np

from timebench.data.windows import Windows
from timebench.evaluation.timing import EvaluationTimer
from timebench.pipeline.workflow import Workflow, seed_run, write_json, log
from timebench.proposal.candidates import UNIVARIATE, MULTIVARIATE, candidate_k, retrieval_covariates


TIMING_STAGES = ('measure', 'report')


def fitted_weights(method, alternative, selections, refs):
    """Use frozen validation choices; timing never refits on the test batch."""
    if alternative is None:
        return np.zeros(len(refs))
    weights = np.zeros(len(refs))
    for entry in selections['selections']:
        positions = ((refs[:, :2] == (entry['item'], entry['channel'])).all(axis=1)
                     if selections['granularity'] == 'per_variate' else np.ones(len(refs), dtype=bool))
        weights[positions] = (float(entry['selected_method'] == alternative)
                             if method.startswith('scope_selector') else float(entry['alternative_weight']))
    return weights


def cold_sample(request):
    """Load a backbone once, then execute one batch without any forecast/search cache."""
    from timebench.model_loading import load_forecaster
    from timebench.pipeline.runtime_resources import log_selected_device
    from timebench.proposal.retrieval import context_representation, blockwise_topk

    workflow = Workflow(request['config'])
    task = next(task for task in workflow.tasks
                if (task.dataset, task.term) == (request['dataset'], request['term']))
    windows = Windows(task, workflow.storage)
    refs = np.asarray(request['references'], dtype=np.int64)
    method = request['method']
    alternative = workflow.controls.get(method)
    weights = (np.ones(len(refs)) if method == MULTIVARIATE else
               fitted_weights(method, alternative, request.get('selection'), refs))
    required = np.asarray(request['required_target_mask'], dtype=bool)
    log_selected_device(workflow.device, stage='cold_batch', component='selectime_timing_worker',
                        model=workflow.model)
    seed_run(workflow.seed)
    load_timer = EvaluationTimer()
    load_timer.start()
    model = load_forecaster(workflow.model, workflow.weights, workflow.device, workflow.context_length)
    model_load_seconds = load_timer.stop()
    components = {'datastore_preprocessing_seconds': 0.0, 'query_retrieval_seconds': 0.0,
                  'backbone_seconds': 0.0, 'assembly_seconds': 0.0}
    calls, datastore_rows = 0, 0

    def forecast(positions, *, multivariate=False, past=None, future=None):
        nonlocal calls
        if not len(positions):
            return np.empty((0, task.prediction_length), dtype=np.float32)
        timer = EvaluationTimer()
        timer.start()
        if multivariate:
            keys, inverse = np.unique(refs[positions][:, (0, 2)], axis=0, return_inverse=True)
            histories = [windows.multivariate_history(int(item), int(origin), workflow.context_length)
                         for item, origin in keys]
            predicted = model.forecast(histories, task.prediction_length)
            values = np.stack([predicted[group][refs[row, 1]]
                               for row, group in zip(positions, inverse)])
        else:
            histories = windows.histories(refs[positions], workflow.context_length)
            predicted = model.forecast(histories, task.prediction_length,
                                       past_covariates=past, future_covariates=future)
            values = np.stack([value[0] for value in predicted])
        calls += 1
        components['backbone_seconds'] += timer.stop()
        return values

    timer = EvaluationTimer()
    timer.start()
    vanilla = np.full((len(refs), task.prediction_length), np.nan, dtype=np.float32)
    candidate = np.full_like(vanilla, np.nan)
    candidate_produced = np.zeros(len(refs), dtype=bool)
    vanilla_needed = np.flatnonzero(weights != 1)
    vanilla[vanilla_needed] = forecast(vanilla_needed)
    positions = np.flatnonzero(weights != 0)
    fallback = np.zeros(len(refs), dtype=bool)
    k = candidate_k(method)
    if len(positions) and k:
        started = perf_counter()
        all_test_refs = windows.references('test')
        store, ends = windows.datastore('test', all_test_refs)
        datastore_rows = len(store)
        r = task.retrieval_context_length
        representations = np.empty((len(store), r), dtype=np.float32)
        for start in range(0, len(store), workflow.config['datastore_block_size']):
            sequences = windows.sequences(store[start:start + workflow.config['datastore_block_size']])
            represented = context_representation(sequences[:, :r],
                normalize=workflow.config.get('retrieval_instance_normalization', True))
            represented[~np.isfinite(sequences[:, r:]).all(axis=1)] = np.nan
            representations[start:start + len(sequences)] = represented
        components['datastore_preprocessing_seconds'] = perf_counter() - started
        started = perf_counter()
        histories = windows.histories(refs[positions], workflow.context_length)
        query = np.full((len(positions), r), np.nan, dtype=np.float32)
        for index, history in enumerate(histories):
            if len(history) >= r:
                query[index] = context_representation(history[-r:],
                    normalize=workflow.config.get('retrieval_instance_normalization', True))
        _, ids = blockwise_topk(query, representations, refs[positions], store,
            windows.ticks(refs[positions]), windows.ticks(store), ends,
            k=max(workflow.k_values), period=task.alignment_period,
            stride=task.datastore_stride, horizon=task.prediction_length,
            scope=task.datastore_scope, minimum_overlap_fraction=workflow.config['minimum_overlap_fraction'],
            query_block_size=workflow.config['query_block_size'],
            datastore_block_size=workflow.config['datastore_block_size'])
        usable = (ids >= 0).all(axis=1)
        fallback[positions[~usable]] = True
        past, future = [], []
        for index in np.flatnonzero(usable):
            neighbors = windows.sequences(store[ids[index, :k]])
            p, f = retrieval_covariates(histories[index], neighbors, r, task.prediction_length,
                                        query_scaling=workflow.config.get('query_scaling', True))
            past.append(p)
            future.append(f)
        components['query_retrieval_seconds'] = perf_counter() - started
        candidate[positions[usable]] = forecast(positions[usable], past=past, future=future)
        candidate_produced[positions[usable]] = True
    elif len(positions):
        candidate[positions] = forecast(positions, multivariate=True)
        candidate_produced[positions] = True
    infinite = required & (((weights[:, None] != 0) & np.isinf(candidate)) | np.isinf(vanilla))
    if infinite.any():
        raise ValueError('Infinite cold-batch forecast on required test support')
    if len(positions):
        bad = ~np.all(~required[positions] | np.isfinite(candidate[positions]), axis=1)
        fallback[positions[bad]] = True
    missing_vanilla = np.flatnonzero(fallback & (weights == 1))
    vanilla[missing_vanilla] = forecast(missing_vanilla)
    vanilla_produced = np.unique(np.r_[vanilla_needed, missing_vanilla])
    produced_nan_values = int((required[candidate_produced] & np.isnan(candidate[candidate_produced])).sum()
                              + (required[vanilla_produced] & np.isnan(vanilla[vanilla_produced])).sum())
    started = perf_counter()
    values = vanilla.copy()
    active = (weights != 0) & ~fallback
    pure = active & (weights == 1)
    values[pure] = candidate[pure]
    mixed = active & ~pure
    values[mixed] = ((1 - weights[mixed, None]) * vanilla[mixed]
                     + weights[mixed, None] * candidate[mixed])
    invalid = ~np.all(~required | np.isfinite(values), axis=1)
    values[invalid] = vanilla[invalid]
    fallback[invalid & (weights != 0)] = True
    if (required & np.isinf(values)).any():
        raise ValueError('Infinite cold-batch forecast on required test support')
    components['assembly_seconds'] = perf_counter() - started
    seconds = timer.stop()
    return {'model_load_seconds': model_load_seconds, 'cold_batch_seconds': seconds,
            'model_load_plus_batch_seconds': model_load_seconds + seconds,
            **components, 'batch_rows': len(refs), 'backbone_calls': calls,
            'datastore_rows': datastore_rows, 'active_alternative_rows': int((weights != 0).sum()),
            'fallback_rows': int(fallback.sum()), 'produced_nan_values': produced_nan_values,
            'final_nan_values': int((required & np.isnan(values)).sum()),
            'cache_policy': 'no_predictions_representations_or_neighbors_loaded',
            'warmup_batches': 0, 'device': workflow.device}


class TimingWorkflow(Workflow):
    experiment = 'scope_timing'

    def __init__(self, config):
        super().__init__(config)
        self.source = Workflow(config)
        self.timed_methods = config['timing_methods'] or self.source.methods()
        if set(self.timed_methods) - set(self.source.methods()):
            raise ValueError(f'Unsupported timing methods for {self.model}: {self.timed_methods}')
        if int(config['timing_repetitions']) < 1:
            raise ValueError('timing_repetitions must be positive')

    def dependencies(self, task, method):
        dependencies = {'reference_evaluation': self.source.evaluation(task, method)}
        if method in self.source.controls:
            dependencies['calibration'] = self.source.resolve(task, 'selections', method,
                self.source.calibration_dependencies(task, method))
        return dependencies

    def science(self, task, phase, method, dependencies):
        scientific = super().science(task, phase, method, dependencies)
        if phase == 'measurements':
            scientific['model_config'].update(backbone=self.model, context_length=self.context_length)
            scientific['pipeline_config'].update(
                timing_policy='fresh_process_one_cold_batch_no_candidate_representation_neighbor_cache',
                model_loading='separately_measured_before_first_forecast',
                fitting='completed_validation_selection_outside_timer',
                reference_batch='first_required_official_test_rows', batch_size=self.batch_size,
                requested_device=self.device,
                repetitions=int(self.config['timing_repetitions']),
                maximum_k_support=max(self.k_values),
                query_block_size=self.config['query_block_size'],
                datastore_block_size=self.config['datastore_block_size'])
            scientific['experiment_config']['seed'] = self.seed
        return scientific

    def measure(self):
        from timebench.paths import PROJECT_ROOT
        worker = PROJECT_ROOT / 'src/scripts/selectime_timing_worker.py'
        for task in self.tasks:
            windows = Windows(task, self.storage)
            refs = windows.references('test')
            targets, cells = self.source.support(task, 'test', windows, refs)
            positions = np.flatnonzero(cells)[:self.batch_size]
            if not len(positions):
                raise ValueError(f'No required timing batch for {task.dataset}/{task.term}')
            for method in self.timed_methods:
                deps = self.dependencies(task, method)
                with self.allocate(task, 'measurements', method, deps) as run:
                    if not run.should_run:
                        continue
                    request = {'config': self.config, 'dataset': task.dataset, 'term': task.term,
                               'method': method, 'references': refs[positions].tolist(),
                               'required_target_mask': targets[positions].tolist()}
                    if 'calibration' in deps:
                        request['selection'] = json.loads((deps['calibration'] / 'selection.json').read_text())
                    samples = []
                    for repeat in range(int(self.config['timing_repetitions'])):
                        log(f'cold batch model={self.model} method={method} task={task.dataset}/{task.term} repeat={repeat}')
                        with tempfile.TemporaryDirectory(prefix='cold_batch_', dir=run.run_dir) as scratch:
                            scratch = Path(scratch)
                            write_json(scratch / 'worker_request.json', request)
                            with (run.run_dir / 'cold_process.log').open('a', encoding='utf-8') as stream:
                                subprocess.run([sys.executable, str(worker), str(scratch / 'worker_request.json'),
                                                str(scratch / 'worker_result.json')], cwd=PROJECT_ROOT,
                                               stdout=stream, stderr=subprocess.STDOUT, check=True)
                            samples.append(json.loads((scratch / 'worker_result.json').read_text()))
                    write_json(run.run_dir / 'timing_summary.json', {
                        'schema_version': 1, 'model': self.model, 'method': method,
                        'dataset': task.dataset, 'term': task.term, 'references': request['references'],
                        'samples': samples, 'repetitions': len(samples),
                        'cold_batch_seconds': float(np.median([sample['cold_batch_seconds'] for sample in samples])),
                        'model_load_seconds': float(np.median([sample['model_load_seconds'] for sample in samples])),
                        'process_per_sample': 'fresh', 'warmup_batches': 0,
                        'cache_policy': 'none; only_frozen_validation_choices_are_supplied',
                        'batch_definition': 'up_to_batch_size_required_test_window_variate_rows',
                        'calibration_and_arrow_loading': 'outside_cold_batch_timer',
                        'native_backbone_calls': 'one_per_needed_branch_in_this_single_batch',
                    })
                    self.finish(run, ['timing_summary.json', 'cold_process.log'])

    def report(self):
        from timebench.results.comparison import write_csv
        rows, deps = [], {}
        for task in self.tasks:
            for method in self.timed_methods:
                measured = self.resolve(task, 'measurements', method, self.dependencies(task, method))
                deps[f'{task.dataset}/{task.term}/{method}'] = measured
                summary = json.loads((measured / 'timing_summary.json').read_text())
                for repeat, sample in enumerate(summary['samples']):
                    rows.append({'model': self.model, 'dataset': task.dataset, 'term': task.term,
                                 'method': method, 'repeat': repeat, **sample})
        with self.allocate(self.tasks[0], 'reports', 'comparison', deps, stale_policy='new') as run:
            if not run.should_run:
                return
            write_csv(run.run_dir / 'timings.csv', rows)
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['method']].append(row)
            overall = []
            for method, values in grouped.items():
                overall.append({'model': self.model, 'method': method, 'samples': len(values),
                                'median_cold_batch_seconds': float(np.median([v['cold_batch_seconds'] for v in values])),
                                'median_model_load_seconds': float(np.median([v['model_load_seconds'] for v in values])),
                                'total_fallback_rows': sum(v['fallback_rows'] for v in values)})
            write_csv(run.run_dir / 'timing_comparison.csv', overall)
            write_json(run.run_dir / 'timing_manifest.json', {
                'schema_version': 1, 'experiment': self.experiment,
                'inputs': {name: self.dependency_reference(path) for name, path in deps.items()},
                'aggregation': 'median_over_task_batches_and_fresh_process_repeats',
                'coverage': 'one_test_batch_per_task_and_method; not_full_test_loop_walltime',
            })
            self.finish(run, ['timings.csv', 'timing_comparison.csv', 'timing_manifest.json'])

    def run(self, stage):
        from timebench.pipeline.runtime_resources import log_selected_device
        log_selected_device('cpu', stage=stage, component='selectime_timing_orchestrator')
        if stage == 'pipeline':
            if os.getenv('SELECTIME_DEFER_COMPLETION') == '1':
                raise ValueError('Slurm must invoke one stage per srun before finalization')
            for selected in TIMING_STAGES:
                getattr(self, selected)()
        elif stage in TIMING_STAGES:
            getattr(self, stage)()
        else:
            raise ValueError(f'Timing stages are {TIMING_STAGES}')
