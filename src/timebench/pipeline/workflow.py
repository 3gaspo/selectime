"""Task-level stages, exact scientific identities, and shared candidate artifacts."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import os
import random
import numpy as np

from timebench.data.windows import Task, Windows, write_prepared
from timebench.paths import PROJECT_ROOT, dataset_storage_root, outputs_root, weights_root
from timebench.pipeline.runs import allocate_run, load_manifest, select_completed_runs
from timebench.proposal.candidates import (UNIVARIATE, MULTIVARIATE, SELF_AUGMENTATION,
    candidate_names, candidate_k, self_covariates, query_scaled_sequences, align_covariates,
    control_candidates, HORIZON, HORIZON_MIX)

SOURCE_REVISIONS = {'improved_time': '541a2802cd2a35d39156aef4c37de6964d112786',
                    'adaptime': '33e75400d8c64414e4e13567c4e899803908bc44'}
STAGES = ('prepare', 'extract_validation', 'predict_validation', 'select_task', 'select_per_variate',
          'extract_test', 'predict_test', 'assemble', 'evaluate', 'report')
SELECTED_METHODS = ('selected_task', 'selected_per_variate')


def log(message):
    print(f'[{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}] {message}', flush=True)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def seed_run(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_array(path, shape, dtype=np.float32):
    # Empty validation is a valid selection input, but cannot be memory-mapped.
    if not np.prod(shape):
        return np.empty(shape, dtype=dtype)
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def finish_array(path, values):
    if isinstance(values, np.memmap):
        values.flush()
    else:
        np.save(path, values, allow_pickle=False)


class Workflow:
    def __init__(self, config):
        import yaml
        from gluonts.time_feature import get_seasonality
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / '.env')
        self.config = config
        self.model = config['model']
        limits = {'chronos2': 8192, 'chronos_bolt': 2048, 'ts_icl': 4096}
        if self.model not in limits:
            raise ValueError('Supported backbones: chronos2, chronos_bolt, ts_icl')
        if config['prediction_group'] not in ('all', 'vanilla', 'remaining'):
            raise ValueError('prediction_group must be all, vanilla or remaining')
        from timebench.pipeline.runs import CONFIG_POLICIES, REPEAT_POLICIES
        if config['report_config_policy'] not in CONFIG_POLICIES or config['report_repeat_policy'] not in REPEAT_POLICIES:
            raise ValueError('Invalid report configuration or repeat policy')
        self.root, self.storage, self.weights = outputs_root() / 'selectime', dataset_storage_root(), weights_root()
        self.seed, self.device = int(config['seed']), config['device']
        self.batch_size = int(config['batch_size'])
        default_context = 8192 if self.model == 'chronos2' else 2048
        self.context_length = int(config['context_length'] or default_context)
        self.k_values = (1,) if self.model == 'chronos_bolt' else tuple(map(int, config['k_values']))
        if not self.k_values or min(self.k_values) <= 0 or tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError('k_values must be a sorted, distinct, positive list')
        if not 1 <= self.context_length <= limits[self.model] or self.batch_size < 1:
            raise ValueError('Context exceeds backbone limit or batch_size is not positive')
        if self.model != 'chronos_bolt' and 5 not in self.k_values:
            raise ValueError('top5_mix requires K=5 in k_values')
        if config['bootstrap_replications'] < 2 or (config['bootstrap_block_length'] is not None and config['bootstrap_block_length'] < 1):
            raise ValueError('Invalid bootstrap settings')
        if min(config['query_block_size'], config['datastore_block_size']) < 1 or not 0 < config['minimum_overlap_fraction'] <= 1:
            raise ValueError('Invalid retrieval block size or overlap fraction')
        self.candidates = candidate_names(self.k_values, self.model)
        self.controls = control_candidates(self.model)
        self.raw_methods = [*self.candidates, HORIZON]
        self.config_path = Path(config['dataset_config'] or Path(__file__).parents[1] / 'config/datasets.yaml').resolve()
        settings = yaml.safe_load(self.config_path.read_text(encoding='utf-8'))
        names = list(settings['datasets']) if config['datasets'] == ['all'] else config['datasets']
        terms = config['terms']
        if config['experiment_mode'] not in ('full', 'test'):
            raise ValueError('experiment_mode must be full or test')
        if config['experiment_mode'] == 'test' and config['datasets'] == ['all']:
            names, terms = ['SG_Weather/D'], ['short']
        self.tasks = []
        for name in names:
            if name in config['excluded_datasets']:
                continue
            ds = settings['datasets'][name]
            protocol = settings['selectime_tasks'][name]
            for term in terms:
                if term not in ds:
                    continue
                effective = {**protocol, **protocol.get('ranges', {}).get(term, {})}
                horizon, test_length = int(ds[term]['prediction_length']), int(ds['test_length'])
                stride = int(effective['validation_stride'] if config['validation_stride'] is None else config['validation_stride'])
                # Preserve Adaptime's validation-date count; discard its training interval.
                validation = config['validation_length']
                validation = horizon + (test_length // horizon - 1) * stride if validation is None else int(validation)
                task = Task(name, term, horizon, test_length, validation,
                            int(get_seasonality(name.rpartition('/')[2])),
                            int(effective['retrieval_context_length'] if config['retrieval_context_length'] is None else config['retrieval_context_length']),
                            int(effective['alignment_period']),
                            int(effective['datastore_stride'] if config['datastore_stride'] is None else config['datastore_stride']), stride,
                            config['datastore_scope'], config['max_datastore_windows'])
                task.validate()
                if task.retrieval_context_length > self.context_length:
                    raise ValueError('Retrieval lookback must not exceed model context_length')
                self.tasks.append(task)
        if not self.tasks:
            raise ValueError('No tasks selected')
        log(f'dataset_config={self.config_path} applied_keys=datasets,selectime_tasks tasks={len(self.tasks)}')

    def identity(self, task, method):
        return {'model': self.model, 'target_mode': 'multivariate' if method == MULTIVARIATE else 'univariate',
                'dataset': task.dataset.rpartition('/')[0], 'frequency': task.dataset.rpartition('/')[2],
                'term': task.term, 'method': method}

    def path(self, task, phase, method):
        return self.root / self.model / phase / method / task.dataset / task.term

    def science(self, task, phase, method, dependencies):
        from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION
        pipeline = {'task': task.config(), 'k_values': list(self.k_values), 'maximum_k_support': True,
                    'representation': 'instance', 'distance': 'euclidean',
                    'minimum_overlap_fraction': float(self.config['minimum_overlap_fraction']),
                    'neighbor_scaling': 'lookback_statistics_to_query_lookback_scale',
                    'datastore_policy': 'validation_before_validation_start_test_before_test_start',
                    'self_covariates': ['sqrt_abs_past_only', 'sign_past_only'],
                    'candidate_methods': self.candidates,
                    'control_candidates': self.controls,
                    'horizon_support': 'one_neighbor_independent_of_maximum_k',
                    'mixture_rule': 'beta_1_1_validation_window_msse_win_frequency_half_ties',
                    'dependencies': {name: {key: load_manifest(path)[key] for key in
                        ('schema_version', 'identity', 'model_config', 'pipeline_config', 'experiment_config')}
                        for name, path in dependencies.items()}}
        if phase == 'selections':
            pipeline.update(bootstrap_replications=int(self.config['bootstrap_replications']),
                            bootstrap_block_length=self.config['bootstrap_block_length'],
                            selection_rule='paired_moving_date_block_bootstrap_one_standard_error',
                            selection_preference='univariate_then_multivariate_then_self_augmentation_then_smallest_k')
        if phase == 'evaluations':
            pipeline['evaluation_grid'] = EVALUATION_GRID_DEFINITION
        if phase == 'reports':
            pipeline['report_selection'] = {key: self.config[key] for key in
                ('report_current_config', 'report_config_filters', 'report_config_policy', 'report_repeat_policy')}
        checkpoints = {'chronos2': 'chronos2', 'chronos_bolt': 'chronos-bolt-base', 'ts_icl': 'tsicl/tsicl-v1.ckpt'}
        return {'model_config': {'backbone': self.model, 'checkpoint': checkpoints[self.model],
                                 'context_length': self.context_length, 'method': method,
                                 'point_forecast': 'median', 'cross_learning': False},
                'pipeline_config': pipeline, 'experiment_config': {'seed': self.seed}}

    def allocate(self, task, phase, method, dependencies=None):
        dependencies = dependencies or {}
        return allocate_run(self.path(task, phase, method), experiment='selectime', identity=self.identity(task, method),
                            **self.science(task, phase, method, dependencies),
                            runtime_config={'device': self.device, 'batch_size': self.batch_size,
                                            'query_block_size': self.config['query_block_size'],
                                            'datastore_block_size': self.config['datastore_block_size']},
                            provenance={'source_revisions': SOURCE_REVISIONS, 'dataset_config_path': str(self.config_path),
                                        'dataset_path': str(self.storage / task.dataset),
                                        'upstream_manifests': {name: str(path / 'manifest.json') for name, path in dependencies.items()}})

    def resolve(self, task, phase, method, dependencies=None):
        expected = self.science(task, phase, method, dependencies or {})
        selected = select_completed_runs(self.path(task, phase, method), config_policy='distinct', repeat_policy='selected')
        matches = [path for path, manifest in selected if manifest['identity'] == self.identity(task, method)
                   and all(manifest[key] == value for key, value in expected.items())]
        if len(matches) != 1:
            raise ValueError(f'Expected one exact completed {phase}/{method}: {task.dataset}/{task.term}; found {len(matches)}')
        return matches[0]

    def finish(self, run, files):
        if os.getenv('SELECTIME_DEFER_COMPLETION') == '1':
            write_json(run.run_dir / 'stage_ready.json', {'required_artifacts': files})
            run._completed = True  # Finalizer marks completion only after successful srun.
        else:
            run.complete(files)

    def prepared(self, task):
        return self.resolve(task, 'data', 'shared')

    def extraction(self, task, split):
        return self.resolve(task, 'retrieval', split, {'data': self.prepared(task)})

    def prediction_dependencies(self, task, split, method):
        deps = {'data': self.prepared(task)}
        if method != UNIVARIATE:
            deps['vanilla'] = self.raw(task, split, UNIVARIATE)
        if candidate_k(method) or method == HORIZON:
            deps['retrieval'] = self.extraction(task, split)
        return deps

    def raw(self, task, split, method):
        return self.resolve(task, f'predictions/{split}', method, self.prediction_dependencies(task, split, method))

    def selection_dependencies(self, task):
        return {'data': self.prepared(task), **{method: self.raw(task, 'validation', method) for method in self.candidates}}

    def selection(self, task, granularity):
        return self.resolve(task, 'selections', granularity, self.selection_dependencies(task))

    def assembled_dependencies(self, task, granularity):
        return {'selection': self.selection(task, granularity), 'data': self.prepared(task),
                **{method: self.raw(task, 'test', method) for method in self.candidates}}

    def prediction(self, task, method):
        if method in self.controls:
            return self.resolve(task, 'predictions/test', method, self.control_dependencies(task, method, 'test'))
        if method in SELECTED_METHODS:
            granularity = 'task' if method == 'selected_task' else 'per_variate'
            return self.resolve(task, 'predictions/test', method, self.assembled_dependencies(task, granularity))
        return self.raw(task, 'test', method)

    def control_dependencies(self, task, method, split):
        alternative = self.controls[method]
        deps = {'data': self.prepared(task), 'vanilla': self.raw(task, split, UNIVARIATE),
                'alternative': self.raw(task, split, alternative)}
        if split == 'test':
            deps['calibration'] = self.resolve(task, 'selections', method,
                                               self.control_dependencies(task, method, 'validation'))
        return deps

    def array(self, data, name):
        # np.load handles zero-row arrays as well as ordinary mmap products.
        path = data / f'{name}.npy'
        try:
            return np.load(path, mmap_mode='r', allow_pickle=False)
        except ValueError:
            return np.load(path, allow_pickle=False)

    def support(self, task, split, windows, refs):
        targets = np.isfinite(windows.labels(refs))
        if split == 'test':
            from timebench.evaluation.grid import flatten_univariate_grid, load_evaluation_grid
            from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
            expected, cells = flatten_univariate_grid(*load_evaluation_grid(
                resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
            if not np.array_equal(expected, targets):
                raise ValueError('Official test references and Seasonal grid do not align')
            return expected, cells
        return targets, targets.any(axis=-1)

    def prepare(self):
        for task in self.tasks:
            with self.allocate(task, 'data', 'shared') as run:
                if run.should_run:
                    log(f'prepare {task.dataset}/{task.term}')
                    self.finish(run, write_prepared(Windows(task, self.storage), run.run_dir))

    def extract(self, split):
        from timebench.proposal.retrieval import context_representation, blockwise_topk
        for task in self.tasks:
            data = self.prepared(task)
            with self.allocate(task, 'retrieval', split, {'data': data}) as run:
                if not run.should_run:
                    continue
                log(f'extract_{split} {task.dataset}/{task.term}')
                windows = Windows(task, self.storage)
                refs = self.array(data, f'{split}_references')
                datastore_refs = self.array(data, f'{split}_datastore')
                r, h = task.retrieval_context_length, task.prediction_length
                started = perf_counter()
                representations = save_array(run.run_dir / 'datastore_representation.npy', (len(datastore_refs), r))
                for start in range(0, len(datastore_refs), self.config['datastore_block_size']):
                    rows = datastore_refs[start:start + self.config['datastore_block_size']]
                    sequences = windows.sequences(rows)
                    represented = context_representation(sequences[:, :r])
                    represented[~np.isfinite(sequences[:, r:]).all(axis=-1)] = np.nan
                    representations[start:start + len(rows)] = represented
                finish_array(run.run_dir / 'datastore_representation.npy', representations)
                preprocessing_seconds = perf_counter() - started
                started = perf_counter()
                query = save_array(run.run_dir / 'query_representation.npy', (len(refs), r))
                query[:] = np.nan
                for start in range(0, len(refs), self.config['query_block_size']):
                    rows = refs[start:start + self.config['query_block_size']]
                    for local, history in enumerate(windows.histories(rows, r)):
                        if len(history) == r:
                            query[start + local] = context_representation(history)
                finish_array(run.run_dir / 'query_representation.npy', query)
                distances, ids = blockwise_topk(query, representations, refs, datastore_refs,
                    self.array(data, f'{split}_ticks'), self.array(data, f'{split}_datastore_ticks'),
                    self.array(data, f'{split}_datastore_ends'), k=max(self.k_values),
                    period=task.alignment_period, stride=task.datastore_stride, horizon=h,
                    scope=task.datastore_scope, minimum_overlap_fraction=self.config['minimum_overlap_fraction'],
                    query_block_size=self.config['query_block_size'], datastore_block_size=self.config['datastore_block_size'])
                np.save(run.run_dir / 'neighbor_ids.npy', ids, allow_pickle=False)
                # Horizon transfer needs one neighbor even when max-K covariates are ineligible.
                _, horizon_ids = blockwise_topk(query, representations, refs, datastore_refs,
                    self.array(data, f'{split}_ticks'), self.array(data, f'{split}_datastore_ticks'),
                    self.array(data, f'{split}_datastore_ends'), k=1,
                    period=task.alignment_period, stride=task.datastore_stride, horizon=h,
                    scope=task.datastore_scope, minimum_overlap_fraction=self.config['minimum_overlap_fraction'],
                    query_block_size=self.config['query_block_size'], datastore_block_size=self.config['datastore_block_size'])
                np.save(run.run_dir / 'horizon_neighbor_ids.npy', horizon_ids, allow_pickle=False)
                np.save(run.run_dir / 'neighbor_distances.npy', distances, allow_pickle=False)
                np.save(run.run_dir / 'eligible.npy', (ids >= 0).all(axis=-1), allow_pickle=False)
                write_json(run.run_dir / 'retrieval.json', {'schema_version': 1, 'split': split,
                    'max_k': max(self.k_values), 'queries': len(refs), 'eligible_queries': int((ids >= 0).all(axis=-1).sum()),
                    'datastore_windows': len(datastore_refs), 'datastore_preprocessing_seconds': preprocessing_seconds,
                    'query_retrieval_seconds': perf_counter() - started})
                self.finish(run, ['datastore_representation.npy', 'query_representation.npy', 'neighbor_ids.npy', 'horizon_neighbor_ids.npy', 'neighbor_distances.npy', 'eligible.npy', 'retrieval.json'])

    def predict(self, split):
        from timebench.model_loading import load_forecaster
        from timebench.evaluation.timing import EvaluationTimer
        model = None
        group = self.config['prediction_group']
        methods = [method for method in self.raw_methods
                   if group == 'all' or (method == UNIVARIATE) == (group == 'vanilla')]
        for task in self.tasks:
            data = self.prepared(task)
            windows = Windows(task, self.storage)
            refs = self.array(data, f'{split}_references')
            targets, cells = self.support(task, split, windows, refs)
            for method in methods:
                deps = self.prediction_dependencies(task, split, method)
                with self.allocate(task, f'predictions/{split}', method, deps) as run:
                    if not run.should_run:
                        continue
                    log(f'predict_{split} method={method} {task.dataset}/{task.term}')
                    seed_run(self.seed)
                    if model is None and len(refs) and method != HORIZON:
                        model = load_forecaster(self.model, self.weights, self.device, self.context_length)
                    values = save_array(run.run_dir / 'prediction.npy', (len(refs), task.prediction_length))
                    fallback = np.zeros(len(refs), dtype=bool)
                    vanilla = None if method == UNIVARIATE else self.array(deps['vanilla'], 'prediction')
                    if vanilla is not None:
                        values[:] = vanilla
                    extraction = deps.get('retrieval')
                    k = candidate_k(method)
                    retrieval = bool(k) or method == HORIZON
                    ids = self.array(extraction, 'horizon_neighbor_ids' if method == HORIZON else 'neighbor_ids') if retrieval else None
                    datastore = self.array(data, f'{split}_datastore') if retrieval else None
                    timer = EvaluationTimer()
                    timer.start()
                    if method == MULTIVARIATE:
                        # Forecast each item/date once, then map channels to canonical univariate rows.
                        keys, inverse = np.unique(refs[:, (0, 2)], axis=0, return_inverse=True)
                        row_order = np.argsort(inverse, kind='stable')
                        offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(keys)))]
                        for start in range(0, len(keys), self.batch_size):
                            histories = [windows.multivariate_history(item, origin, self.context_length)
                                         for item, origin in keys[start:start + self.batch_size]]
                            forecasts = model.forecast(histories, task.prediction_length)
                            for local, forecast in enumerate(forecasts):
                                group = start + local
                                positions = row_order[offsets[group]:offsets[group + 1]]
                                values[positions] = forecast[refs[positions, 1]]
                    else:
                        for start in range(0, len(refs), self.batch_size):
                            positions = np.arange(start, min(start + self.batch_size, len(refs)))
                            if retrieval:
                                usable = (ids[positions] >= 0).all(axis=-1)
                                fallback[positions[~usable]] = True
                                positions = positions[usable]
                            if not len(positions):
                                continue
                            histories = windows.histories(refs[positions], self.context_length)
                            past, future = None, None
                            if method == SELF_AUGMENTATION:
                                past = [self_covariates(history) for history in histories]
                            elif retrieval:
                                past, future = [], []
                                for row, history in zip(positions, histories):
                                    neighbors = windows.sequences(datastore[ids[row, :k or 1]])
                                    scaled = query_scaled_sequences(history[-task.retrieval_context_length:], neighbors, task.prediction_length)
                                    p, f = align_covariates(scaled, len(history), task.prediction_length)
                                    past.append(p)
                                    future.append(f)
                            forecasts = ([np.asarray(value) for value in future] if method == HORIZON else
                                model.forecast(histories, task.prediction_length,
                                               past_covariates=past, future_covariates=future))
                            values[positions] = np.stack([forecast[0] for forecast in forecasts])
                    seconds = timer.stop()
                    invalid = cells & ~np.all(~targets | np.isfinite(values), axis=-1)
                    if invalid.any() and method == UNIVARIATE:
                        raise ValueError('Canonical univariate forecast is non-finite on required support')
                    if vanilla is not None:
                        values[invalid] = vanilla[invalid]
                        fallback[invalid] = True
                    finish_array(run.run_dir / 'prediction.npy', values)
                    np.save(run.run_dir / 'fallback.npy', fallback, allow_pickle=False)
                    retrieval_metadata = json.loads((extraction / 'retrieval.json').read_text()) if retrieval else {}
                    write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'split': split,
                        'context_length': self.context_length, 'inference_seconds': seconds,
                        'query_retrieval_seconds': retrieval_metadata.get('query_retrieval_seconds', 0),
                        'datastore_preprocessing_seconds': retrieval_metadata.get('datastore_preprocessing_seconds', 0),
                        'fallback_count': int((fallback & cells).sum()), 'grid_rows': int(cells.sum()),
                        'insufficient_retrieval_count': int((~(ids >= 0).all(axis=-1) & cells).sum()) if retrieval else 0,
                        'nonfinite_fallback_count': int(invalid.sum()),
                        'timing_policy': 'measured_candidate_query_loop_precomputed_vanilla_fallback_separate_retrieval'})
                    self.finish(run, ['prediction.npy', 'fallback.npy', 'prediction.json'])
        del model

    def select(self, granularity):
        from timebench.proposal.selection import select_candidates, select_with_block_bootstrap
        if granularity == 'task':
            self.calibrate_controls()
        if self.model == 'chronos_bolt':
            return
        for task in self.tasks:
            deps = self.selection_dependencies(task)
            with self.allocate(task, 'selections', granularity, deps) as run:
                if not run.should_run:
                    continue
                log(f'select_{granularity} {task.dataset}/{task.term}')
                windows = Windows(task, self.storage)
                refs = self.array(deps['data'], 'validation_references')
                predictions = {method: self.array(deps[method], 'prediction') for method in self.candidates}
                result = select_candidates(predictions, windows.labels(refs), windows.msse_scales(refs), refs,
                    self.array(deps['data'], 'validation_ticks'), granularity=granularity, seed=self.seed,
                    prediction_length=task.prediction_length, validation_stride=task.validation_stride,
                    replications=self.config['bootstrap_replications'], block_length=self.config['bootstrap_block_length'])
                if granularity == 'per_variate':
                    present = {(entry['item'], entry['channel']) for entry in result['selections']}
                    for item, channel in np.unique(self.array(deps['data'], 'test_references')[:, :2], axis=0):
                        if (int(item), int(channel)) not in present:
                            entry = select_with_block_bootstrap({name: np.empty(0) for name in self.candidates},
                                seed=self.seed, prediction_length=task.prediction_length, validation_stride=task.validation_stride)
                            entry.update(item=int(item), channel=int(channel), validation_date_ticks=[])
                            result['selections'].append(entry)
                write_json(run.run_dir / 'selection.json', result)
                self.finish(run, ['selection.json'])

    def calibrate_controls(self):
        from timebench.proposal.selection import select_candidates, win_frequency_mixture
        for task in self.tasks:
            windows = Windows(task, self.storage)
            for method, alternative in self.controls.items():
                deps = self.control_dependencies(task, method, 'validation')
                with self.allocate(task, 'selections', method, deps) as run:
                    if not run.should_run:
                        continue
                    refs = self.array(deps['data'], 'validation_references')
                    predictions = {UNIVARIATE: self.array(deps['vanilla'], 'prediction'),
                                   alternative: self.array(deps['alternative'], 'prediction')}
                    labels, scales = windows.labels(refs), windows.msse_scales(refs)
                    if method == 'scope_selector':
                        result = select_candidates(predictions, labels, scales, refs,
                            self.array(deps['data'], 'validation_ticks'), granularity='task', seed=self.seed,
                            prediction_length=task.prediction_length, validation_stride=task.validation_stride,
                            replications=self.config['bootstrap_replications'], block_length=self.config['bootstrap_block_length'])
                    else:
                        result = win_frequency_mixture(predictions, labels, scales,
                            eligible=~self.array(deps['alternative'], 'fallback'))
                    write_json(run.run_dir / 'selection.json', result)
                    self.finish(run, ['selection.json'])

    def assemble(self):
        self.assemble_controls()
        if self.model == 'chronos_bolt':
            return
        for task in self.tasks:
            refs = self.array(self.prepared(task), 'test_references')
            for granularity, method in zip(('task', 'per_variate'), SELECTED_METHODS):
                deps = self.assembled_dependencies(task, granularity)
                with self.allocate(task, 'predictions/test', method, deps) as run:
                    if not run.should_run:
                        continue
                    selection = json.loads((deps['selection'] / 'selection.json').read_text())
                    values = save_array(run.run_dir / 'prediction.npy', (len(refs), task.prediction_length))
                    fallback = np.zeros(len(refs), dtype=bool)
                    chosen = np.empty(len(refs), dtype=np.int64)
                    records = selection['selections']
                    for entry in records:
                        positions = (np.flatnonzero((refs[:, :2] == (entry['item'], entry['channel'])).all(axis=-1))
                                     if granularity == 'per_variate' else np.arange(len(refs)))
                        selected = entry['selected_method']
                        values[positions] = self.array(deps[selected], 'prediction')[positions]
                        fallback[positions] = self.array(deps[selected], 'fallback')[positions]
                        chosen[positions] = self.candidates.index(selected)
                    finish_array(run.run_dir / 'prediction.npy', values)
                    np.save(run.run_dir / 'fallback.npy', fallback, allow_pickle=False)
                    np.save(run.run_dir / 'selected_candidate.npy', chosen, allow_pickle=False)
                    write_json(run.run_dir / 'selection.json', selection)
                    windows = Windows(task, self.storage)
                    _, cells = self.support(task, 'test', windows, refs)
                    write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'split': 'test',
                        'context_length': self.context_length, 'inference_seconds': None,
                        'timing_policy': 'assembled_from_candidate_artifacts_no_independent_selected_method_latency',
                        'fallback_count': int((fallback & cells).sum()), 'grid_rows': int(cells.sum()),
                        'candidate_names': self.candidates, 'granularity': granularity})
                    self.finish(run, ['prediction.npy', 'fallback.npy', 'selected_candidate.npy', 'selection.json', 'prediction.json'])

    def assemble_controls(self):
        from timebench.proposal.selection import blend
        for task in self.tasks:
            windows = Windows(task, self.storage)
            refs = self.array(self.prepared(task), 'test_references')
            targets, cells = self.support(task, 'test', windows, refs)
            for method, alternative in self.controls.items():
                deps = self.control_dependencies(task, method, 'test')
                with self.allocate(task, 'predictions/test', method, deps) as run:
                    if not run.should_run:
                        continue
                    calibration = json.loads((deps['calibration'] / 'selection.json').read_text())
                    weight = (float(calibration['selections'][0]['selected_method'] == alternative)
                              if method == 'scope_selector' else calibration['alternative_weight'])
                    vanilla = self.array(deps['vanilla'], 'prediction')
                    candidate = self.array(deps['alternative'], 'prediction')
                    # Assemble in batches without copying both full candidate payloads into RAM.
                    values = save_array(run.run_dir / 'prediction.npy', vanilla.shape)
                    for start in range(0, len(refs), self.batch_size):
                        stop = start + self.batch_size
                        values[start:stop] = blend(vanilla[start:stop], candidate[start:stop], weight)
                    fallback = (np.asarray(self.array(deps['alternative'], 'fallback')).copy() if weight else
                                np.zeros(len(refs), dtype=bool))
                    invalid = cells & ~np.all(~targets | np.isfinite(values), axis=-1)
                    values[invalid], fallback[invalid] = vanilla[invalid], True
                    finish_array(run.run_dir / 'prediction.npy', values)
                    np.save(run.run_dir / 'fallback.npy', fallback, allow_pickle=False)
                    write_json(run.run_dir / 'selection.json', calibration)
                    write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'split': 'test',
                        'context_length': self.context_length, 'inference_seconds': None,
                        'timing_policy': 'assembled_from_candidate_artifacts_no_independent_selected_method_latency',
                        'alternative': alternative, 'alternative_weight': weight,
                        'fallback_count': int((fallback & cells).sum()), 'grid_rows': int(cells.sum())})
                    self.finish(run, ['prediction.npy', 'fallback.npy', 'selection.json', 'prediction.json'])

    def methods(self):
        if self.model == 'chronos_bolt':
            return [UNIVARIATE, HORIZON_MIX]
        return [*self.candidates, *SELECTED_METHODS, *self.controls]

    def evaluation(self, task, method):
        return self.resolve(task, 'evaluations', method, {'prediction': self.prediction(task, method)})

    def evaluate(self):
        from timebench.evaluation.data import Dataset
        from timebench.evaluation.saver import save_window_predictions
        from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
        for task in self.tasks:
            dataset = Dataset(task.dataset, term=task.term, prediction_length=task.prediction_length,
                              test_length=task.test_length, storage_path=self.storage)
            for method in self.methods():
                prediction = self.prediction(task, method)
                with self.allocate(task, 'evaluations', method, {'prediction': prediction}) as run:
                    if not run.should_run:
                        continue
                    log(f'evaluate {method} {task.dataset}/{task.term}')
                    metadata = json.loads((prediction / 'prediction.json').read_text())
                    save_window_predictions(dataset, self.array(prediction, 'prediction')[:, None, :],
                        f'{task.dataset}/{task.term}', str(self.root), seasonality=task.seasonality, quantile_levels=[0.5],
                        task_output_dir=str(run.run_dir), inference_seconds=metadata['inference_seconds'],
                        model_hyperparams={'model': self.model, 'method': method, 'experiment': 'selectime',
                            'target_mode': 'univariate', 'forecast_input_mode': self.identity(task, method)['target_mode'],
                            'context_length': self.context_length, 'prediction_manifest': str(prediction / 'manifest.json'),
                            'fallback_count': metadata['fallback_count'], 'timing_policy': metadata['timing_policy']},
                        evaluation_grid_path=str(resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
                    self.finish(run, ['predictions.npz', 'metrics.npz', 'metrics_summary.json', 'config.json'])

    def report(self):
        from timebench.results.comparison import build_report
        inputs = []
        for task in self.tasks:
            for method in self.methods():
                filters = dict(self.config['report_config_filters'])
                if self.config['report_current_config']:
                    expected = self.science(task, 'evaluations', method, {'prediction': self.prediction(task, method)})
                    filters.update(expected)
                selected = select_completed_runs(self.path(task, 'evaluations', method), config_filters=filters,
                    config_policy=self.config['report_config_policy'], repeat_policy=self.config['report_repeat_policy'])
                if not selected:
                    raise ValueError(f'No completed report inputs for {task.dataset}/{task.term}/{method}')
                for evaluation, manifest in selected:
                    # The recorded run name survives relocation between this project's execution surfaces.
                    recorded = Path(manifest['provenance']['upstream_manifests']['prediction'])
                    prediction = self.path(task, 'predictions/test', method) / recorded.parent.name
                    upstream = load_manifest(prediction)
                    expected = manifest['pipeline_config']['dependencies']['prediction']
                    if upstream['status'] != 'completed' or any(upstream[key] != value for key, value in expected.items()):
                        raise ValueError(f'Report prediction does not match its evaluation: {prediction}')
                    inputs.append((task, method, evaluation, prediction, manifest['selection']))
        # Report is itself a recoverable task and records its complete selected input identities.
        anchor = self.tasks[0]
        deps = {f'{task.dataset}/{task.term}/{method}/input_{index}': evaluation
                for index, (task, method, evaluation, _, _) in enumerate(inputs)}
        with self.allocate(anchor, 'reports', 'comparison', deps) as run:
            if run.should_run:
                self.finish(run, build_report(inputs, run.run_dir, self.config))

    def run(self, stage):
        log(f'stage={stage} tasks={len(self.tasks)} seed={self.seed} Slurm={os.getenv("SLURM_JOB_ID")}')
        if stage == 'pipeline':
            if os.getenv('SELECTIME_DEFER_COMPLETION') == '1':
                raise ValueError('Slurm must invoke one stage per srun before finalization')
            for item in STAGES:
                self.run(item)
        elif stage.startswith('extract_') and stage in STAGES:
            self.extract(stage.removeprefix('extract_'))
        elif stage.startswith('predict_') and stage in STAGES:
            self.predict(stage.removeprefix('predict_'))
        elif stage in ('select_task', 'select_per_variate'):
            self.select(stage.removeprefix('select_'))
        elif stage in ('prepare', 'assemble', 'evaluate', 'report'):
            getattr(self, stage)()
        else:
            raise ValueError(f'Unknown stage {stage}')
