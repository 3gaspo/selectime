"""Task-level stages, exact scientific identities, and shared candidate artifacts."""
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import json
import os
import random
import numpy as np

from timebench.data.windows import Task, Windows, write_prepared
from timebench.evaluation.fallback import summarize_fallbacks
from timebench.evaluation.validation import finite_row_mask, validation_window_mask
from timebench.paths import PROJECT_ROOT, dataset_storage_root, outputs_root, weights_root
from timebench.pipeline.runs import (allocate_run, load_manifest, manifest_reference,
    select_completed_runs)
from timebench.proposal.candidates import (UNIVARIATE, MULTIVARIATE, candidate_names,
    candidate_k, query_scaled_sequences, align_covariates, control_candidates)

SOURCE_REVISIONS = {'improved_time': '541a2802cd2a35d39156aef4c37de6964d112786',
                    'adaptime': '33e75400d8c64414e4e13567c4e899803908bc44'}
STAGES = ('prepare', 'extract_validation', 'predict_validation', 'calibrate',
          'extract_test', 'predict_test', 'assemble', 'evaluate', 'report')


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
        self.vanilla_predictions_path = (
            Path(config['vanilla_predictions_path']).expanduser().resolve()
            if config.get('vanilla_predictions_path') else None
        )
        self.seed, self.device = int(config['seed']), config['device']
        self.batch_size = int(config['batch_size'])
        default_context = limits[self.model]
        self.context_length = int(config['context_length'] or default_context)
        self.k_values = (1,) if self.model == 'chronos_bolt' else tuple(map(int, config['k_values']))
        if not self.k_values or min(self.k_values) <= 0 or tuple(sorted(set(self.k_values))) != self.k_values:
            raise ValueError('k_values must be a sorted, distinct, positive list')
        if not 1 <= self.context_length <= limits[self.model] or self.batch_size < 1:
            raise ValueError('Context exceeds backbone limit or batch_size is not positive')
        if config['bootstrap_replications'] < 2 or (config['bootstrap_block_length'] is not None and config['bootstrap_block_length'] < 1):
            raise ValueError('Invalid bootstrap settings')
        if float(config['scope_ridge_alpha']) <= 0:
            raise ValueError('scope_ridge_alpha must be positive')
        if min(config['query_block_size'], config['datastore_block_size']) < 1 or not 0 < config['minimum_overlap_fraction'] <= 1:
            raise ValueError('Invalid retrieval block size or overlap fraction')
        self.candidates = candidate_names(self.k_values, self.model)
        self.controls = control_candidates(self.model, self.k_values)
        self.raw_methods = list(self.candidates)
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
                validation = config['validation_length']
                validation = test_length if validation is None else int(validation)
                task = Task(name, term, horizon, test_length, validation,
                            int(get_seasonality(name.rpartition('/')[2])),
                            int(effective['retrieval_context_length'] if config['retrieval_context_length'] is None else config['retrieval_context_length']),
                            int(effective['alignment_period']),
                            int(effective['datastore_stride'] if config['datastore_stride'] is None else config['datastore_stride']), horizon,
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
        if phase == 'reports':
            return self.root.parent / 'reports' / 'selectime' / self.model / method / task.dataset / task.term
        return self.root / self.model / phase / method / task.dataset / task.term

    def dependency_reference(self, path):
        return manifest_reference(path)

    def science(self, task, phase, method, dependencies):
        from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION
        task_base = {
            'dataset': task.dataset,
            'term': task.term,
            'prediction_length': task.prediction_length,
            'test_length': task.test_length,
            'seasonality': task.seasonality,
        }
        pipeline = {
            'phase': phase,
            'method': method,
            'task': task_base,
            'dependencies': {
                name: self.dependency_reference(path)
                for name, path in dependencies.items()
            },
        }
        model = {'component': phase, 'method': method}
        experiment = {}
        if phase.startswith('data/'):
            split = phase.rpartition('/')[2]
            pipeline.update(
                split=split,
                retrieval_context_length=task.retrieval_context_length,
                alignment_period=task.alignment_period,
                datastore_stride=task.datastore_stride,
                datastore_scope=task.datastore_scope,
                max_datastore_windows=task.max_datastore_windows,
                datastore_policy=(
                    'before_first_validation_origin'
                    if split == 'validation'
                    else 'before_first_test_origin'
                ),
            )
            if split == 'validation':
                pipeline.update(
                    validation_length=task.validation_length,
                    validation_stride=task.prediction_length,
                    validation_schedule='walk_backward_from_first_test_origin_at_stride_H',
                )
        elif phase.startswith('retrieval/'):
            split = phase.rpartition('/')[2]
            pipeline.update(
                split=split,
                representation='instance_normalized_lookback',
                distance='euclidean',
                minimum_overlap_fraction=float(self.config['minimum_overlap_fraction']),
                neighbor_scaling='lookback_statistics_to_query_lookback_scale',
                retrieval_context_length=task.retrieval_context_length,
                alignment_period=task.alignment_period,
                datastore_stride=task.datastore_stride,
                datastore_scope=task.datastore_scope,
                neighbor_count=max(self.k_values) if method == 'covariate' else 1,
            )
        elif phase.startswith('predictions/'):
            split = phase.rpartition('/')[2]
            checkpoints = {'chronos2': 'chronos2', 'chronos_bolt': 'chronos-bolt-base', 'ts_icl': 'tsicl/tsicl-v1.ckpt'}
            model = {
                'backbone': self.model,
                'checkpoint': checkpoints[self.model],
                'context_length': self.context_length,
                'method': method,
                'point_forecast': 'median',
                'cross_learning': False,
            }
            pipeline.update(
                split=split,
                validation_support='finite_context_and_future' if split == 'validation' else None,
                prediction_artifact_contract='vanilla_reuse_nan_counts_retrieval_provenance',
            )
            if candidate_k(method):
                pipeline.update(
                    k=candidate_k(method),
                    neighbor_scaling='lookback_statistics_to_query_lookback_scale',
                )
            experiment['seed'] = self.seed
        if phase == 'selections':
            granularity = 'per_variate' if method.endswith('_per_variate') else 'task'
            if method.startswith('scope_selector'):
                rule = 'paired_moving_date_block_bootstrap_one_standard_error'
            elif method == 'scope_ridge':
                rule = 'closed_form_msse_weighted_ridge_alpha_1'
            else:
                rule = 'beta_1_1_validation_window_msse_win_frequency_half_ties'
            pipeline.update(
                validation_support='finite_context_and_future',
                granularity=granularity,
                no_validation_fallback='univariate',
                selection_rule=rule,
            )
            if method.startswith('scope_selector'):
                pipeline.update(
                    bootstrap_replications=int(self.config['bootstrap_replications']),
                    bootstrap_block_length=self.config['bootstrap_block_length'],
                    selection_preference='univariate_then_multivariate',
                )
            if method == 'scope_ridge':
                pipeline.update(
                    alpha=float(self.config['scope_ridge_alpha']),
                    fitting_windows='same_test_aligned_pretest_validation_windows_as_other_controls',
                    no_training_support='unfitted_vanilla_fallback',
                )
            experiment['seed'] = self.seed
        if phase == 'evaluations':
            pipeline['evaluation_grid'] = EVALUATION_GRID_DEFINITION
            pipeline['nan_policy'] = 'omit_nan_predictions_report_counts_reject_infinity'
        if phase == 'reports':
            pipeline['report_selection'] = {key: self.config[key] for key in
                ('report_current_config', 'report_config_filters', 'report_config_policy', 'report_repeat_policy')}
        return {
            'model_config': model,
            'pipeline_config': pipeline,
            'experiment_config': experiment,
        }

    def allocate(self, task, phase, method, dependencies=None):
        dependencies = dependencies or {}
        return allocate_run(self.path(task, phase, method), experiment='selectime', identity=self.identity(task, method),
                            **self.science(task, phase, method, dependencies),
                            runtime_config={'device': self.device, 'batch_size': self.batch_size,
                                            'query_block_size': self.config['query_block_size'],
                                            'datastore_block_size': self.config['datastore_block_size']},
                            provenance={'source_revisions': SOURCE_REVISIONS, 'dataset_config_path': str(self.config_path),
                                        'task_config': task.config(),
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
            run.compute(files)
        else:
            run.complete(files)

    def prepared(self, task, split):
        return self.resolve(task, f'data/{split}', 'shared')

    def extraction(self, task, split, kind):
        return self.resolve(task, f'retrieval/{split}', kind, {'data': self.prepared(task, split)})

    def prediction_dependencies(self, task, split, method):
        deps = {'data': self.prepared(task, split)}
        if method != UNIVARIATE:
            deps['vanilla'] = self.raw(task, split, UNIVARIATE)
        if candidate_k(method):
            deps['retrieval'] = self.extraction(task, split, 'covariate')
        return deps

    def raw(self, task, split, method):
        return self.resolve(task, f'predictions/{split}', method, self.prediction_dependencies(task, split, method))

    def prediction(self, task, method):
        if method in self.controls:
            return self.resolve(task, 'predictions/test', method, self.control_dependencies(task, method, 'test'))
        return self.raw(task, 'test', method)

    def control_dependencies(self, task, method, split):
        alternative = self.controls[method]
        deps = {'data': self.prepared(task, split), 'vanilla': self.raw(task, split, UNIVARIATE),
                'alternative': self.raw(task, split, alternative)}
        if split == 'validation' and method.endswith('_per_variate'):
            deps['test_data'] = self.prepared(task, 'test')
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

    def reuse_univariate_rows(self, task, split, refs, current_run, values, fallback):
        """Copy completed forecasts for unchanged item/channel/origin references."""
        reused = np.zeros(len(refs), dtype=bool)
        sources = []
        current = current_run.manifest
        manifests = sorted(
            self.path(task, f'predictions/{split}', UNIVARIATE).glob('run_*/manifest.json'),
            key=lambda path: path.parent.name,
            reverse=True,
        )
        destination_rows = {tuple(map(int, row)): index for index, row in enumerate(refs)}
        for manifest_path in manifests:
            if manifest_path.parent == current_run.run_dir:
                continue
            manifest = load_manifest(manifest_path)
            metadata_path = manifest_path.parent / 'prediction.json'
            if (
                manifest['status'] != 'completed'
                or manifest.get('identity') != current.get('identity')
                or manifest.get('model_config') != current.get('model_config')
                or manifest.get('experiment_config') != current.get('experiment_config')
                or not metadata_path.is_file()
            ):
                continue
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if (
                metadata.get('method') != UNIVARIATE
                or metadata.get('split') != split
                or metadata.get('context_length') != self.context_length
            ):
                continue
            recorded = manifest.get('provenance', {}).get('upstream_manifests', {}).get('data')
            data_run = Path(recorded).parent if recorded else None
            if data_run is None or not data_run.is_dir():
                reference = manifest.get('pipeline_config', {}).get('dependencies', {}).get('data', {})
                run_name = reference.get('run') if isinstance(reference, dict) else None
                candidates = (
                    self.path(task, f'data/{split}', 'shared') / run_name,
                    self.path(task, 'data', 'shared') / run_name,
                ) if run_name else ()
                data_run = next((path for path in candidates if path.is_dir()), None)
            source_refs_path = data_run / f'{split}_references.npy' if data_run else None
            prediction_path = manifest_path.parent / 'prediction.npy'
            fallback_path = manifest_path.parent / 'fallback.npy'
            if not source_refs_path or not source_refs_path.is_file() or not prediction_path.is_file():
                continue
            source_refs = np.load(source_refs_path, mmap_mode='r', allow_pickle=False)
            predictions = np.load(prediction_path, mmap_mode='r', allow_pickle=False)
            source_fallback = (
                np.load(fallback_path, mmap_mode='r', allow_pickle=False)
                if fallback_path.is_file()
                else np.zeros(len(source_refs), dtype=bool)
            )
            copied = 0
            for source_index, reference in enumerate(source_refs):
                destination = destination_rows.get(tuple(map(int, reference)))
                if destination is None or reused[destination]:
                    continue
                values[destination] = predictions[source_index]
                fallback[destination] = source_fallback[source_index]
                reused[destination] = True
                copied += 1
            if copied:
                sources.append({'manifest': str(manifest_path), 'rows': copied})
            if reused.all():
                break
        return reused, sources

    def support(self, task, split, windows, refs):
        labels = windows.labels(refs)
        targets = np.isfinite(labels)
        if split == 'test':
            from timebench.evaluation.grid import flatten_univariate_grid, load_evaluation_grid
            from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
            expected, cells = flatten_univariate_grid(*load_evaluation_grid(
                resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate')))
            if not np.array_equal(expected, targets):
                raise ValueError('Official test references and Seasonal grid do not align')
            return expected, cells
        return targets, validation_window_mask(
            windows.histories(refs, self.context_length), labels
        )

    def prepare(self):
        for task in self.tasks:
            windows = Windows(task, self.storage)
            for split in ('validation', 'test'):
                with self.allocate(task, f'data/{split}', 'shared') as run:
                    if run.should_run:
                        log(f'prepare_{split} {task.dataset}/{task.term}')
                        self.finish(run, write_prepared(windows, run.run_dir, split))

    def extract(self, split):
        from timebench.proposal.retrieval import context_representation, blockwise_topk
        for task in self.tasks:
            data = self.prepared(task, split)
            for kind in ('covariate',):
                with self.allocate(task, f'retrieval/{split}', kind, {'data': data}) as run:
                    if not run.should_run:
                        continue
                    log(f'extract_{split} kind={kind} {task.dataset}/{task.term}')
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
                    k = max(self.k_values)
                    distances, ids = blockwise_topk(query, representations, refs, datastore_refs,
                        self.array(data, f'{split}_ticks'), self.array(data, f'{split}_datastore_ticks'),
                        self.array(data, f'{split}_datastore_ends'), k=k,
                        period=task.alignment_period, stride=task.datastore_stride, horizon=h,
                        scope=task.datastore_scope, minimum_overlap_fraction=self.config['minimum_overlap_fraction'],
                        query_block_size=self.config['query_block_size'], datastore_block_size=self.config['datastore_block_size'])
                    np.save(run.run_dir / 'neighbor_ids.npy', ids, allow_pickle=False)
                    np.save(run.run_dir / 'neighbor_distances.npy', distances, allow_pickle=False)
                    np.save(run.run_dir / 'eligible.npy', (ids >= 0).all(axis=-1), allow_pickle=False)
                    write_json(run.run_dir / 'retrieval.json', {'schema_version': 1, 'split': split,
                        'kind': kind, 'max_k': k, 'queries': len(refs), 'eligible_queries': int((ids >= 0).all(axis=-1).sum()),
                        'datastore_windows': len(datastore_refs), 'datastore_preprocessing_seconds': preprocessing_seconds,
                        'query_retrieval_seconds': perf_counter() - started})
                    self.finish(run, ['datastore_representation.npy', 'query_representation.npy', 'neighbor_ids.npy', 'neighbor_distances.npy', 'eligible.npy', 'retrieval.json'])

    def predict(self, split):
        from timebench.model_loading import load_forecaster
        from timebench.evaluation.timing import EvaluationTimer
        from timebench.pipeline.vanilla_reuse import load_vanilla_test_rows
        from timebench.proposal.retrieval import (
            retrieval_provenance_rows, summarize_retrieval_provenance,
        )
        model = None
        group = self.config['prediction_group']
        methods = [method for method in self.raw_methods
                   if group == 'all' or (method == UNIVARIATE) == (group == 'vanilla')]
        for task in self.tasks:
            data = self.prepared(task, split)
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
                    values = save_array(run.run_dir / 'prediction.npy', (len(refs), task.prediction_length))
                    values[:] = np.nan
                    fallback = np.zeros(len(refs), dtype=bool)
                    invalid_context = np.zeros(len(refs), dtype=bool)
                    produced = np.zeros(len(refs), dtype=bool)
                    reused = np.zeros(len(refs), dtype=bool)
                    reuse_sources = []
                    external_vanilla = None
                    if split == 'test' and method in (UNIVARIATE, MULTIVARIATE):
                        external_rows, external_vanilla = load_vanilla_test_rows(
                            self.vanilla_predictions_path,
                            backbone=self.model,
                            target_mode=('multivariate' if method == MULTIVARIATE else 'univariate'),
                            dataset=task.dataset,
                            term=task.term,
                            context_length=self.context_length,
                            prediction_length=task.prediction_length,
                            test_length=task.test_length,
                            references=refs,
                        )
                        if external_rows is not None:
                            values[:] = external_rows
                            reused[:] = True
                            produced[:] = True
                            reuse_sources.append({
                                'manifest': external_vanilla['manifest'],
                                'rows': int(len(refs)),
                                'kind': 'evaluating_tsfms_canonical_vanilla',
                            })
                    if method == UNIVARIATE and not reused.all():
                        reused, reuse_sources = self.reuse_univariate_rows(
                            task, split, refs, run, values, fallback
                        )
                        produced[reused] = True
                    vanilla = None if method == UNIVARIATE else self.array(deps['vanilla'], 'prediction')
                    if vanilla is not None and not reused.all():
                        values[:] = vanilla
                    extraction = deps.get('retrieval')
                    k = candidate_k(method)
                    retrieval = bool(k)
                    ids = self.array(extraction, 'neighbor_ids') if retrieval else None
                    datastore = self.array(data, f'{split}_datastore') if retrieval else None
                    if model is None and (~reused).any():
                        model = load_forecaster(self.model, self.weights, self.device, self.context_length)
                    timer = EvaluationTimer()
                    timer.start()
                    if method == MULTIVARIATE:
                        # Forecast each item/date once, then map channels to canonical univariate rows.
                        keys, inverse = np.unique(refs[:, (0, 2)], axis=0, return_inverse=True)
                        row_order = np.argsort(inverse, kind='stable')
                        offsets = np.r_[0, np.cumsum(np.bincount(inverse, minlength=len(keys)))]
                        for start in range(0, len(keys), self.batch_size):
                            groups, histories = [], []
                            for group in range(start, min(start + self.batch_size, len(keys))):
                                item, origin = keys[group]
                                positions = row_order[offsets[group]:offsets[group + 1]]
                                positions = positions[~reused[positions]]
                                if not len(positions):
                                    continue
                                history = windows.multivariate_history(item, origin, self.context_length)
                                if split == 'validation':
                                    finite_channels = finite_row_mask(history)
                                    row_context = finite_channels[refs[positions, 1]]
                                    invalid_context[positions[~row_context]] = True
                                    fallback[positions[~cells[positions]]] = True
                                    if not cells[positions].any() or not finite_channels.all():
                                        fallback[positions] = True
                                        continue
                                groups.append(group)
                                histories.append(history)
                            if not groups:
                                continue
                            forecasts = model.forecast(histories, task.prediction_length)
                            for group, forecast in zip(groups, forecasts):
                                positions = row_order[offsets[group]:offsets[group + 1]]
                                positions = positions[~reused[positions]]
                                if split == 'validation':
                                    positions = positions[cells[positions]]
                                values[positions] = forecast[refs[positions, 1]]
                                produced[positions] = True
                    else:
                        for start in range(0, len(refs), self.batch_size):
                            positions = np.arange(start, min(start + self.batch_size, len(refs)))
                            positions = positions[~reused[positions]]
                            if not len(positions):
                                continue
                            histories = windows.histories(refs[positions], self.context_length)
                            if split == 'validation':
                                labels = windows.labels(refs[positions])
                                usable_history = finite_row_mask(histories)
                                usable_window = validation_window_mask(histories, labels)
                                invalid_context[positions[~usable_history]] = True
                                fallback[positions[~usable_window]] = True
                                positions = positions[usable_window]
                                histories = [history for history, usable in zip(histories, usable_window) if usable]
                            if retrieval:
                                usable = (ids[positions] >= 0).all(axis=-1)
                                fallback[positions[~usable]] = True
                                positions = positions[usable]
                                histories = [history for history, keep in zip(histories, usable) if keep]
                            if not len(positions):
                                continue
                            past, future = None, None
                            if retrieval:
                                past, future = [], []
                                for row, history in zip(positions, histories):
                                    neighbors = windows.sequences(datastore[ids[row, :k or 1]])
                                    scaled = query_scaled_sequences(history[-task.retrieval_context_length:], neighbors, task.prediction_length)
                                    p, f = align_covariates(scaled, len(history), task.prediction_length)
                                    past.append(p)
                                    future.append(f)
                            forecasts = model.forecast(histories, task.prediction_length,
                                                       past_covariates=past, future_covariates=future)
                            values[positions] = np.stack([forecast[0] for forecast in forecasts])
                            produced[positions] = True
                    seconds = timer.stop()
                    required = targets & cells[:, None]
                    produced_nan_counts = (required & np.isnan(values)).sum(axis=1).astype(np.int64)
                    produced_nan_values = int(produced_nan_counts.sum())
                    invalid = cells & ~np.all(~targets | np.isfinite(values), axis=-1)
                    infinite = cells & np.any(targets & np.isinf(values), axis=-1)
                    if infinite.any():
                        raise ValueError(f'Infinite {method} forecast on required support')
                    if vanilla is not None:
                        values[invalid] = vanilla[invalid]
                        fallback[invalid] = True
                    finish_array(run.run_dir / 'prediction.npy', values)
                    np.save(run.run_dir / 'fallback.npy', fallback, allow_pickle=False)
                    np.save(run.run_dir / 'produced_nan_counts.npy', produced_nan_counts,
                            allow_pickle=False)
                    retrieval_metadata = json.loads((extraction / 'retrieval.json').read_text()) if retrieval else {}
                    prepared_metadata = json.loads((data / 'prepared.json').read_text(encoding='utf-8'))
                    usable_ticks = np.unique(self.array(data, f'{split}_ticks')[cells]) if len(refs) else []
                    fallback_summary = summarize_fallbacks(fallback, cells)
                    prediction_nan_values = int((required & np.isnan(values)).sum())
                    prediction_values = int(required.sum())
                    provenance_rows = None
                    retrieval_provenance = None
                    if retrieval:
                        provenance_rows = retrieval_provenance_rows(
                            refs,
                            datastore,
                            ids[:, :k or 1],
                            self.array(data, f'{split}_ticks'),
                            self.array(data, f'{split}_datastore_ticks'),
                        )
                        np.save(run.run_dir / 'retrieval_provenance.npy', provenance_rows,
                                allow_pickle=False)
                        retrieval_provenance = summarize_retrieval_provenance(provenance_rows)
                    write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'split': split,
                        'context_length': self.context_length, 'inference_seconds': seconds,
                        'query_retrieval_seconds': retrieval_metadata.get('query_retrieval_seconds', 0),
                        'datastore_preprocessing_seconds': retrieval_metadata.get('datastore_preprocessing_seconds', 0),
                        'fallback_count': fallback_summary['fallback_count'],
                        'grid_rows': fallback_summary['evaluated_rows'],
                        'validation_counts': ({**prepared_metadata['counts'],
                            'usable_rows': int(cells.sum()), 'usable_dates': int(len(usable_ticks))}
                            if split == 'validation' else None),
                        'reused_rows': int(reused.sum()),
                        'newly_inferred_rows': int((produced & ~reused).sum()),
                        'reuse_sources': reuse_sources,
                        'external_vanilla': external_vanilla,
                        'prediction_nan_values': prediction_nan_values,
                        'produced_nan_values': produced_nan_values,
                        'prediction_values': prediction_values,
                        'prediction_nan_rate': (prediction_nan_values / prediction_values
                                                if prediction_values else None),
                        'retrieval_provenance': retrieval_provenance,
                        'invalid_validation_context_count': int((invalid_context & cells).sum()),
                        'insufficient_retrieval_count': int((~(ids >= 0).all(axis=-1) & cells).sum()) if retrieval else 0,
                        'nonfinite_fallback_count': int(invalid.sum()),
                        'timing_policy': 'measured_candidate_query_loop_precomputed_vanilla_fallback_separate_retrieval'})
                    files = ['prediction.npy', 'fallback.npy', 'produced_nan_counts.npy',
                             'prediction.json']
                    if provenance_rows is not None:
                        files.append('retrieval_provenance.npy')
                    self.finish(run, files)
        del model

    def calibrate(self):
        from timebench.proposal.selection import (
            select_candidates, select_with_block_bootstrap, scope_ridge,
            win_frequency_mixture, win_frequency_mixtures,
        )
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
                    granularity = 'per_variate' if method.endswith('_per_variate') else 'task'
                    eligible = ~self.array(deps['alternative'], 'fallback')
                    if method.startswith('scope_selector'):
                        result = select_candidates(predictions, labels, scales, refs,
                            self.array(deps['data'], 'validation_ticks'), granularity=granularity, seed=self.seed,
                            prediction_length=task.prediction_length, validation_stride=task.validation_stride,
                            replications=self.config['bootstrap_replications'], block_length=self.config['bootstrap_block_length'])
                    elif method == 'scope_ridge':
                        result = scope_ridge(
                            predictions, labels, scales, eligible=eligible,
                            alpha=float(self.config['scope_ridge_alpha']),
                        )
                    elif granularity == 'per_variate':
                        result = win_frequency_mixtures(
                            predictions, labels, scales, refs, granularity=granularity,
                            eligible=eligible,
                        )
                    else:
                        result = {'schema_version': 1, 'granularity': 'task', 'selections': [
                            win_frequency_mixture(predictions, labels, scales, eligible=eligible)
                        ]}
                    if granularity == 'per_variate':
                        present = {(entry['item'], entry['channel']) for entry in result['selections']}
                        test_refs = self.array(deps['test_data'], 'test_references')
                        for item, channel in np.unique(test_refs[:, :2], axis=0):
                            key = int(item), int(channel)
                            if key in present:
                                continue
                            if method.startswith('scope_selector'):
                                entry = select_with_block_bootstrap(
                                    {UNIVARIATE: np.empty(0), alternative: np.empty(0)},
                                    seed=self.seed, prediction_length=task.prediction_length,
                                    validation_stride=task.validation_stride,
                                    replications=self.config['bootstrap_replications'],
                                    block_length=self.config['bootstrap_block_length'],
                                )
                                entry['validation_date_ticks'] = []
                            else:
                                entry = {
                                    'alternative': alternative,
                                    'rule': 'beta_1_1_validation_window_msse_win_frequency_half_ties',
                                    'trials': 0,
                                    'wins_including_half_ties': 0.0,
                                    'alternative_weight': 0.0,
                                    'fallback_reason': 'no_usable_validation_rows',
                                }
                            entry.update(item=key[0], channel=key[1])
                            result['selections'].append(entry)
                    write_json(run.run_dir / 'selection.json', result)
                    self.finish(run, ['selection.json'])

    def assemble(self):
        self.assemble_controls()

    def assemble_controls(self):
        from timebench.proposal.selection import blend
        from timebench.proposal.retrieval import summarize_retrieval_provenance
        for task in self.tasks:
            windows = Windows(task, self.storage)
            refs = self.array(self.prepared(task, 'test'), 'test_references')
            targets, cells = self.support(task, 'test', windows, refs)
            for method, alternative in self.controls.items():
                deps = self.control_dependencies(task, method, 'test')
                with self.allocate(task, 'predictions/test', method, deps) as run:
                    if not run.should_run:
                        continue
                    calibration = json.loads((deps['calibration'] / 'selection.json').read_text())
                    granularity = calibration['granularity']
                    vanilla = self.array(deps['vanilla'], 'prediction')
                    candidate = self.array(deps['alternative'], 'prediction')
                    values = save_array(run.run_dir / 'prediction.npy', vanilla.shape)
                    values[:] = vanilla
                    weights = np.zeros(len(refs), dtype=np.float64)
                    for entry in calibration['selections']:
                        positions = (np.flatnonzero(
                            (refs[:, :2] == (entry['item'], entry['channel'])).all(axis=-1)
                        ) if granularity == 'per_variate' else np.arange(len(refs)))
                        weight = (float(entry['selected_method'] == alternative)
                                  if method.startswith('scope_selector')
                                  else float(entry['alternative_weight']))
                        weights[positions] = weight
                        values[positions] = blend(vanilla[positions], candidate[positions], weight)
                    alternative_fallback = self.array(deps['alternative'], 'fallback')
                    fallback = np.asarray(alternative_fallback & (weights != 0), dtype=bool)
                    invalid = cells & ~np.all(~targets | np.isfinite(values), axis=-1)
                    values[invalid], fallback[invalid] = vanilla[invalid], True
                    finish_array(run.run_dir / 'prediction.npy', values)
                    np.save(run.run_dir / 'fallback.npy', fallback, allow_pickle=False)
                    np.save(run.run_dir / 'alternative_weight.npy', weights, allow_pickle=False)
                    provenance_rows = np.zeros((len(refs), 4), dtype=np.float64)
                    produced_nan_counts = np.zeros(len(refs), dtype=np.int64)
                    source = deps['alternative'] / 'retrieval_provenance.npy'
                    used = weights != 0
                    if used.any():
                        if source.is_file():
                            provenance_rows[used] = np.load(
                                source, mmap_mode='r', allow_pickle=False
                            )[used]
                        produced_nan_counts[used] = np.load(
                            deps['alternative'] / 'produced_nan_counts.npy', mmap_mode='r',
                            allow_pickle=False,
                        )[used]
                    np.save(run.run_dir / 'retrieval_provenance.npy', provenance_rows,
                            allow_pickle=False)
                    np.save(run.run_dir / 'produced_nan_counts.npy', produced_nan_counts,
                            allow_pickle=False)
                    write_json(run.run_dir / 'selection.json', calibration)
                    fallback_summary = summarize_fallbacks(fallback, cells)
                    write_json(run.run_dir / 'prediction.json', {'schema_version': 1, 'method': method, 'split': 'test',
                        'context_length': self.context_length, 'inference_seconds': None,
                        'timing_policy': 'assembled_from_candidate_artifacts_no_independent_selected_method_latency',
                        'alternative': alternative,
                        'alternative_weight': (float(weights[0]) if granularity == 'task' else None),
                        'weight_granularity': granularity,
                        'weight_min': float(weights.min()) if len(weights) else None,
                        'weight_max': float(weights.max()) if len(weights) else None,
                        'produced_nan_values': int(produced_nan_counts.sum()),
                        'retrieval_provenance': summarize_retrieval_provenance(provenance_rows),
                        'fallback_count': fallback_summary['fallback_count'],
                        'grid_rows': fallback_summary['evaluated_rows']})
                    self.finish(run, ['prediction.npy', 'fallback.npy', 'alternative_weight.npy',
                                      'retrieval_provenance.npy', 'produced_nan_counts.npy',
                                      'selection.json', 'prediction.json'])

    def methods(self):
        if self.model == 'chronos_bolt':
            return [UNIVARIATE]
        raw_baselines = [UNIVARIATE, *([MULTIVARIATE] if self.model == 'chronos2' else [])]
        return [*raw_baselines, *self.controls]

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
                    if upstream['status'] != 'completed' or self.dependency_reference(prediction) != expected:
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
        from timebench.pipeline.runtime_resources import log_selected_device
        selected_device = self.device if stage.startswith('predict') else 'cpu'
        log_selected_device(selected_device, stage=stage, component='selectime')
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
        elif stage in ('prepare', 'calibrate', 'assemble', 'evaluate', 'report'):
            getattr(self, stage)()
        else:
            raise ValueError(f'Unknown stage {stage}')
