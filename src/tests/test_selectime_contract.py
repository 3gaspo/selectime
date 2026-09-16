"""Lightweight scientific regressions: NumPy/pandas/PyYAML and stdlib only."""
from dataclasses import replace
from pathlib import Path
import ast
import unittest
import numpy as np
import yaml
from timebench.data.windows import Task, Windows, aligned_origins
from timebench.proposal.candidates import (UNIVARIATE, MULTIVARIATE, SELF_AUGMENTATION,
    candidate_names, self_covariates, query_scaled_sequences, align_covariates)
from timebench.proposal.retrieval import blockwise_topk, context_representation, eligible
from timebench.proposal.selection import date_losses, row_msse, select_candidates, select_with_block_bootstrap
from timebench.results.comparison import aggregate_rows
from timebench.pipeline.runs import allocate_run, select_completed_runs, set_selected_run, ManifestError
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def task(**overrides):
    return replace(Task('synthetic/D', 'short', 4, 24, 20, 2, 8, 2, 4, 3), **overrides)


def windows(selected=None):
    targets = [np.stack((np.arange(100), 2 * np.arange(100) + 5)),
               np.stack((np.arange(100) + 2, 3 * np.arange(100) + 7))]
    source = [{'target': values, 'start': '2020-01-01', 'freq': 'D', 'item_id': str(item)}
              for item, values in enumerate(targets)]
    return Windows(selected or task(), Path('unused'), source=source)


class DatastoreTests(unittest.TestCase):
    def test_boundaries_alignment_and_expansion(self):
        data = windows()
        validation, test = data.references('validation'), data.references('test')
        np.testing.assert_array_equal(test[:6, 2], [76, 80, 84, 88, 92, 96])
        ds_validation, ends_validation = data.datastore('validation', validation)
        ds_test, ends_test = data.datastore('test', test)
        self.assertTrue((ds_validation[:, 2] + 4 <= 56).all())
        self.assertTrue((ds_test[:, 2] + 4 <= 76).all())
        self.assertGreater(ds_test[:, 2].max(), ds_validation[:, 2].max())
        allowed = eligible(test, ds_test, data.ticks(test), data.ticks(ds_test), ends_test,
                           period=2, stride=4, horizon=4, scope='all')
        q = data.ticks(test)[:, None]
        d = data.ticks(ds_test)[None]
        self.assertTrue((((q - d) % 2 == 0) | ~allowed).all())
        self.assertTrue(((d + 4 <= q) | ~allowed).all())
        self.assertTrue((ends_test > ends_validation).all())

    def test_cap_and_same_series_scope(self):
        data = windows(task(max_datastore_windows=20))
        refs = data.references('test')
        datastore, ends = data.datastore('test', refs)
        self.assertLessEqual(len(datastore), 20)
        counts = [((datastore[:, :2] == key).all(axis=1)).sum() for key in np.unique(datastore[:, :2], axis=0)]
        self.assertEqual(counts, [5, 5, 5, 5])
        allowed = eligible(refs, datastore, data.ticks(refs), data.ticks(datastore), ends,
                           period=2, stride=4, horizon=4, scope='same_series')
        same = (refs[:, None, :2] == datastore[None, :, :2]).all(axis=-1)
        self.assertTrue((same | ~allowed).all())

    def test_sampling_multiple_calendar_ticks(self):
        source = [{'target': np.arange(100), 'start': '2020-01-01 00:00', 'freq': '15T'},
                  {'target': np.arange(100), 'start': '2020-01-01 01:00', 'freq': '15T'}]
        data = Windows(task(), Path('unused'), source=source)
        self.assertEqual(int(data.start_ticks[1] - data.start_ticks[0]), 4)

    def test_no_validation_does_not_change_official_test(self):
        np.testing.assert_array_equal(windows().references('test'), windows(task(validation_length=0)).references('test'))
        self.assertEqual(len(windows(task(validation_length=0)).references('validation')), 0)


class RetrievalTests(unittest.TestCase):
    def test_instance_representation_and_no_future_statistics(self):
        history = np.array([1, 2, 3, 4], dtype=np.float32)
        np.testing.assert_allclose(context_representation(history), context_representation(5 * history + 100), atol=1e-6)
        neighbors = np.array([[10, 20, 30, 40, 50, 60]], dtype=np.float32)
        scaled = query_scaled_sequences(history, neighbors, 2)
        np.testing.assert_allclose(scaled, [[1, 2, 3, 4, 5, 6]], atol=1e-6)
        changed = neighbors.copy()
        changed[:, -2:] = 10000
        np.testing.assert_allclose(query_scaled_sequences(history, changed, 2)[:, :-2], scaled[:, :-2])
        past, future = align_covariates(scaled, 6, 2)
        self.assertTrue(np.isnan(past[:, :2]).all())
        np.testing.assert_allclose(past[:, 2:], [[1, 2, 3, 4]], atol=1e-6)
        np.testing.assert_allclose(future, [[5, 6]], atol=1e-6)

    def test_ordered_prefix_and_complete_maximum_k(self):
        query = np.array([[0, 1]], dtype=np.float32)
        datastore = np.array([[0, 1], [1, 2], [2, 3]], dtype=np.float32)
        refs = np.array([[0, 0, 10]])
        drefs = np.array([[0, 0, 3], [0, 1, 4], [1, 0, 5]])
        distances, ids = blockwise_topk(query, datastore, refs, drefs, np.array([10]), np.array([3, 4, 5]),
            np.array([6, 6]), k=3, period=1, stride=1, horizon=2, datastore_block_size=2)
        np.testing.assert_array_equal(ids, [[0, 1, 2]])
        self.assertTrue((np.diff(distances[0]) >= 0).all())
        _, incomplete = blockwise_topk(query, datastore, refs, drefs, np.array([10]), np.array([3, 4, 5]),
            np.array([6, 6]), k=3, period=1, stride=1, horizon=2, scope='same_series')
        np.testing.assert_array_equal(incomplete, [[-1, -1, -1]])

    def test_negative_target_self_augmentation(self):
        np.testing.assert_allclose(self_covariates([-4, 0, 9]), [[2, 0, 3], [-1, 0, 1]])


class SelectionTests(unittest.TestCase):
    def test_task_and_variate_selectors_differ(self):
        refs = np.array([[0, channel, date] for channel in range(2) for date in range(4)])
        ticks = refs[:, 2]
        labels = np.zeros((8, 2))
        predictions = {UNIVARIATE: np.repeat([[0], [0], [0], [0], [2], [2], [2], [2]], 2, axis=1),
                       MULTIVARIATE: np.repeat([[3], [3], [3], [3], [0], [0], [0], [0]], 2, axis=1)}
        kwargs = dict(seed=0, prediction_length=2, validation_stride=2, replications=100)
        by_task = select_candidates(predictions, labels, np.ones(8), refs, ticks, granularity='task', **kwargs)
        by_variate = select_candidates(predictions, labels, np.ones(8), refs, ticks, granularity='per_variate', **kwargs)
        self.assertEqual(by_task['selections'][0]['selected_method'], UNIVARIATE)
        self.assertEqual([entry['selected_method'] for entry in by_variate['selections']], [UNIVARIATE, MULTIVARIATE])

    def test_shared_support_date_weight_and_scaled_loss(self):
        predictions = {UNIVARIATE: np.array([[1, 99], [3, 3], [2, 2]]),
                       SELF_AUGMENTATION: np.array([[1, 0], [3, 3], [2, 2]])}
        labels = np.array([[0, np.nan], [0, 0], [0, 0]])
        losses = row_msse(predictions, labels, np.array([1, 9, 1]))
        np.testing.assert_allclose(losses[UNIVARIATE], [1, 1, 4])
        dates, values = date_losses(losses, np.array([1, 1, 2]), np.arange(3))
        np.testing.assert_array_equal(dates, [1, 2])
        np.testing.assert_allclose(values[UNIVARIATE], [1, 4])

    def test_bootstrap_ties_seed_and_empty_validation(self):
        losses = {method: np.ones(20) for method in candidate_names([1, 5, 10, 15])}
        kwargs = dict(seed=4, prediction_length=8, validation_stride=3, replications=100)
        result = select_with_block_bootstrap(losses, **kwargs)
        self.assertEqual(result['selected_method'], UNIVARIATE)
        self.assertEqual(result['block_length'], 3)
        self.assertEqual(result, select_with_block_bootstrap(losses, **kwargs))
        empty = select_with_block_bootstrap({UNIVARIATE: np.empty(0)}, **kwargs)
        self.assertEqual(empty['fallback_reason'], 'no_usable_validation_dates')


class SourceContracts(unittest.TestCase):
    def test_cluster_headers_and_script_ownership(self):
        required = {'--gres': 'gpu:1', '--partition': 'an', '--qos': 'an_preemptable',
                    '--nodes': '1', '--ntasks': '1', '--cpus-per-task': '8', '--mem': '80000',
                    '--time': '23:00:00', '--wckey': 'P12CU:DATASCIENCE'}
        for name in ('selectime_selena.slurm', 'seasonal_naive_selena.slurm'):
            text = (ROOT / name).read_text()
            directives = dict(line.removeprefix('#SBATCH ').split('=', 1)
                              for line in text.splitlines() if line.startswith('#SBATCH ') and '=' in line)
            for key, value in required.items():
                self.assertEqual(directives[key], value)
            self.assertTrue(directives['--job-name'].startswith('s'))
            self.assertIn('#SBATCH --exclusive', text)
            self.assertNotIn('--no-requeue', text)
            for key in ('--output', '--error'):
                self.assertTrue(directives[key].startswith('/scratch/users/%u/codes/selectime/logs/'))
            self.assertIn('PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"', text)
            self.assertIn('selena_', text)
        for name in ('submit_experiment.sh', 'submit_seasonal_naive.sh'):
            self.assertTrue((ROOT / 'scripts' / name).is_file())
            self.assertFalse((ROOT / name).exists())
        for name in ('selectime.slurm', 'seasonal_naive.slurm'):
            text = (ROOT / name).read_text()
            self.assertIn('#SBATCH --gres=gpu:1', text)
            self.assertIn('#SBATCH --ntasks=1', text)
        for name in ('run_experiment.sh', 'run_seasonal.sh'):
            text = (ROOT / 'src/slurm' / name).read_text()
            self.assertIn('srun --ntasks=1', text)
            self.assertIn('src/scripts/', text)
            self.assertIn('finalize_stage', text)
        workflow = (ROOT / 'src/slurm/run_experiment.sh').read_text()
        self.assertIn('groups=(vanilla remaining)', workflow)
        self.assertIn('"prediction_group=$group"', workflow)
        self.assertLess(workflow.index('srun --ntasks=1'), workflow.index('finalize_stage', workflow.index('srun --ntasks=1')))

    def test_grid_and_excluded_implementation(self):
        config = yaml.safe_load((ROOT / 'src/timebench/conf/experiment.yaml').read_text())
        datasets = yaml.safe_load((ROOT / 'src/timebench/config/datasets.yaml').read_text())
        plan = [(name, term) for name, settings in datasets['datasets'].items()
                if name not in config['excluded_datasets'] for term in config['terms'] if term in settings]
        self.assertEqual(len(plan), 90)
        self.assertEqual({name for name, _ in plan}, set(datasets['selectime_tasks']))
        for settings in datasets['selectime_tasks'].values():
            self.assertNotIn('fitting_stride', settings)
            self.assertNotIn('rolling_fitting_stride', settings)
        self.assertEqual(config['model'], 'chronos2')
        self.assertEqual(config['datastore_scope'], 'all')
        self.assertIsNone(config['max_datastore_windows'])
        for name in ('ridge.py', 'adaptime_training.py', 'adaptime_rolling.py', 'tsrag.py'):
            self.assertFalse(list((ROOT / 'src').rglob(name)))

    def test_syntax_and_proposal_dependency_direction(self):
        for path in (ROOT / 'src').rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for path in (ROOT / 'src/timebench/proposal').glob('*.py'):
            tree = ast.parse(path.read_text())
            imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any(name.startswith(('timebench.pipeline', 'timebench.results', 'hydra')) for name in imports))


class ReportingContracts(unittest.TestCase):
    def test_average_repeats_before_configurations(self):
        def row(value):
            return {'dataset': 'synthetic/D', 'term': 'short', 'method': UNIVARIATE,
                    'report_label': UNIVARIATE, 'MASE_mean': value, 'inference_seconds': None}
        result = aggregate_rows([(row(0), {'cap': None}), (row(2), {'cap': None}), (row(9), {'cap': 10})])
        self.assertEqual(result[0]['MASE_mean'], 5)
        self.assertIsNone(result[0]['inference_seconds'])

    def test_policies_and_selected_repeat(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {'model': 'chronos2', 'target_mode': 'univariate', 'dataset': 'synthetic',
                        'frequency': 'D', 'term': 'short', 'method': UNIVARIATE}
            paths = []
            for stride in (1, 1, 2):
                with allocate_run(root, experiment='selectime', identity=identity, model_config={'method': UNIVARIATE},
                        pipeline_config={'stride': stride}, runtime_config={}, experiment_config={'seed': 0},
                        policy='new') as run:
                    (run.run_dir / 'prediction.json').write_text('{}')
                    run.complete(['prediction.json'])
                    paths.append(run.run_dir)
            with self.assertRaises(ManifestError):
                select_completed_runs(root, config_policy='error')
            self.assertEqual(len(select_completed_runs(root, config_policy='distinct', repeat_policy='average')), 3)
            self.assertEqual(len(select_completed_runs(root, config_policy='latest', repeat_policy='latest')), 1)
            set_selected_run(paths[0])
            selected = select_completed_runs(root, config_filters={'pipeline_config.stride': 1}, repeat_policy='selected')
            self.assertEqual(selected[0][0], paths[0])


if __name__ == '__main__':
    unittest.main()
