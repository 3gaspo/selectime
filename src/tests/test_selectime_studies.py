"""Scientific contrasts for the independent Selectime studies (NumPy/pandas only)."""
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from timebench.data.windows import Task, Windows
from timebench.proposal.candidates import UNIVARIATE, MULTIVARIATE, retrieval_covariates
from timebench.proposal.retrieval import eligible, context_representation, squared_distance
from timebench.pipeline.studies import OracleWorkflow
from timebench.pipeline.timing import cold_sample


class StudyContrasts(unittest.TestCase):
    def test_same_user_keeps_other_channels_but_excludes_other_items(self):
        query = np.array([[0, 0, 20]])
        store = np.array([[0, 0, 8], [0, 1, 8], [1, 0, 8]])
        kwargs = dict(period=1, stride=1, horizon=2)
        user = eligible(query, store, np.array([20]), np.array([8, 8, 8]),
                        np.array([18, 18]), scope='same_user', **kwargs)
        series = eligible(query, store, np.array([20]), np.array([8, 8, 8]),
                          np.array([18, 18]), scope='same_series', **kwargs)
        np.testing.assert_array_equal(user, [[True, True, False]])
        np.testing.assert_array_equal(series, [[True, False, False]])

    def test_raw_scale_changes_the_nearest_neighbor(self):
        query = np.array([[0, 1, 2]], dtype=np.float32)
        candidates = np.array([[100, 101, 102], [0, 1, 3]], dtype=np.float32)
        raw = squared_distance(context_representation(query, normalize=False),
                               context_representation(candidates, normalize=False), 0.8)
        normalized = squared_distance(context_representation(query),
                                      context_representation(candidates), 0.8)
        self.assertEqual(int(raw.argmin()), 1)
        self.assertEqual(int(normalized.argmin()), 0)

    def test_no_query_scaling_retains_the_neighbor_units_and_future(self):
        history = np.array([0, 1, 2], dtype=np.float32)
        neighbors = np.array([[100, 101, 102, 103, 104]], dtype=np.float32)
        past, future = retrieval_covariates(history, neighbors, 3, 2, query_scaling=False)
        scaled_past, scaled_future = retrieval_covariates(history, neighbors, 3, 2)
        np.testing.assert_array_equal(past, [[100, 101, 102]])
        np.testing.assert_array_equal(future, [[103, 104]])
        np.testing.assert_allclose(scaled_past, [[0, 1, 2]], atol=1e-6)
        np.testing.assert_allclose(scaled_future, [[3, 4]], atol=1e-6)

    def test_dense_datastore_counts_match_enumerated_windows_and_stay_pretest(self):
        task = Task('toy/D', 'short', 2, 8, 8, 1, 4, 4, 4, 2)
        source = [{'start': '2000-01-01', 'freq': 'D', 'target': np.zeros((2, 28))}]
        aligned = Windows(task, 'unused', source=source)
        dense = Windows(replace(task, alignment_period=1, datastore_stride=1), 'unused', source=source)
        for split in ('validation', 'test'):
            refs = aligned.references(split)
            sizes = aligned.datastore_size(split, refs)
            original, _ = aligned.datastore(split, refs)
            full, _ = dense.datastore(split, refs)
            self.assertEqual(sizes['aligned_rows_before_cap'], len(original))
            self.assertEqual(sizes['full_stride_1_rows_before_cap'], len(full))
            self.assertTrue(np.all(full[:, 2] + task.prediction_length <= dense.boundary(0, split)))

    def test_oracle_dependencies_use_test_predictions_for_every_control(self):
        oracle = OracleWorkflow.__new__(OracleWorkflow)
        oracle.controls = {'scope_mix': MULTIVARIATE, 'top_k_5_mix': 'top_k_5'}
        oracle.source = SimpleNamespace(
            prepared=lambda task, split: f'data/{split}',
            raw=lambda task, split, method: f'{split}/{method}')
        for method, alternative in oracle.controls.items():
            deps = oracle.calibration_dependencies(None, method)
            self.assertEqual(deps['data'], 'data/test')
            self.assertEqual(deps['alternative'], f'test/{alternative}')
            self.assertEqual(deps['vanilla'], f'test/{UNIVARIATE}')

    def test_cold_batch_calls_no_prediction_or_neighbor_cache(self):
        task = Task('toy/D', 'short', 2, 4, 4, 1, 4, 1, 1, 2)
        windows = Windows(task, 'unused', source=[{
            'start': '2000-01-01', 'freq': 'D',
            'target': np.arange(28, dtype=np.float32)[None, :],
        }])
        refs = windows.references('test')[:1]
        workflow = SimpleNamespace(tasks=[task], storage='unused', weights='unused',
            seed=0, device='cpu', model='chronos2', context_length=8, controls={})
        calls = []

        class Model:
            def forecast(self, histories, horizon, **kwargs):
                calls.append(len(histories))
                return [np.repeat(history[-1], horizon).reshape(1, horizon) for history in histories]

        class Timer:
            def start(self): pass
            def stop(self): return 1.0

        request = {'config': {}, 'dataset': 'toy/D', 'term': 'short',
                   'method': UNIVARIATE, 'references': refs.tolist(),
                   'required_target_mask': [[True, True]]}
        with patch('timebench.pipeline.timing.Workflow', return_value=workflow), \
             patch('timebench.pipeline.timing.Windows', return_value=windows), \
             patch('timebench.pipeline.timing.seed_run'), \
             patch('timebench.pipeline.timing.EvaluationTimer', Timer), \
             patch('timebench.model_loading.load_forecaster', return_value=Model()), \
             patch('numpy.load', side_effect=AssertionError('candidate caches must not be loaded')):
            sample = cold_sample(request)
        self.assertEqual(calls, [1])
        self.assertEqual(sample['backbone_calls'], 1)
        self.assertEqual(sample['warmup_batches'], 0)
        self.assertEqual(sample['final_nan_values'], 0)


if __name__ == '__main__':
    unittest.main()
