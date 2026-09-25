"""Produce Seasonal Naive with the current Improved TIME shared-grid identity."""
import hydra
from omegaconf import OmegaConf
from timebench.scripts.entry import configure_paths

configure_paths()


@hydra.main(version_base=None, config_path='../timebench/conf', config_name='experiment')
def main(config):
    import os
    import numpy as np
    from pathlib import Path
    from timebench.pipeline.workflow import Workflow, log
    from timebench.pipeline.runs import allocate_run
    from timebench.evaluation.data import Dataset, load_dataset_config
    from timebench.evaluation.grid import EVALUATION_GRID_DEFINITION, EVALUATION_GRID_FILE
    from timebench.evaluation.metrics import seasonal_naive_point_forecast
    from timebench.evaluation.saver import save_window_predictions
    from timebench.evaluation.timing import EvaluationTimer
    from timebench.pipeline.runtime_resources import log_selected_device

    log_selected_device('cpu', stage='forecast', model='seasonal_naive')
    workflow = Workflow(OmegaConf.to_container(config, resolve=True))
    settings = load_dataset_config(workflow.config_path)
    tasks_root = Path(os.environ['TIME_SEASONAL_TASKS_ROOT'])
    for task in workflow.tasks:
        dataset = Dataset(task.dataset, term=task.term, prediction_length=task.prediction_length,
                          test_length=task.test_length, storage_path=workflow.storage)
        val_length = settings['datasets'][task.dataset].get('val_length')
        with allocate_run(tasks_root / 'seasonal_naive/univariate' / task.dataset / task.term,
            experiment='foundation_models',
            identity={'model': 'seasonal_naive', 'target_mode': 'univariate',
                      'dataset': task.dataset.rpartition('/')[0], 'frequency': dataset.freq, 'term': task.term},
            model_config={'quantile_levels': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            pipeline_config={'prediction_length': task.prediction_length, 'test_length': task.test_length,
                             'val_length': val_length, 'windows': dataset.windows, 'seasonality': task.seasonality,
                             'evaluation_grid': EVALUATION_GRID_DEFINITION},
            runtime_config={'device': 'cpu'}, experiment_config={'covariate_mode': 'none', 'covariate_channels': 0},
            provenance={'dataset_config_path': str(workflow.config_path)}) as run:
            if not run.should_run:
                continue
            log(f'Seasonal {task.dataset}/{task.term}')
            timer = EvaluationTimer()
            timer.start()
            forecasts = np.stack([seasonal_naive_point_forecast(np.asarray(entry['target']), task.prediction_length,
                                                               task.seasonality) for entry in dataset.test_data.input])
            seconds = timer.stop()
            levels = run.manifest['model_config']['quantile_levels']
            save_window_predictions(dataset, np.repeat(forecasts[:, None, :], len(levels), axis=1),
                f'{task.dataset}/{task.term}', str(tasks_root), seasonality=task.seasonality,
                quantile_levels=levels, task_output_dir=str(run.run_dir), create_evaluation_grid=True,
                inference_seconds=seconds, model_hyperparams={'model': 'seasonal_naive', 'experiment': 'foundation_models',
                    'target_mode': 'univariate', 'season_length': task.seasonality, 'covariate_mode': 'none', 'covariate_channels': 0})
            workflow.finish(run, ['predictions.npz', 'metrics.npz', 'metrics_summary.json', 'config.json', EVALUATION_GRID_FILE])


if __name__ == '__main__':
    main()
