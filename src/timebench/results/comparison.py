"""Matched TIME metrics, validation selections, coverage, and honest timings."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from timebench.proposal.candidates import candidate_names


def write_csv(path, rows):
    if not rows:
        raise ValueError('Cannot report an empty task plan')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average_statistics(rows):
    """Average declared task statistics; undefined latency stays undefined."""
    result = dict(rows[0])
    for key in result:
        values = [row[key] for row in rows]
        if all(value is None or isinstance(value, (int, float)) for value in values):
            result[key] = float(np.mean(values)) if all(value is not None for value in values) else None
        elif any(value != values[0] for value in values):
            raise ValueError(f'Cannot average incompatible report field {key}')
    return result


def aggregate_rows(rows):
    """Reduce exact repeats first, then equally weight scientific configurations."""
    by_display = defaultdict(lambda: defaultdict(list))
    for row, scientific in rows:
        key = (row['dataset'], row['term'], row['method'], row['report_label'])
        by_display[key][json.dumps(scientific, sort_keys=True)].append(row)
    return [average_statistics([average_statistics(repeats) for repeats in configurations.values()])
            for configurations in by_display.values()]


def build_report(inputs, destination, config):
    from timebench.pipeline.workflow import write_json, log
    from timebench.pipeline.evaluation_grid import resolve_shared_evaluation_grid
    rows, sources, selection_rows = [], [], []
    grouped = defaultdict(list)
    for task, method, evaluation, prediction, selection in inputs:
        summary = json.loads((evaluation / 'metrics_summary.json').read_text())
        metadata = json.loads((prediction / 'prediction.json').read_text())
        seasonal_root = resolve_shared_evaluation_grid(task.dataset, task.term, 'univariate').parent
        seasonal = json.loads((seasonal_root / 'metrics_summary.json').read_text())
        if summary['evaluation_grid']['definition'] != seasonal['evaluation_grid']['definition'] or summary['evaluation_grid']['valid_values'] != seasonal['evaluation_grid']['valid_values']:
            raise ValueError('Comparison requires the same Seasonal metric grid')
        label = method + selection['model_label'][len('chronos2'):]
        row = {'dataset': task.dataset, 'term': task.term, 'method': method, 'report_label': label,
               'inference_seconds': summary.get('inference_seconds'),
               'query_retrieval_seconds': metadata.get('query_retrieval_seconds'),
               'datastore_preprocessing_seconds': metadata.get('datastore_preprocessing_seconds'),
               'timing_policy': metadata['timing_policy'], 'fallback_count': metadata['fallback_count'],
               'grid_rows': metadata['grid_rows'],
               'fallback_rate': metadata['fallback_count'] / metadata['grid_rows'] if metadata['grid_rows'] else None}
        for metric, values in summary['metrics'].items():
            if values['finite_values'] < seasonal['metrics'][metric]['finite_values']:
                raise ValueError(f'{task.dataset}/{task.term}/{method}: lost {metric} coverage')
            for field in ('mean', 'variance', 'std', 'dispersion_ddof', 'finite_values', 'evaluation_values', 'total_values'):
                row[f'{metric}_{field}'] = values[field]
        mase, baseline = summary['metrics']['MASE'], seasonal['metrics']['MASE']
        scale = baseline['mean']
        variance = baseline.get('variance')
        row.update(seasonal_MASE_mean=scale, seasonal_MASE_variance=variance,
                   scaled_MASE_mean=mase['mean'] / scale if scale and mase['mean'] is not None else None,
                   scaled_MASE_std=mase['std'] / scale if scale and mase['std'] is not None else None,
                   scaled_MASE_variance=mase['variance'] / scale ** 2 if scale and mase['variance'] is not None else None,
                   MASE_variance_ratio_to_seasonal=mase['variance'] / variance if variance and mase['variance'] is not None else None)
        rows.append((row, selection['scientific_config']))
        sources.append({'dataset': task.dataset, 'term': task.term, 'method': method,
                        'evaluation_manifest': str(evaluation / 'manifest.json'),
                        'prediction_manifest': str(prediction / 'manifest.json'),
                        'seasonal_manifest': str(seasonal_root / 'manifest.json'), 'selection': selection})
        if method.startswith('selected_'):
            selection = json.loads((prediction / 'selection.json').read_text())
            for entry in selection['selections']:
                selection_rows.append({'dataset': task.dataset, 'term': task.term, 'selector': method,
                    'report_label': label, 'evaluation_run': evaluation.name,
                    'item': entry.get('item'), 'channel': entry.get('channel'),
                    'selected_method': entry['selected_method'], 'validation_dates': entry['validation_dates'],
                    'observed_best': entry.get('observed_best'), 'block_length': entry['block_length'],
                    'fallback_reason': entry.get('fallback_reason')})
    rows = aggregate_rows(rows)
    for row in rows:
        grouped[row['report_label']].append(row)
    write_csv(destination / 'comparison.csv', rows)
    write_csv(destination / 'selections.csv', selection_rows)
    selection_summary = []
    for selector in ('selected_task', 'selected_per_variate'):
        selected = [row['selected_method'] for row in selection_rows if row['selector'] == selector]
        counts = Counter(selected)
        for method in candidate_names(config['k_values']):
            selection_summary.append({'selector': selector, 'candidate': method, 'count': counts[method],
                                      'total_selections': len(selected),
                                      'rate': counts[method] / len(selected) if selected else None})
    write_csv(destination / 'selection_summary.csv', selection_summary)
    overall = {}
    for method, values in grouped.items():
        mase = [row['MASE_mean'] for row in values if row['MASE_mean'] is not None]
        scaled = [row['scaled_MASE_mean'] for row in values if row['scaled_MASE_mean'] is not None]
        timings = [row['inference_seconds'] for row in values]
        counts, support = sum(row['fallback_count'] for row in values), sum(row['grid_rows'] for row in values)
        overall[method] = {'tasks': len(values), 'mean_task_MASE': float(np.mean(mase)) if mase else None,
            'mean_task_scaled_MASE': float(np.mean(scaled)) if scaled else None,
            'summed_inference_seconds': sum(timings) if all(value is not None for value in timings) else None,
            'fallback_count': counts, 'grid_rows': support, 'pooled_fallback_rate': counts / support if support else None}
        log(f'{method}: tasks={len(values)} mean_task_MASE={overall[method]["mean_task_MASE"]}')
    write_json(destination / 'comparison_summary.json', overall)
    write_json(destination / 'report_manifest.json', {'schema_version': 1, 'experiment': 'selectime',
        'requested_config': config, 'inputs': sources,
        'selection': {key: config[key] for key in
            ('report_current_config', 'report_config_filters', 'report_config_policy', 'report_repeat_policy')},
        'aggregation': 'average_exact_repeat_statistics_then_average_configuration_statistics',
        'selection_frequencies': 'observed_choices_in_selected_input_manifests',
        'task_dispersion': 'population_variance_over_finite_series_window_variate_cells',
        'selected_method_timing': 'unmeasured_not_sum_of_all_candidate_search_costs'})
    return ['comparison.csv', 'selections.csv', 'selection_summary.csv', 'comparison_summary.json', 'report_manifest.json']
