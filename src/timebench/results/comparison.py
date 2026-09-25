"""Matched TIME metrics, validation selections, coverage, and honest timings."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import numpy as np
from timebench.proposal.candidates import UNIVARIATE, MULTIVARIATE
from timebench.results.performance import write_performance_report


def write_csv(path, rows, fieldnames=None):
    if not rows and fieldnames is None:
        raise ValueError('Cannot report an empty task plan')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def average_statistics(rows):
    """Average declared task statistics; undefined latency stays undefined."""
    result = dict(rows[0])
    for key in result:
        values = [row[key] for row in rows]
        if all(value is None or isinstance(value, (int, float)) for value in values):
            result[key] = float(np.nanmean(values)) if all(value is not None for value in values) else None
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
        label = method + selection['model_label'][len(config['model']):]
        provenance = metadata.get('retrieval_provenance') or {}
        prediction_outputs = summary.get('prediction_outputs', {})
        row = {'dataset': task.dataset, 'term': task.term, 'method': method, 'report_label': label,
               'inference_seconds': summary.get('inference_seconds'),
               'query_retrieval_seconds': metadata.get('query_retrieval_seconds'),
               'datastore_preprocessing_seconds': metadata.get('datastore_preprocessing_seconds'),
               'timing_policy': metadata['timing_policy'], 'fallback_count': metadata['fallback_count'],
               'alternative': metadata.get('alternative'), 'alternative_weight': metadata.get('alternative_weight'),
               'grid_rows': metadata['grid_rows'],
               'prediction_nan_values': prediction_outputs.get('evaluation_nan_values'),
               'produced_nan_values': metadata.get('produced_nan_values'),
               'prediction_values': prediction_outputs.get('evaluation_values'),
               'prediction_nan_rate': (
                   prediction_outputs.get('evaluation_nan_values', 0)
                   / prediction_outputs['evaluation_values']
                   if prediction_outputs.get('evaluation_values') else None),
               'retrieval_extractions': provenance.get('retrieval_extractions', 0),
               'retrieved_neighbors': provenance.get('retrieved_neighbors', 0),
               'same_user_neighbors': provenance.get('same_user_neighbors', 0),
               'same_user_retrieval_percentage': provenance.get('same_user_retrieval_percentage'),
               'normalized_time_distance_sum': provenance.get('normalized_time_distance_sum', 0),
               'average_normalized_time_distance_to_query': provenance.get('average_normalized_time_distance_to_query'),
               'fallback_rate': metadata['fallback_count'] / metadata['grid_rows'] if metadata['grid_rows'] else None}
        for metric, values in summary['metrics'].items():
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
        if method.startswith('scope_selector'):
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
    write_csv(destination / 'selections.csv', selection_rows, fieldnames=[
        'dataset', 'term', 'selector', 'report_label', 'evaluation_run', 'item', 'channel',
        'selected_method', 'validation_dates', 'observed_best', 'block_length', 'fallback_reason'])
    selection_summary = []
    for selector in ('scope_selector', 'scope_selector_per_variate'):
        selected = [row['selected_method'] for row in selection_rows if row['selector'] == selector]
        counts = Counter(selected)
        for method in (UNIVARIATE, MULTIVARIATE):
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
        neighbors = sum(row['retrieved_neighbors'] for row in values)
        same_user = sum(row['same_user_neighbors'] for row in values)
        distance_sum = sum(row['normalized_time_distance_sum'] for row in values)
        prediction_nans = sum(row['prediction_nan_values'] or 0 for row in values)
        prediction_values = sum(row['prediction_values'] or 0 for row in values)
        overall[method] = {'tasks': len(values), 'mean_task_MASE': float(np.nanmean(mase)) if mase else None,
            'mean_task_scaled_MASE': float(np.nanmean(scaled)) if scaled else None,
            'summed_inference_seconds': sum(timings) if all(value is not None for value in timings) else None,
            'fallback_count': counts, 'grid_rows': support, 'pooled_fallback_rate': counts / support if support else None,
            'prediction_nan_values': prediction_nans, 'prediction_values': prediction_values,
            'prediction_nan_rate': prediction_nans / prediction_values if prediction_values else None,
            'produced_nan_values': sum(row['produced_nan_values'] or 0 for row in values),
            'retrieval_extractions': sum(row['retrieval_extractions'] for row in values),
            'retrieved_neighbors': neighbors,
            'same_user_retrieval_percentage': 100 * same_user / neighbors if neighbors else None,
            'average_normalized_time_distance_to_query': distance_sum / neighbors if neighbors else None}
        log(f'{method}: tasks={len(values)} mean_task_MASE={overall[method]["mean_task_MASE"]}')
    write_json(destination / 'comparison_summary.json', overall)
    horizons = {(task.dataset, task.term): task.prediction_length for task, *_ in inputs}
    performance_rows = [{
        'model': row['report_label'], 'dataset': row['dataset'].rsplit('/', 1)[0],
        'frequency': row['dataset'].rsplit('/', 1)[1], 'term': row['term'],
        'horizon_steps': horizons[row['dataset'], row['term']],
        'MASE': row['MASE_mean'], 'scaled_MASE': row['scaled_MASE_mean'],
        'MASE_std': row['MASE_std'], 'MASE_variance': row['MASE_variance'],
        'seasonal_MASE_variance': row['seasonal_MASE_variance'],
        'inference_seconds': row['inference_seconds'],
        'query_retrieval_seconds': row['query_retrieval_seconds'],
        'datastore_preprocessing_seconds': row['datastore_preprocessing_seconds'],
        'prediction_nan_values': row['prediction_nan_values'],
        'prediction_values': row['prediction_values'],
        'prediction_nan_rate': row['prediction_nan_rate'],
    } for row in rows]
    reference_labels = {row['report_label'] for row in rows if row['method'] == 'vanilla_univariate'}
    performance_artifacts = write_performance_report(
        performance_rows, destination / 'performance',
        reference=next(iter(reference_labels)) if len(reference_labels) == 1 else None,
        scaled_aggregation='arithmetic', inputs=sources)
    performance_files = [path.relative_to(destination).as_posix() for path in performance_artifacts]
    write_json(destination / 'report_manifest.json', {'schema_version': 1, 'experiment': 'selectime',
        'performance_artifacts': performance_files,
        'requested_config': config, 'inputs': sources,
        'selection': {key: config[key] for key in
            ('report_current_config', 'report_config_filters', 'report_config_policy', 'report_repeat_policy')},
        'aggregation': 'average_exact_repeat_statistics_then_average_configuration_statistics',
        'selection_frequencies': 'observed_choices_in_selected_input_manifests',
        'retrieval_provenance': {
            'same_user': 'same dataset item/user as the query',
            'normalized_time_distance': '(query_tick-neighbor_tick)/(query_tick-earliest_datastore_tick)',
            'extraction_unit': 'one query retrieval',
        },
        'task_dispersion': 'population_variance_over_finite_series_window_variate_cells',
        'selected_method_timing': 'unmeasured_not_sum_of_all_candidate_search_costs'})
    return ['comparison.csv', 'selections.csv', 'selection_summary.csv', 'comparison_summary.json',
            'report_manifest.json', *performance_files]
