"""Instance-normalized exact top-K search, adapted from Adaptime's retrieval."""
import warnings
import numpy as np


def retrieval_provenance_rows(query_refs, datastore_refs, neighbor_ids,
                              query_ticks, datastore_ticks):
    """Per-query counts: extractions, neighbors, same-user neighbors, lag sum."""
    query_refs = np.asarray(query_refs)
    datastore_refs = np.asarray(datastore_refs)
    neighbor_ids = np.asarray(neighbor_ids)
    query_ticks = np.asarray(query_ticks)
    datastore_ticks = np.asarray(datastore_ticks)
    rows = np.zeros((len(query_refs), 4), dtype=np.float64)
    rows[:, 0] = 1
    if not len(datastore_refs):
        return rows
    earliest = float(np.min(datastore_ticks))
    for row, ids in enumerate(neighbor_ids):
        valid = ids[ids >= 0]
        if not len(valid):
            continue
        neighbors = datastore_refs[valid]
        rows[row, 1] = len(valid)
        rows[row, 2] = np.sum(neighbors[:, 0] == query_refs[row, 0])
        denominator = max(float(query_ticks[row]) - earliest, 1.0)
        rows[row, 3] = np.sum(
            (float(query_ticks[row]) - datastore_ticks[valid]) / denominator
        )
    return rows


def summarize_retrieval_provenance(rows):
    rows = np.asarray(rows, dtype=np.float64)
    totals = rows.sum(axis=0) if len(rows) else np.zeros(4, dtype=np.float64)
    extractions, neighbors, same_user, distance_sum = totals
    return {
        'retrieval_extractions': int(extractions),
        'retrieved_neighbors': int(neighbors),
        'same_user_neighbors': int(same_user),
        'same_user_retrieval_percentage': (
            float(100 * same_user / neighbors) if neighbors else None
        ),
        'normalized_time_distance_sum': float(distance_sum),
        'average_normalized_time_distance_to_query': (
            float(distance_sum / neighbors) if neighbors else None
        ),
        'same_user_definition': 'same dataset item/user as the query',
        'normalized_time_distance_definition': (
            '(query_tick-neighbor_tick)/(query_tick-earliest_datastore_tick)'
        ),
    }


def context_representation(context):
    """Normalize each lookback independently, retaining missing positions."""
    values = np.asarray(context, dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError('Retrieval contexts must not contain infinities')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        mean = np.nanmean(values, axis=-1, keepdims=True)
        scale = np.maximum(np.nanstd(values, axis=-1, keepdims=True), 1e-8)
    return np.ascontiguousarray((values - mean) / scale)


def eligible(query_refs, datastore_refs, query_ticks, datastore_ticks, ends, *, period, stride, horizon, scope):
    """Both fixed boundary and query causality use actual calendar ticks."""
    last = np.minimum(query_ticks[:, None] - horizon, ends[datastore_refs[:, 0]][None, :])
    last = last - (last - query_ticks[:, None]) % period
    origins = datastore_ticks[None, :]
    allowed = (origins <= last) & ((last - origins) % stride == 0)
    if scope == 'same_series':
        allowed &= ((query_refs[:, None, :2] == datastore_refs[None, :, :2]).all(axis=-1))
    elif scope != 'all':
        raise ValueError('scope must be all or same_series')
    return allowed


def squared_distance(query, datastore, minimum_overlap_fraction):
    q_valid, d_valid = np.isfinite(query), np.isfinite(datastore)
    q, d = np.where(q_valid, query, 0), np.where(d_valid, datastore, 0)
    q_mask, d_mask = q_valid.astype(np.float32), d_valid.astype(np.float32)
    overlap = q_mask @ d_mask.T
    squared = np.maximum(q ** 2 @ d_mask.T + q_mask @ (d ** 2).T - 2 * q @ d.T, 0)
    squared *= np.divide(query.shape[1], overlap, out=np.zeros_like(overlap), where=overlap > 0)
    squared[overlap < max(1, int(np.ceil(query.shape[1] * minimum_overlap_fraction)))] = np.inf
    squared[~np.isfinite(squared)] = np.inf
    return squared


def blockwise_topk(query, datastore, query_refs, datastore_refs, query_ticks, datastore_ticks, ends,
                   *, k, period, stride, horizon, scope='all', minimum_overlap_fraction=0.8,
                   query_block_size=256, datastore_block_size=4096):
    """Return ordered prefixes; an incomplete maximum-K list is entirely unusable."""
    ids = np.full((len(query), k), -1, dtype=np.int64)
    distances = np.full((len(query), k), np.inf, dtype=np.float32)
    if len(datastore) < k:
        return distances, ids
    for q_start in range(0, len(query), query_block_size):
        q_stop = min(q_start + query_block_size, len(query))
        best_distance = np.full((q_stop - q_start, k), np.inf, dtype=np.float32)
        best_ids = np.full((q_stop - q_start, k), -1, dtype=np.int64)
        for d_start in range(0, len(datastore), datastore_block_size):
            d_stop = min(d_start + datastore_block_size, len(datastore))
            distance = squared_distance(query[q_start:q_stop], datastore[d_start:d_stop], minimum_overlap_fraction)
            allowed = eligible(query_refs[q_start:q_stop], datastore_refs[d_start:d_stop],
                               query_ticks[q_start:q_stop], datastore_ticks[d_start:d_stop], ends,
                               period=period, stride=stride, horizon=horizon, scope=scope)
            distance[~allowed] = np.inf
            local_k = min(k, d_stop - d_start)
            positions = np.argpartition(distance, local_k - 1, axis=1)[:, :local_k]
            local_distance = np.take_along_axis(distance, positions, axis=1)
            combined = np.concatenate((best_distance, local_distance), axis=1)
            combined_ids = np.concatenate((best_ids, positions + d_start), axis=1)
            positions = np.argpartition(combined, k - 1, axis=1)[:, :k]
            best_distance = np.take_along_axis(combined, positions, axis=1)
            best_ids = np.take_along_axis(combined_ids, positions, axis=1)
        order = np.argsort(best_distance, axis=1, kind='stable')
        best_distance = np.take_along_axis(best_distance, order, axis=1)
        best_ids = np.take_along_axis(best_ids, order, axis=1)
        incomplete = ~np.isfinite(best_distance[:, -1])
        best_ids[incomplete] = -1
        best_distance[incomplete] = np.inf
        ids[q_start:q_stop] = best_ids
        distances[q_start:q_stop] = np.sqrt(best_distance)
    return distances, ids
