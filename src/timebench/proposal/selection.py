"""Paired moving-date-block-bootstrap selection, source-adapted from Adaptime."""
import numpy as np
from timebench.proposal.candidates import UNIVARIATE, selection_rank


def row_msse(predictions, labels, scales):
    """All candidates use identical finite target steps and seasonal scales."""
    labels = np.asarray(labels, dtype=np.float64)
    valid = np.isfinite(labels)
    count = valid.sum(axis=-1)
    losses = {}
    for method, prediction in predictions.items():
        prediction = np.asarray(prediction, dtype=np.float64)
        usable = (count > 0) & np.isfinite(scales) & (scales > 0)
        usable &= np.all(~valid | np.isfinite(prediction), axis=-1)
        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            squared = np.where(valid, (prediction - labels) ** 2, 0).sum(axis=-1)
            values = squared / np.maximum(count, 1) / scales
        losses[method] = np.where(usable & np.isfinite(values), values, np.nan)
    return losses


def date_losses(losses, ticks, positions):
    """Average variates within each date, then compare dates with equal weight."""
    positions = np.asarray(positions, dtype=np.int64)
    common = np.ones(len(positions), dtype=bool)
    for values in losses.values():
        common &= np.isfinite(values[positions])
    positions = positions[common]
    dates, inverse = np.unique(ticks[positions], return_inverse=True)
    count = np.bincount(inverse, minlength=len(dates))
    return dates, {method: np.bincount(inverse, weights=values[positions], minlength=len(dates)) / count
                   for method, values in losses.items()}


def select_with_block_bootstrap(losses, *, seed, prediction_length, validation_stride,
                                replications=1000, block_length=None):
    """Adaptime's paired one-standard-error rule without fitting or alpha ranks."""
    n_dates = len(next(iter(losses.values())))
    if not n_dates:
        return {'selected_method': UNIVARIATE, 'validation_dates': 0,
                'fallback_reason': 'no_usable_validation_dates', 'candidates': [], 'block_length': 0}
    if any(len(values) != n_dates or not np.isfinite(values).all() for values in losses.values()):
        raise ValueError('Bootstrap candidates must have finite, date-aligned losses')
    best = min(losses, key=lambda method: (float(np.mean(losses[method])), selection_rank(method)))
    if n_dates < 2:
        length, blocks = 1, 0
    else:
        overlap = max(1, int(np.ceil(prediction_length / validation_stride)))
        automatic = max(1, int(round(n_dates ** (1 / 3))), overlap)
        length = min(block_length or automatic, n_dates - 1)
        blocks = int(np.ceil(n_dates / length))
    records, admissible = [], []
    for method, values in losses.items():
        differences = np.asarray(values, dtype=np.float64) - losses[best]
        difference = float(np.mean(differences))
        standard_error = 0.0
        if blocks:
            # Reinitializing the same run seed pairs the sampled blocks across candidates.
            rng = np.random.default_rng(seed)
            sums = np.zeros(replications, dtype=np.float64)
            remaining = n_dates
            offsets = np.arange(length)
            while remaining:
                width = min(length, remaining)
                starts = rng.integers(0, n_dates - length + 1, size=replications)
                sums += differences[starts[:, None] + offsets[None, :width]].sum(axis=1)
                remaining -= width
            standard_error = float(np.std(sums / n_dates, ddof=1))
        within = difference <= standard_error + 1e-12
        records.append({'method': method, 'validation_msse': float(np.mean(values)),
                        'validation_difference_from_best': difference,
                        'bootstrap_standard_error': standard_error, 'within_one_standard_error': within})
        if within:
            admissible.append(method)
    return {'selected_method': min(admissible, key=selection_rank), 'observed_best': best,
            'selection_rule': 'paired_moving_date_block_bootstrap_one_standard_error',
            'validation_dates': n_dates, 'replications': replications, 'block_length': length,
            'block_length_source': 'configured' if block_length else 'automatic',
            'fallback_reason': 'fewer_than_two_validation_dates' if n_dates < 2 else None,
            'preference': 'univariate_then_multivariate_then_self_augmentation_then_smallest_k',
            'candidates': records}


def select_candidates(predictions, labels, scales, references, ticks, *, granularity,
                      seed, prediction_length, validation_stride, replications=1000, block_length=None):
    if granularity not in ('task', 'per_variate'):
        raise ValueError('granularity must be task or per_variate')
    losses = row_msse(predictions, labels, scales)
    keys = np.unique(references[:, :2], axis=0) if granularity == 'per_variate' else [None]
    selections = []
    for key in keys:
        positions = (np.flatnonzero((references[:, :2] == key).all(axis=1))
                     if key is not None else np.arange(len(references)))
        dates, scored = date_losses(losses, ticks, positions)
        selected = select_with_block_bootstrap(scored, seed=seed, prediction_length=prediction_length,
                                              validation_stride=validation_stride, replications=replications,
                                              block_length=block_length)
        selected['validation_date_ticks'] = dates.tolist()
        if key is not None:
            selected.update(item=int(key[0]), channel=int(key[1]))
        selections.append(selected)
    return {'schema_version': 1, 'granularity': granularity, 'selections': selections}
