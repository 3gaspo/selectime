"""Arrow-backed validation/test rows and two strictly pre-boundary datastores."""
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Task:
    dataset: str
    term: str
    prediction_length: int
    test_length: int
    validation_length: int
    seasonality: int
    retrieval_context_length: int
    alignment_period: int
    datastore_stride: int
    validation_stride: int
    datastore_scope: str = 'all'
    max_datastore_windows: int | None = None

    def config(self):
        return asdict(self)

    def validate(self):
        positive = [self.prediction_length, self.test_length, self.seasonality,
                    self.retrieval_context_length, self.alignment_period,
                    self.datastore_stride, self.validation_stride]
        if min(positive) < 1 or self.test_length < self.prediction_length:
            raise ValueError('Task lengths and strides must be positive and test must cover one horizon')
        if self.validation_length < 0 or 0 < self.validation_length < self.prediction_length:
            raise ValueError('Validation must be disabled or contain a complete horizon')
        if self.datastore_stride % self.alignment_period:
            raise ValueError('datastore_stride must be a multiple of alignment_period')
        if self.validation_stride != self.prediction_length:
            raise ValueError('validation_stride must equal the official test stride H')
        if self.datastore_scope not in ('all', 'same_series'):
            raise ValueError('datastore_scope must be all or same_series')
        if self.max_datastore_windows is not None and self.max_datastore_windows < 1:
            raise ValueError('max_datastore_windows must be positive or null')


def interval_origins(start, stop, horizon, stride):
    return np.arange(start, stop - horizon + 1, stride, dtype=np.int64)


def aligned_origins(start_tick, cutoff, lookback, horizon, period, stride, residues):
    """Union of query phases, anchored at the datastore's last complete future."""
    latest = start_tick + cutoff - horizon
    phases = []
    for residue in residues:
        last = latest - (latest - residue) % period - start_tick
        phases.append(np.arange(last, lookback - 1, -stride, dtype=np.int64)[::-1])
    return np.unique(np.concatenate(phases)) if phases else np.empty(0, dtype=np.int64)


def start_tick(row):
    frequency = row['freq']
    if frequency.endswith('T'):
        frequency = frequency[:-1] + 'min'
    elif frequency.endswith('H'):
        frequency = frequency[:-1] + 'h'
    period = pd.Period(pd.Timestamp(row['start']), freq=frequency)
    return period.ordinal // period.freq.n


class Windows:
    def __init__(self, task, storage, source=None):
        self.task = task
        self.source_path = Path(storage) / task.dataset
        if source is None:
            import datasets
            source = datasets.load_from_disk(str(self.source_path))
        self.source = source
        # Period ordinals count the base unit even for 5T/15T etc. Convert to
        # sampling-step units before adding an array origin or a step stride.
        self.start_ticks = np.asarray([start_tick(row) for row in self.source], dtype=np.int64)
        shapes = [np.asarray(row['target']).shape for row in self.source]
        self.shapes = [(shape[0], shape[-1]) if len(shape) == 2 else (1, shape[0]) for shape in shapes]
        if any(length <= task.test_length for _, length in self.shapes):
            raise ValueError('Official test interval leaves no forecasting history')
        if len({channels for channels, _ in self.shapes}) != 1:
            raise ValueError('TIME evaluation requires a common variate count within a task')

    @lru_cache(maxsize=2)
    def target(self, item):
        values = np.asarray(self.source[item]['target'], dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError('TIME targets must not contain infinite values')
        return values[None, :] if values.ndim == 1 else values

    def boundary(self, item, split):
        test_start = self.shapes[item][1] - self.task.test_length
        if split == 'validation':
            return max(0, test_start - self.validation_date_count(item) * self.task.prediction_length)
        return test_start

    def validation_date_count(self, item):
        if self.task.validation_length == 0:
            return 0
        test_start = self.shapes[item][1] - self.task.test_length
        test_count = len(interval_origins(
            test_start, self.shapes[item][1], self.task.prediction_length,
            self.task.prediction_length,
        ))
        return min(test_count, self.task.validation_length // self.task.prediction_length)

    def references(self, split):
        rows = []
        for item, (channels, length) in enumerate(self.shapes):
            if split == 'test':
                start, stop, stride = self.boundary(item, split), length, self.task.prediction_length
            elif split == 'validation':
                test_start = self.boundary(item, 'test')
                count = self.validation_date_count(item)
                origins = test_start - self.task.prediction_length * np.arange(
                    count, 0, -1, dtype=np.int64
                )
            else:
                raise ValueError(split)
            if split == 'test':
                origins = interval_origins(start, stop, self.task.prediction_length, stride)
            origins = origins[origins > 0]
            for channel in range(channels):
                rows.extend((item, channel, int(origin)) for origin in origins)
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)

    def ticks(self, references):
        return self.start_ticks[references[:, 0]] + references[:, 2]

    def histories(self, references, limit):
        return [self.target(int(item))[int(channel), max(0, int(origin) - limit):int(origin)]
                for item, channel, origin in references]

    def multivariate_history(self, item, origin, limit):
        return self.target(int(item))[:, max(0, int(origin) - limit):int(origin)]

    def labels(self, references):
        h = self.task.prediction_length
        return np.asarray([self.target(int(item))[int(channel), int(origin):int(origin) + h]
                           for item, channel, origin in references], dtype=np.float32).reshape(-1, h)

    def sequences(self, references):
        r, h = self.task.retrieval_context_length, self.task.prediction_length
        return np.asarray([self.target(int(item))[int(channel), int(origin) - r:int(origin) + h]
                           for item, channel, origin in references], dtype=np.float32).reshape(-1, r + h)

    @lru_cache(maxsize=2)
    def scale_prefix(self, item):
        values = self.target(item).astype(np.float64)
        p = self.task.seasonality
        left, right = values[:, :-p], values[:, p:]
        valid = np.isfinite(left) & np.isfinite(right)
        differences = np.where(valid, right - left, 0)
        squared = np.pad(np.cumsum(differences ** 2, axis=-1), ((0, 0), (1, 0)))
        counts = np.pad(np.cumsum(valid, axis=-1), ((0, 0), (1, 0)))
        return squared, counts

    def msse_scales(self, references):
        """Mean seasonal squared differences over the complete observed prefix."""
        scales = np.full(len(references), np.nan, dtype=np.float64)
        for row, (item, channel, origin) in enumerate(references):
            squared, counts = self.scale_prefix(int(item))
            position = max(0, int(origin) - self.task.seasonality)
            count = counts[int(channel), position]
            value = squared[int(channel), position] / count if count else np.nan
            if value > 0:
                scales[row] = value
        return scales

    def datastore(self, split, references):
        residues = np.unique(self.ticks(references) % self.task.alignment_period)
        candidates = []
        for item, (channels, _) in enumerate(self.shapes):
            origins = aligned_origins(int(self.start_ticks[item]), self.boundary(item, split),
                                      self.task.retrieval_context_length, self.task.prediction_length,
                                      self.task.alignment_period, self.task.datastore_stride, residues)
            for channel in range(channels):
                candidates.append((item, channel, origins))
        cap = self.task.max_datastore_windows
        if cap is not None:
            per_variate = cap // len(candidates) if self.task.datastore_scope == 'all' else cap
            candidates = [(item, channel, origins[-per_variate:] if per_variate else origins[:0])
                          for item, channel, origins in candidates]
        rows = [(item, channel, int(origin)) for item, channel, origins in candidates for origin in origins]
        refs = np.asarray(rows, dtype=np.int64).reshape(-1, 3)
        end_ticks = self.start_ticks + np.asarray([self.boundary(item, split) - self.task.prediction_length
                                                  for item in range(len(self.shapes))])
        return refs, end_ticks


def write_prepared(windows, destination, split):
    destination = Path(destination)
    refs = windows.references(split)
    datastore, ends = windows.datastore(split, refs)
    products = {f'{split}_references': refs, f'{split}_ticks': windows.ticks(refs),
                f'{split}_datastore': datastore, f'{split}_datastore_ticks': windows.ticks(datastore),
                f'{split}_datastore_ends': ends}
    files = []
    for name, values in products.items():
        np.save(destination / f'{name}.npy', values, allow_pickle=False)
        files.append(f'{name}.npy')
    channels = windows.shapes[0][0]
    if any(shape[0] != channels for shape in windows.shapes):
        raise ValueError('Prepared task has inconsistent channel counts')
    requested_dates = (
        sum(windows.validation_date_count(item) for item in range(len(windows.shapes)))
        if split == 'validation'
        else sum(windows.task.test_length // windows.task.prediction_length for _ in windows.shapes)
    )
    counts = {
        'requested_dates': int(requested_dates),
        'requested_rows': int(requested_dates * channels),
        'available_dates': int(len(refs) // channels),
        'available_rows': int(len(refs)),
        'datastore_rows': int(len(datastore)),
    }
    (destination / 'prepared.json').write_text(json.dumps({
        'schema_version': 1, 'task': windows.task.config(), 'split': split, 'counts': counts,
        'source_path': str(windows.source_path),
        'datastore_policy': 'validation_before_validation_start_test_before_test_start',
        'validation_schedule': 'walk_backward_from_first_test_origin_at_stride_H',
        'reference_columns': ['item', 'channel', 'origin'], 'calendar_tick': 'pandas Period ordinal',
    }, indent=2), encoding='utf-8')
    return [*files, 'prepared.json']
