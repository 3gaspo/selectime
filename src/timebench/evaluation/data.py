# Copyright (c) 2023, Salesforce, Inc. SPDX-License-Identifier: Apache-2.0
"""Narrowed TIME/GluonTS official-test adapter; no fitting splits."""
from functools import cached_property
from pathlib import Path
import numpy as np
import yaml
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import split
from gluonts.itertools import Map
from gluonts.transform import Transformation
from timebench.paths import dataset_storage_root


def load_dataset_config(config_path=None):
    path = Path(config_path) if config_path else Path(__file__).parents[1] / 'config/datasets.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def prepare_data_entry(entry):
    entry = dict(entry)
    entry['target'] = np.asarray(entry['target'])
    if entry['target'].ndim == 2 and entry['target'].shape[0] == 1:
        entry['target'] = entry['target'][0]
    if hasattr(entry['start'], 'item'):
        entry['start'] = entry['start'].item()
    entry.pop('feat_dynamic_real', None)
    entry.pop('past_feat_dynamic_real', None)
    return entry


class MultivariateToUnivariate(Transformation):
    def __call__(self, data_it, is_train=False):
        for entry in data_it:
            for channel, target in enumerate(entry['target']):
                row = dict(entry)
                row['target'] = target
                row['item_id'] = f"{entry['item_id']}_dim{channel}"
                yield row


class Dataset:
    def __init__(self, name, *, term, prediction_length, test_length, storage_path=None):
        import datasets

        self.name, self.term = name, term
        self.prediction_length, self.test_length = prediction_length, test_length
        self.hf_dataset = datasets.load_from_disk(str(Path(storage_path or dataset_storage_root()) / name))
        process = ProcessDataEntry(self.freq, one_dim_target=self.target_dim == 1)
        self.gluonts_dataset = Map(lambda row: process(prepare_data_entry(row)), self.hf_dataset)
        if self.target_dim > 1:
            self.gluonts_dataset = MultivariateToUnivariate().apply(self.gluonts_dataset)

    @cached_property
    def freq(self):
        return self.hf_dataset[0]['freq']

    @cached_property
    def target_dim(self):
        target = np.asarray(self.hf_dataset[0]['target'])
        return target.shape[0] if target.ndim == 2 else 1

    @property
    def windows(self):
        return self.test_length // self.prediction_length

    @property
    def test_data(self):
        _, template = split(self.gluonts_dataset, offset=-self.test_length)
        return template.generate_instances(prediction_length=self.prediction_length,
                                           windows=self.windows, distance=self.prediction_length)
