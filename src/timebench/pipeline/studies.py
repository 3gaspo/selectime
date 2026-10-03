"""Independent retrieval ablations and test-refitted Selectime controls."""
import json
import os
import numpy as np

from timebench.data.windows import Windows
from timebench.proposal.candidates import UNIVARIATE, MULTIVARIATE, candidate_k
from timebench.pipeline.workflow import Workflow, write_json


ABLATIONS = {
    'same_user': {'datastore_scope': 'same_user'},
    'full_datastore': {'alignment_period': 1, 'datastore_stride': 1},
    'raw_distance': {'retrieval_instance_normalization': False},
    'no_query_scaling': {'query_scaling': False},
}
ORACLE_STAGES = ('calibrate', 'assemble', 'evaluate', 'report')


class AblationWorkflow(Workflow):
    """One change relative to a completed default method; no factorial sweep."""

    def __init__(self, config):
        self.variant = config['variant']
        if self.variant not in ABLATIONS:
            raise ValueError(f'Unknown ablation {self.variant}; choose {tuple(ABLATIONS)}')
        self.changes = ABLATIONS[self.variant]
        self.source = Workflow(config)
        self.experiment = f'scope_ablation_{self.variant}'
        super().__init__({**config, **self.changes})
        method = config['ablation_method']
        if method not in self.source.controls or not candidate_k(method):
            raise ValueError('ablation_method must be a supported top_k_<K>_mix in k_values')
        self.controls = {method: self.source.controls[method]}
        self.raw_methods = [self.controls[method]]
        self.source_tasks = {(task.dataset, task.term): task for task in self.source.tasks}

    def source_task(self, task):
        return self.source_tasks[task.dataset, task.term]

    def raw(self, task, split, method):
        if method in (UNIVARIATE, MULTIVARIATE):
            source_task = self.source_task(task)
            original = self.source.prepared(source_task, split)
            current = self.prepared(task, split)
            if not np.array_equal(self.array(original, f'{split}_references'),
                                  self.array(current, f'{split}_references')):
                raise ValueError('A retrieval ablation must retain the default forecast dates')
            return self.source.raw(source_task, split, method)
        return super().raw(task, split, method)

    def prediction_identity_root(self, task, method):
        if method == UNIVARIATE:
            return self.source.prediction_identity_root(self.source_task(task), method)
        return super().prediction_identity_root(task, method)

    def methods(self):
        return [UNIVARIATE, *self.controls]

    def preparation_artifacts(self, task, windows, destination, split):
        files = super().preparation_artifacts(task, windows, destination, split)
        baseline = Windows(self.source_task(task), self.storage, source=windows.source)
        refs = baseline.references(split)
        sizes = baseline.datastore_size(split, refs)
        metadata = json.loads((destination / 'prepared.json').read_text())
        sizes.update(variant=self.variant, split=split,
                     materialized_variant_rows=metadata['counts']['datastore_rows'],
                     maximum_k_support=max(self.k_values),
                     query_eligible_asymptotic_expansion=float(baseline.task.datastore_stride),
                     same_user_definition='same dataset item; all its variates remain eligible')
        write_json(destination / 'datastore_size.json', sizes)
        return [*files, 'datastore_size.json']

    def report_artifacts(self, inputs, destination):
        from timebench.results.comparison import write_csv, build_ablation_report
        files = super().report_artifacts(inputs, destination)
        sizes = []
        for task in self.tasks:
            for split in ('validation', 'test'):
                prepared = self.prepared(task, split)
                entry = json.loads((prepared / 'datastore_size.json').read_text())
                sizes.append({'dataset': task.dataset, 'term': task.term, **entry})
        write_csv(destination / 'datastore_sizes.csv', sizes)
        return [*files, 'datastore_sizes.csv', *build_ablation_report(inputs, destination, self)]

    def report_dependencies(self, inputs):
        deps = super().report_dependencies(inputs)
        for task, method, *_ in inputs:
            if method in self.controls:
                deps[f'default/{task.dataset}/{task.term}/{method}'] = self.source.evaluation(
                    self.source_task(task), method)
        return deps


class OracleWorkflow(Workflow):
    """Refit all supported controls on test labels without rerunning backbones."""
    experiment = 'scope_oracles'
    fitting_split = 'test'

    def __init__(self, config):
        super().__init__(config)
        self.source = Workflow(config)

    def prepared(self, task, split):
        return self.source.prepared(task, split)

    def raw(self, task, split, method):
        return self.source.raw(task, split, method)

    def prediction_identity_root(self, task, method):
        if method not in self.controls:
            return self.source.prediction_identity_root(task, method)
        return super().prediction_identity_root(task, method)

    def report_artifacts(self, inputs, destination):
        from timebench.results.comparison import build_oracle_report
        files = super().report_artifacts(inputs, destination)
        return [*files, *build_oracle_report(inputs, destination, self.source)]

    def report_dependencies(self, inputs):
        deps = super().report_dependencies(inputs)
        for task, method, *_ in inputs:
            deps[f'default/{task.dataset}/{task.term}/{method}'] = self.source.evaluation(task, method)
        return deps

    def run(self, stage):
        if stage == 'pipeline':
            if os.getenv('SELECTIME_DEFER_COMPLETION') == '1':
                raise ValueError('Slurm must invoke one stage per srun before finalization')
            for selected in ORACLE_STAGES:
                super().run(selected)
        elif stage in ORACLE_STAGES:
            super().run(stage)
        else:
            raise ValueError(f'Oracle stages are {ORACLE_STAGES}; candidates must already be completed')
