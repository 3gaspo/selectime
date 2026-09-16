"""Common configuration resolution for the explicit experiment entry points."""
import os
from timebench.paths import logs_root, outputs_root


def configure_paths():
    # Set these before Hydra allocates its own log directory.
    os.environ['TIME_LOGS'] = str(logs_root())
    os.environ['TIME_OUTPUTS'] = str(outputs_root())
    os.environ['TIME_PROJECT_NAME'] = 'selectime'


def run(config, stage=None):
    from omegaconf import OmegaConf
    from timebench.pipeline.workflow import Workflow
    resolved = OmegaConf.to_container(config, resolve=True)
    Workflow(resolved).run(stage or resolved['stage'])
