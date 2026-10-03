"""Measure fresh, uncached batch inference for every supported reported method."""
import hydra
from omegaconf import OmegaConf
from timebench.scripts.entry import configure_paths

configure_paths()


@hydra.main(version_base=None, config_path='../timebench/conf', config_name='timing')
def main(config):
    from timebench.pipeline.timing import TimingWorkflow
    resolved = OmegaConf.to_container(config, resolve=True)
    for model in resolved['models']:
        TimingWorkflow({**resolved, 'model': model}).run(resolved['stage'])


if __name__ == '__main__':
    main()
