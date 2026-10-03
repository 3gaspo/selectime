"""Refit every reported control on the official test dates for each backbone."""
import hydra
from omegaconf import OmegaConf
from timebench.scripts.entry import configure_paths

configure_paths()


@hydra.main(version_base=None, config_path='../timebench/conf', config_name='oracles')
def main(config):
    from timebench.pipeline.studies import OracleWorkflow
    resolved = OmegaConf.to_container(config, resolve=True)
    for model in resolved['models']:
        OracleWorkflow({**resolved, 'model': model}).run(resolved['stage'])


if __name__ == '__main__':
    main()
