"""Run one independently configured Selectime retrieval ablation."""
import hydra
from omegaconf import OmegaConf
from timebench.scripts.entry import configure_paths

configure_paths()


@hydra.main(version_base=None, config_path='../timebench/conf', config_name='ablation')
def main(config):
    from timebench.pipeline.studies import AblationWorkflow
    resolved = OmegaConf.to_container(config, resolve=True)
    AblationWorkflow(resolved).run(resolved['stage'])


if __name__ == '__main__':
    main()
