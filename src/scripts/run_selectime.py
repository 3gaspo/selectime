"""Run the candidate experiment or one recoverable stage."""
import hydra
from timebench.scripts.entry import configure_paths, run

configure_paths()


@hydra.main(version_base=None, config_path='../timebench/conf', config_name='experiment')
def main(config):
    run(config)


if __name__ == '__main__':
    main()
