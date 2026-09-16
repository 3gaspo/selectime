"""Freeze a bootstrap-selected candidate independently for each item/variate."""
import hydra
from timebench.scripts.entry import configure_paths, run

configure_paths()


@hydra.main(version_base=None, config_path='../timebench/conf', config_name='experiment')
def main(config):
    run(config, 'select_per_variate')


if __name__ == '__main__':
    main()
