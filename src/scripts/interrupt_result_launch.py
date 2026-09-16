"""Mark unfinished TIME tasks from one failed cluster launch interrupted."""
import argparse
from pathlib import Path
from timebench.pipeline.runs import interrupt_launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--launch-id', required=True)
    args = parser.parse_args()
    changed = interrupt_launch(args.root, args.launch_id)
    print(f'TIME interrupted-launch recovery launch_id={args.launch_id} tasks={len(changed)} root={args.root}')


if __name__ == '__main__':
    main()
