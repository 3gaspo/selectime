"""Pin or restore automatic selection for one completed TIME run."""
import argparse
from pathlib import Path
from timebench.pipeline.runs import set_selected_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Completed run_n directory to select')
    parser.add_argument('--auto', action='store_true', help='Let later completed repeats replace this selection')
    args = parser.parse_args()
    set_selected_run(args.run, pinned=not args.auto)
    print(f"Selected {args.run} ({'auto' if args.auto else 'pinned'})")


if __name__ == '__main__':
    main()
