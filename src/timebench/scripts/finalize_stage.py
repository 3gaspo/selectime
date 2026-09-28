"""Complete ready runs after srun success, or interrupt all owned task runs."""
import argparse
import json
from pathlib import Path
from timebench.pipeline.runs import RunHandle, interrupt_launch, load_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('launch')
    parser.add_argument('--interrupt', action='store_true')
    args = parser.parse_args()
    for root in [args.root]:
        if args.interrupt:
            interrupt_launch(root, args.launch)
            continue
        for path in root.rglob('stage_ready.json'):
            manifest = load_manifest(path.parent)
            if manifest['status'] == 'computed' and manifest['launch']['launch_id'] == args.launch:
                ready = json.loads(path.read_text(encoding='utf-8'))
                RunHandle(path.parent, manifest, 'finalize').complete(
                    ready['required_artifacts'],
                    artifact_metadata=ready.get('artifact_metadata'),
                )
                path.unlink()



if __name__ == '__main__':
    main()
