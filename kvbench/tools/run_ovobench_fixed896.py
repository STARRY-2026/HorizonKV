"""Run the checked all-method fixed-canvas OVO subset from any directory."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kvbench import RunConfig, run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    values = json.loads((ROOT / 'configs/ovobench_fixed896_all_methods.json').read_text())
    values['anno_path'] = ROOT / values['anno_path']
    values.update(output_dir=args.output_dir, dry_run=args.dry_run)
    print(json.dumps(run(RunConfig(**values)), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
