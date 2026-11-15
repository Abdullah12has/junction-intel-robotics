"""Reconstruct captured source from the manifest; never execute it."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True, help='Original path relative to /home/intel')
    parser.add_argument('--output', type=Path, required=True, help='New, nonexistent directory')
    args = parser.parse_args()
    prefix = args.source_root.strip('/') + '/'
    manifest = json.loads((ROOT / 'intel-station/IMPORT-MANIFEST.json').read_text())
    items = [i for i in manifest['files'] if i['path'].startswith(prefix)
             and i['disposition'] != 'excluded']
    if not items:
        parser.error('No captured files under that source root')
    for item in items:
        source = ROOT / item['repository_path']
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Snapshot hash mismatch: {source}')
        relative = Path(item['path'][len(prefix):])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid manifest path')
    args.output.mkdir(parents=True, exist_ok=False)
    for item in items:
        target = args.output / item['path'][len(prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / item['repository_path'], target)
    print(f'Reconstructed {len(items)} source files in {args.output}; no code executed.')


if __name__ == '__main__':
    main()
