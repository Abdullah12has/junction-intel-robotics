"""Verify captured source bytes and syntax without importing station code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'intel-station/IMPORT-MANIFEST.json').read_text())
    checked = parsed = 0
    for item in manifest['files']:
        if item['disposition'] == 'excluded':
            continue
        path = ROOT / item['repository_path']
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256']:
            raise ValueError(f"Snapshot hash mismatch: {path}")
        checked += 1
        if item['disposition'] == 'imported' and path.suffix == '.py':
            ast.parse(data, filename=str(path))
            parsed += 1
    print(f'Verified {checked} source mappings; parsed {parsed} imported Python files.')


if __name__ == '__main__':
    main()
