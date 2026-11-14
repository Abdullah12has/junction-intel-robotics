"""Download only the SO-101 files from a pinned Menagerie commit."""
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

REVISION = "370c43965323c0944b433b202aac5058b5f907a9"
BASE = f"https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{REVISION}/robotstudio_so101"
DEST = Path(__file__).parent / "assets" / "robotstudio_so101"


def fetch(name):
    path = DEST / name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = urllib.request.urlopen(f"{BASE}/{name}", timeout=60).read()
    path.write_bytes(data)
    return data


def main():
    files = {}
    for name in ("so101.xml", "LICENSE", "README.md"):
        files[name] = fetch(name)
    root = ET.fromstring(files["so101.xml"])
    for mesh in root.findall("./asset/mesh"):
        name = "assets/" + mesh.attrib["file"]
        files[name] = fetch(name)
    manifest = {
        "repository": "https://github.com/google-deepmind/mujoco_menagerie",
        "revision": REVISION,
        "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()},
    }
    (DEST / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Downloaded {len(files)} files to {DEST}")


if __name__ == "__main__":
    main()
