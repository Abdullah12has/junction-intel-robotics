"""Launch the native viewer, including uv Python's macOS library path."""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    env = os.environ.copy()
    executable = Path(sys.executable)
    if sys.platform == "darwin":
        executable = executable.parent / "mjpython"
        library = str(Path(sys.base_prefix) / "lib")
        existing = env.get("DYLD_LIBRARY_PATH")
        env["DYLD_LIBRARY_PATH"] = library + (os.pathsep + existing if existing else "")
    raise SystemExit(subprocess.call(
        [str(executable), "-u", str(Path(__file__).with_name("smoke.py")), "--viewer", *sys.argv[1:]],
        env=env,
    ))
