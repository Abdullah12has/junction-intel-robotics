"""Run only hardware-free checks, never discover physical trial scripts."""
from pathlib import Path
import ast
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = [p for folder in ("cell", "training", "simulation", "scripts")
               for p in (ROOT / folder).rglob("*.py")
               if not any(part.startswith(".") or part in {"assets", "outputs", "__pycache__"}
                          for part in p.relative_to(ROOT).parts)]
    for path in sources:
        ast.parse(path.read_text(), filename=str(path))
    print(f"Syntax checked {len(sources)} Python files", flush=True)
    for directory, tests in (
        ("cell", ["test_policy_action_bounds", "test_visual_servo"]),
        ("training/blue_plate", ["test_variants"]),
    ):
        subprocess.run([sys.executable, "-m", "unittest", "-v", *tests],
                       cwd=ROOT / directory, check=True)


if __name__ == "__main__":
    main()
