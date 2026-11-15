"""P4-FUSION E1 read-only check: module-level calls in the loop modules that will be imported (must not open devices)."""
import ast
import os

R = "/home/intel/intel-junction-2026"
FILES = ["app/tools/grasp_auto.py", "app/tools/handdet.py", "app/tools/sort_loop.py", "app/tools/sort_any.py",
         "app/tools/pick_any.py", "app/tools/kin.py", "app/anomaly/det2_f2/inspector.py"]
for f in FILES:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        print("MISSING", f)
        continue
    with open(p, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    print("FILE", f, "top-level statements", len(tree.body))
    for node in tree.body:
        if isinstance(node, ast.If) and "__main__" in ast.unparse(node.test):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom,
                             ast.Pass)):
            continue
        calls = sorted({ast.unparse(c.func) for c in ast.walk(node) if isinstance(c, ast.Call)})
        if calls:
            print("  line", node.lineno, type(node).__name__, calls[:10])
