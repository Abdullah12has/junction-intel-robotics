"""Tiny stdlib-only static check (Opus I-61): names that are read but never bound anywhere visible (module globals,
imports, function parameters, locals, comprehension targets, builtins). Catches missing imports and typos in rarely
executed paths (recovery, carry-over). Usage: python undef.py file.py [...]"""
import ast
import builtins
import sys

BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__"}


def bound_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            names.update(node.names)
    return names


def check(path):
    tree = ast.parse(open(path).read(), path)
    bound = bound_names(tree) | BUILTINS
    bad = sorted({(n.lineno, n.id) for n in ast.walk(tree)
                  if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound})
    for ln, name in bad:
        print(f"{path}:{ln}: undefined name '{name}'")
    return len(bad)


n = sum(check(p) for p in sys.argv[1:])
print(f"{n} undefined-name findings in {len(sys.argv) - 1} files")
