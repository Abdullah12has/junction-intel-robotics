"""P4-FUSION E1 helper (read-only): list top-level definitions or print named function sources.

Usage: python3 -I astlist.py list FILE
       python3 -I astlist.py show FILE NAME [NAME ...]
"""
import ast
import sys

mode = sys.argv[1]
path = sys.argv[2]
names = set(sys.argv[3:])
with open(path, encoding="utf-8") as fh:
    src = fh.read()
lines = src.splitlines()
tree = ast.parse(src)
if mode == "list":
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            print(node.lineno, node.end_lineno, type(node).__name__, node.name)
        elif isinstance(node, ast.Assign):
            tg = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if tg:
                print(node.lineno, node.end_lineno, "ASSIGN", ",".join(tg))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            print(node.lineno, node.end_lineno, "IMPORT", ast.unparse(node))
    print("TOTAL_LINES", len(lines))
elif mode == "show":
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            found.add(node.name)
            print("### %s lines %d-%d" % (node.name, node.lineno, node.end_lineno))
            print("\n".join(lines[node.lineno - 1:node.end_lineno]))
    print("MISSING", sorted(names - found))
