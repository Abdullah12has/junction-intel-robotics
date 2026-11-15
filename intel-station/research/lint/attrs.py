"""Cross-module attribute check (Opus I-61, stdlib only): every `alias.name` where alias is an imported local module
must name a top-level definition (def, class, assignment, import) of that module. Catches renamed or missing helpers
in rarely executed paths. Usage: python attrs.py DIR file.py [...]"""
import ast
import os
import sys

DIR = sys.argv[1]
SEARCH = [DIR, os.path.join(DIR, "..", "anomaly", "det2_f2"), os.path.join(DIR, "..", "eyeinhand")]


def top_level(path):
    tree = ast.parse(open(path).read(), path)
    names = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name):
                        names.add(x.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.If, ast.Try, ast.With)):
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)):
                    names.add(x.name)
                elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store):
                    names.add(x.id)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for a in x.names:
                        names.add((a.asname or a.name).split(".")[0])
    # names assigned via "global X" inside functions count as module attributes too
    for x in ast.walk(tree):
        if isinstance(x, ast.Global):
            names.update(x.names)
    return names


def find_module(mod):
    for d in SEARCH:
        p = os.path.join(d, mod + ".py")
        if os.path.exists(p):
            return p
    return None


total = 0
for path in sys.argv[2:]:
    tree = ast.parse(open(path).read(), path)
    aliases = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                mp = find_module(a.name)
                if mp:
                    aliases[a.asname or a.name] = mp
    cache = {al: top_level(mp) for al, mp in aliases.items()}
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in cache:
            if isinstance(n.ctx, ast.Store):
                continue
            if n.attr not in cache[n.value.id]:
                print(f"{path}:{n.lineno}: {n.value.id}.{n.attr} not defined in {os.path.basename(aliases[n.value.id])}")
                total += 1
print(f"{total} attribute findings")
