# Read-only static scan: parses sources with ast, never imports or runs them.
import ast
import os
import re
import sys

APP = sys.argv[1]
START = [s for s in sys.argv[2].split(',') if s]
MAXFILES = 40
KW = re.compile(r"(Serial|serial\.|VideoCapture|pipeline|realsense|\brs\.|socket|create_connection|\.connect\(|Popen|zmq|mqtt|urlopen|requests\.|/dev/|ttyACM|/video|os\.system|Thread\(|Process\(|importlib|__import__|exec\(|eval\()")

index = {}
for dp, dn, fn in os.walk(APP):
    for f in fn:
        if f.endswith('.py'):
            index.setdefault(f[:-3], []).append(os.path.join(dp, f))


def resolve(name, importer_dir):
    cands = index.get(name)
    if not cands:
        return None, 0
    if len(cands) == 1:
        return cands[0], 1
    same = sorted(c for c in cands if os.path.dirname(c) == importer_dir)
    if same:
        return same[0], len(cands)
    tools = sorted(c for c in cands if os.sep + 'tools' + os.sep in c)
    if tools:
        return tools[0], len(cands)
    return sorted(cands)[0], len(cands)


def imports_of(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name.split('.')[-1] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                out.append(node.module.split('.')[-1])
            out += [a.name for a in node.names]
    return out


def dotted(n):
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return dotted(n.value) + '.' + n.attr
    return '?'


class Calls(ast.NodeVisitor):
    def __init__(self):
        self.out = []

    def visit_FunctionDef(self, node):
        pass

    def visit_AsyncFunctionDef(self, node):
        pass

    def visit_ClassDef(self, node):
        pass

    def visit_Lambda(self, node):
        pass

    def visit_Call(self, node):
        self.out.append((node.lineno, dotted(node.func)))
        self.generic_visit(node)


def is_main_guard(node):
    t = getattr(node, 'test', None)
    return (isinstance(node, ast.If) and isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name) and t.left.id == '__name__')


queue = [(s, APP) for s in START]
seen = set()
count = 0
while queue and count < MAXFILES:
    name, imp_dir = queue.pop(0)
    if name.startswith('/'):
        path, n = name, 1
    else:
        path, n = resolve(name, imp_dir)
    if path is None or path in seen:
        continue
    seen.add(path)
    count += 1
    rel = path.replace(APP, 'app', 1)
    try:
        src = open(path, encoding='utf-8', errors='replace').read()
        tree = ast.parse(src, path)
    except SyntaxError as e:
        print('PARSE ERROR', rel, e)
        continue
    print('== FILE %s candidates=%d' % (rel, n))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            seg = ast.get_source_segment(src, node) or ''
            print('  IMP %d %s' % (node.lineno, seg.replace('\n', ' ')[:120]))
    top = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        if is_main_guard(node):
            continue
        c = Calls()
        c.visit(node)
        top += c.out
    for ln, nm in top[:40]:
        print('  TOPCALL %d %s' % (ln, nm))
    hits = [(i, l) for i, l in enumerate(src.splitlines(), 1) if KW.search(l)]
    for i, l in hits[:30]:
        print('  HIT %d %s' % (i, l.strip()[:160]))
    print('  hits_total=%d' % len(hits))
    for child in imports_of(tree):
        queue.append((child, os.path.dirname(path)))
print('SCANNED %d files' % count)
