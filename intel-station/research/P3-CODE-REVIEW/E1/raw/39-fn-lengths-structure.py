import ast, glob, json, os
R='/home/intel/intel-junction-2026'
files=[]
for pat in ['app/tools/*.py','app/tools/tests/*.py','app/motion/*.py','app/motion/tools/*.py','app/eyeinhand/*.py','app/pick_v2/kinematics.py']:
    files+=glob.glob(os.path.join(R,pat))
print('== functions over 60 lines (ast, def to end line)')
n=0
for f in sorted(files):
    src=open(f,encoding='utf-8').read(); t=ast.parse(src)
    for node in ast.walk(t):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            L=node.end_lineno-node.lineno+1; n+=1
            if L>60: print(os.path.relpath(f,R), node.name, node.lineno, L)
print('functions total', n, 'python files', len(files))
print('== gm and motion data: structure only (keys and lengths, no values)')
for f in sorted(glob.glob(os.path.join(R,'app/motion/gm/*'))):
    s=open(f,encoding='utf-8').read()
    keys=[l.split(':')[0] for l in s.splitlines() if l and not l.startswith((' ','#','\t','-','{','}',']','"')) and ':' in l]
    print(os.path.relpath(f,R), 'lines', s.count(chr(10)), 'top-level keys', keys[:12])
p=os.path.join(R,'app/motion/project_robots.json')
d=json.load(open(p,encoding='utf-8'))
def shape(x,depth=0):
    if isinstance(x,dict): return {k:shape(v,depth+1) if depth<3 else type(v).__name__ for k,v in x.items()}
    if isinstance(x,list): return ['list of %d' % len(x)] + ([shape(x[0],depth+1)] if x and depth<3 else [])
    return type(x).__name__
print('project_robots.json shape (values masked):', json.dumps(shape(d))[:900])
