p='/home/intel/intel-junction-2026/work/P3-CODE-REVIEW/E1/findings.tsv'
s=open(p,encoding='utf-8').read()
reps=[('(hand_watch.py:55-60)','(hand_watch.py:71-76)'),('lines 75-100','line 158'),('so101 47-49 and module docstring','so101 47-49 and docstring line 8')]
for a,b in reps:
    n=s.count(a); assert n==1,(a,n); s=s.replace(a,b)
open(p,'w',encoding='utf-8').write(s)
print('corrections applied: 3 replacements (F-11 hand_watch lines, F-36 plan_ramp line, F-04 docstring line)')
