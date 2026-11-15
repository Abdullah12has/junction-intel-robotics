p='/home/intel/intel-junction-2026/work/P3-CODE-REVIEW/E1/findings.tsv'
SEV={'critical','high','medium','low'}
out=[]
for i,line in enumerate(open(p,encoding='utf-8').read().rstrip('\n').split('\n')):
    f=line.split('\t')
    if i>0 and len(f)==6:
        if f[2] in SEV:
            f.insert(2,'in file column')
        else:
            f.append('see finding text (no separate fix column in F-01..F-60 rows)')
    out.append('\t'.join(f))
open(p,'w',encoding='utf-8').write('\n'.join(out)+'\n')
print('rows', len(out)-1)
