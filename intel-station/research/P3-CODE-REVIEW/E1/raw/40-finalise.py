import re, collections
D='/home/intel/intel-junction-2026/work/P3-CODE-REVIEW/E1'
rows=[l.split('\t') for l in open(D+'/findings.tsv',encoding='utf-8').read().splitlines()[1:]]
sev=collections.Counter(r[3] for r in rows)
files=set(l.split(':')[0] for l in open(D+'/raw/24-ai-mentions-full.txt',encoding='utf-8'))
n_files=len(files)
rv=open(D+'/review.md',encoding='utf-8').read()
rv=re.sub(r'Severity count: .*?\.\n', 'Severity count (findings.tsv, %d rows): critical %d, high %d, medium %d, low %d.\n' % (len(rows), sev['critical'], sev['high'], sev['medium'], sev['low']), rv, count=1)
rv=re.sub(r'Total: 200 lines in \d+ files\.', 'Total: 200 lines in %d files.' % n_files, rv, count=1)
reps=[('about 14 ms per watchdog tick in the worst case (3 views of up to 9.3 ms, the wrist and top at their maxima), about 6% of its 0.25 s period','about 17 ms per watchdog tick in the worst case (9.3 ms close, 4.8 ms wrist, 2.5 ms top, all at their maxima), about 7% of its 0.25 s period'),
('the 60 s per-cycle timing claim','the 65-70 s typical cycle claim'),
('in about 40 places (F-26)','in several places (F-26)'),
('about 50 calls used in this resumed session','about 27 Desktop Commander calls in this resumed session (earlier calls are in laptop-log.md)'),
('- [ ] app/tools/scan_pairs.jsonl (read first lines in the saved batch output; 9 lines, shown in full)  [x in practice]','- [x] app/tools/scan_pairs.jsonl (9 lines, shown in full in the saved batch output)'),
('The suite itself is fast and offline; no test writes outside its temp folder (run_tests.py removes it).','The suite is fast and offline. run_tests.py removes its temp folder; no other write check was done.')]
for a,b in reps:
    assert rv.count(a)==1,(a[:50],rv.count(a))
    rv=rv.replace(a,b)
open(D+'/review.md','w',encoding='utf-8').write(rv)
print('severity', dict(sev), 'ai_files', n_files)
