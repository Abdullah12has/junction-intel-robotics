#!/bin/bash
# Facts for the session audit: code, tests, runs, data, models, work items. Read-only.
R=/home/intel/intel-junction-2026
cd $R || exit 1
echo "=== utc $(date -u +%FT%TZ)"
echo "=== git app"; git -C app log --oneline | head -20; echo "commits: $(git -C app rev-list --count HEAD)"; git -C app status --short | head -20
echo "=== app tree"; find app -maxdepth 2 -type d | grep -v __pycache__ | sort | head -60
echo "=== py files / lines"; find app -name "*.py" -not -path "*/__pycache__/*" | wc -l; find app -name "*.py" -not -path "*/__pycache__/*" -exec cat {} + | wc -l
echo "=== sort runs"
for r in logs/sort/*/; do n=$(ls -d $r/ep_* 2>/dev/null | wc -l); c=$r/cycles.jsonl; s=0; f=0; [ -f $c ] && s=$(grep -c '"success": true' $c) && f=$(grep -c '"success": false' $c); echo "$(basename $r): ep_dirs=$n success=$s fail=$f"; done
echo "=== other robot logs"; ls logs | head -40
echo "=== data"; du -sh data/* 2>/dev/null | head -20; ls data/studio-datasets 2>/dev/null | head -20
echo "=== datasets built by us"; find data work -maxdepth 3 -name "info.json" -path "*meta*" 2>/dev/null | head -20
echo "=== models"; find models -maxdepth 3 -type d 2>/dev/null | head -40; du -sh models 2>/dev/null
echo "=== submission"; ls -la submission 2>/dev/null; wc -l submission/*.md 2>/dev/null
echo "=== docs"; ls docs 2>/dev/null
echo "=== work status files"; for s in work/*/E*/STATUS.md work/*/V*/STATUS.md work/*/F*/STATUS.md; do [ -f $s ] && echo "$s: $(head -1 $s | cut -c1-140)"; done 2>/dev/null | tail -60
echo "=== sim"; ls sim 2>/dev/null | head; ls sim/twin 2>/dev/null | head -20
echo "=== running jobs"; ps -eo pid,etime,cmd | grep -E "python|ovbench|pass2|sim_gate|sync_" | grep -v grep | cut -c1-150 | head -20
echo "=== disk"; df -h $R | tail -1
