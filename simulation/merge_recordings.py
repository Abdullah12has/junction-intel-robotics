"""Consolidate completed recorder shards, retaining source attempt provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('inputs',nargs='+',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected',type=int,default=150)
    p.add_argument('--schedule',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[];seen=set()
    schedule={(x['seed'],x['color'],x['destination']):x for x in json.loads(a.schedule.read_text()) if x['success']}
    for root in a.inputs:
        for meta in json.loads((root/'summary.json').read_text()):
            if not meta['accepted']:continue
            key=(meta['seed'],meta['target_color'],meta['destination'])
            if key in seen:raise ValueError(f'Duplicate scenario {key}')
            if key not in schedule:raise ValueError(f'Unscheduled scenario {key}')
            seen.add(key);meta['brick_type']=schedule[key]['type']
            rows.append((root/f"attempt-{meta['attempt']:04d}",meta))
    contracts={(m.get('action_units'),m.get('mapping_sha256')) for _,m in rows}
    if len(contracts)!=1:raise ValueError('Mixed coordinate systems/mappings')
    if len(rows)!=a.expected:raise ValueError(f'Need {a.expected} successful episodes; have {len(rows)}')
    a.output.mkdir(parents=True)
    summary=[]
    for i,(source,meta) in enumerate(sorted(rows,key=lambda x:(x[1]['seed'],x[1]['destination']))):
        destination=a.output/f'attempt-{i:04d}'
        shutil.copytree(source,destination)
        meta.update(source_attempt=str(source),attempt=i)
        (destination/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        summary.append(meta)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    sources=['expert.py','sorting_env.py','station_scene.py','so101_env.py','collect.py','setup_assets.py','uv.lock']
    (a.output/'recording_context.json').write_text(json.dumps({
        'source_hashes':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sources},
        'preflight_sha256':hashlib.sha256(a.schedule.read_bytes()).hexdigest(),
        'source':'mujoco_simulation','action_units':rows[0][1]['action_units'],
        'mapping_sha256':rows[0][1].get('mapping_sha256')},indent=2)+'\n')
    print(f'Merged {len(rows)} verified scenarios into {a.output}')

if __name__=='__main__':main()
