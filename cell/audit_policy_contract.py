"""Verify the frozen checkpoint files and training/deployment camera contract."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file


def audit(root):
    root=Path(root);model=root/'pretrained_model'
    checked=[]
    for line in (root/'final.sha256').read_text().splitlines():
        digest,name=line.split(maxsplit=1);path=model/name.lstrip('*')
        if path.parent.resolve()!=model.resolve():raise ValueError('Unexpected manifest path')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=digest:raise ValueError('Checkpoint hash mismatch: '+name)
        checked.append(name)
    cfg=json.loads((model/'config.json').read_text())
    contract=json.loads((root/'contract.json').read_text())
    provenance=json.loads((root/'cotrain_provenance.json').read_text())
    pre=json.loads((model/'policy_preprocessor.json').read_text())
    rename=next(s['config']['rename_map'] for s in pre['steps'] if s['registry_name']=='rename_observations_processor')
    cameras={k:rename.get(k,k) for k in provenance['cameras']}
    expected={'observation.images.realsense 243622061421':'observation.images.camera1',
              'observation.images.realsense 243722064316':'observation.images.camera2'}
    if cameras!=expected:raise ValueError('Camera contract differs from this station')
    if cfg['input_features']['observation.state']['shape']!=[6] or cfg['output_features']['action']['shape']!=[6]:
        raise ValueError('Joint dimension mismatch')
    if contract['action_units']!='so101_station_normalized_v1' or provenance['units']!=contract['action_units']:
        raise ValueError('Joint units mismatch')
    for file in model.glob('*processor.safetensors'):
        for key,value in load_file(file).items():
            if not np.isfinite(value).all():raise ValueError('Nonfinite normalization: '+key)
            if key.endswith('std') and np.any(value<=0):raise ValueError('Invalid normalization scale: '+key)
    return dict(checkpoint=str(model),files_verified=checked,action_units=contract['action_units'],mapping_sha256=contract['mapping_sha256'],
                trained_cameras=cameras,unused_config_cameras=[k for k in cfg['input_features'] if '.images.' in k and k not in cameras.values()],
                real_tasks=sorted({e['task'] for e in provenance['episodes'] if e['source']=='real'}),chunk_size=cfg['chunk_size'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');a=p.parse_args();print(json.dumps(audit(a.root),indent=2))
