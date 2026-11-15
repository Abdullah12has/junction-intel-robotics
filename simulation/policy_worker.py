"""GPU-only inference worker over SSH stdio. It never controls a robot or simulator."""
import argparse
import base64
import io
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies import make_policy, make_pre_post_processors

PREFIX='VLA_RESPONSE '

def emit(value):print(PREFIX+json.dumps(value),flush=True)

def main():
    root=Path('/var/lib/physicalai-trainer/junction-simulation-training')
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=root/'smolvla-150/checkpoints/002000/pretrained_model')
    p.add_argument('--dataset',type=Path,default=root/'dataset-150')
    p.add_argument('--contract',type=Path,help='JSON with action_units and mapping_sha256, instead of the training dataset')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();checkpoint=a.checkpoint;device=a.device
    if a.contract:   # checkpoint copied off the training machine: its stats live in the saved processors
        contract=json.loads(a.contract.read_text())
    else:
        contract=json.loads((a.dataset/'simulation_provenance.json').read_text())
        training_config=json.loads((checkpoint/'train_config.json').read_text())
        if Path(training_config['dataset']['root']).resolve()!=a.dataset.resolve():raise ValueError('Dataset differs from checkpoint training configuration')
    units=contract['action_units']
    if units not in ('absolute_joint_radians','so101_station_normalized_v1'):raise ValueError('Unknown checkpoint units')
    cfg=PreTrainedConfig.from_pretrained(checkpoint)
    cfg.pretrained_path=checkpoint;cfg.device=device
    if a.contract:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy=SmolVLAPolicy.from_pretrained(checkpoint,config=cfg).to(device).eval()
    else:
        meta=LeRobotDatasetMetadata('local/junction-sim-sorting',root=a.dataset)
        policy=make_policy(cfg=cfg,ds_meta=meta).eval()
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=checkpoint,
                                    preprocessor_overrides={'device_processor':{'device':device}})
    emit({'ready':True,'chunk_size':cfg.chunk_size,'n_action_steps':cfg.n_action_steps,'checkpoint':str(checkpoint),'action_units':units,'mapping_sha256':contract.get('mapping_sha256')})
    for line in sys.stdin:
        try:
            request=json.loads(line)
            if request.get('action_units')!=units or request.get('mapping_sha256')!=contract.get('mapping_sha256'):
                raise ValueError('Inference coordinate contract mismatch')
            if request.get('reset'):
                policy.reset();torch.manual_seed(request['seed'])
            with np.load(io.BytesIO(base64.b64decode(request['observation'])),allow_pickle=False) as arrays:
                batch={'observation.state':torch.from_numpy(arrays['state'].copy()).unsqueeze(0),'task':[request['task']]}
                for name in ('camera1','camera2'):
                    batch['observation.images.'+name]=torch.from_numpy(arrays[name].copy()).permute(2,0,1).unsqueeze(0).float()/255.0
            start=time.monotonic()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=device=='cuda'):
                normalized=policy.predict_action_chunk(pre(batch))
                actions=post(normalized).float().cpu().numpy()[0,:cfg.n_action_steps]
            if actions.ndim!=2 or actions.shape[1]!=6 or not np.isfinite(actions).all():raise ValueError('Invalid policy output')
            emit({'actions':actions.tolist(),'inference_seconds':time.monotonic()-start})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr);emit({'error':str(exc)})

if __name__=='__main__':main()
