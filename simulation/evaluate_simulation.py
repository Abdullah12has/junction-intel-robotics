"""Offline held-out loss; this does not measure closed-loop task success."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.utils.collate import lerobot_collate_fn


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.manual_seed(42)
    manifest=json.loads((a.dataset/'simulation_provenance.json').read_text())
    cfg=PreTrainedConfig.from_pretrained(a.checkpoint)
    cfg.pretrained_path=a.checkpoint;cfg.device='cuda'
    meta=LeRobotDatasetMetadata('local/junction-sim-sorting',root=a.dataset)
    ds=LeRobotDataset('local/junction-sim-sorting',root=a.dataset,
                     episodes=manifest['validation_episodes'],delta_timestamps=resolve_delta_timestamps(cfg,meta),
                     video_backend='pyav',return_uint8=True)
    policy=make_policy(cfg=cfg,ds_meta=meta).eval()
    pre,post=make_pre_post_processors(policy_cfg=cfg,pretrained_path=a.checkpoint,
                                     preprocessor_overrides={'device_processor':{'device':'cuda'}})
    indices=np.linspace(0,len(ds)-1,min(960,len(ds)),dtype=int).tolist()
    loader=DataLoader(Subset(ds,indices),batch_size=32,num_workers=2,collate_fn=lerobot_collate_fn)
    losses=[];start=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for raw in loader:
            for key in ds.meta.camera_keys:
                if raw[key].dtype==torch.uint8:raw[key]=raw[key].float()/255.0
            batch=pre(raw);loss,_=policy.forward(batch)
            if not torch.isfinite(loss):raise ValueError('Non-finite held-out loss')
            losses.append(float(loss))
    result=dict(validation_episodes=manifest['validation_episodes'],sampled_frames=len(indices),
                mean_flow_matching_loss=float(np.mean(losses)),finite=True,seconds=time.monotonic()-start,
                closed_loop_task_success_tested=False,hardware_compatible=False)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
