from pathlib import Path
import json,hashlib,time
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
r=Path('/home/intel/junction-cell/act-checkpoint-test');m=r/'model';p=m/'pretrained_model'
for line in (m/'final.sha256').read_text().splitlines():
 h,f=line.split();assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
cfg=PreTrainedConfig.from_pretrained(p);cfg.pretrained_path=p;cfg.device='xpu'
policy=ACTPolicy.from_pretrained(p,config=cfg).to('xpu').eval()
pre,post=make_pre_post_processors(cfg,pretrained_path=p,preprocessor_overrides={'device_processor':{'device':'xpu'}})
old=Path('/home/intel/junction-cell/blue-plate-checkpoint-test/trial-20261115T064344')
files=sorted(old.glob('observation-*.npz'));chosen=[files[0],files[len(files)//2],files[-1]]
results=[]
for f in chosen:
 a=np.load(f);batch={'observation.state':torch.from_numpy(a['state'].copy()).unsqueeze(0)}
 for source,key in [('camera1','observation.images.realsense 243622061421'),('camera2','observation.images.usb2_0_cam1 external camera')]:batch[key]=torch.from_numpy(a[source].copy()).permute(2,0,1).unsqueeze(0).float()/255
 policy.reset();t=time.monotonic()
 with torch.inference_mode():out=post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
 assert np.isfinite(out).all() and out.shape[1]==6
 results.append({'source':str(f),'state':a['state'].tolist(),'seconds':time.monotonic()-t,'first_action':out[0].tolist(),'first10_min':out[:10].min(0).tolist(),'first10_max':out[:10].max(0).tolist(),'out_of_bounds_fraction':float(np.mean((out<np.array([-99.9]*5+[.1]))|(out>99.9)))})
(r/'offline-check.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
