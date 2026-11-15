import json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
r=Path('/home/intel/junction-cell/blue-plate-checkpoint-test')
m=r/'model';p=m/'pretrained_model'
for line in (m/'final.sha256').read_text().splitlines():
 h,f=line.split();assert hashlib.sha256((p/f).read_bytes()).hexdigest()==h
raw=json.loads((r/'read-only-state.json').read_text());names=list(raw['calibration'])
state=[]
for n in names:
 c=raw['calibration'][n];f=(raw['ticks'][n]-c['range_min'])/(c['range_max']-c['range_min']);state.append(100*f if n=='gripper' else 200*f-100)
contract={'action_units':'so101_station_normalized_v1','mapping_sha256':hashlib.sha256(json.dumps({'calibration':raw['calibration'],'joint_order':names},sort_keys=True).encode()).hexdigest(),'joint_order':names,'calibration':raw['calibration'],'training_task':'Pick up a cube and put it on the blue plate.','camera1':'overhead RealSense 243622061421','camera2':'replacement HBVCAM wrist; operator confirmed same mounting/framing'}
(m/'contract.json').write_text(json.dumps(contract,indent=2))
cfg=PreTrainedConfig.from_pretrained(p);cfg.pretrained_path=p;cfg.device='xpu';cfg.load_vlm_weights=False
policy=SmolVLAPolicy.from_pretrained(p,config=cfg).to('xpu').eval()
pre,post=make_pre_post_processors(cfg,pretrained_path=p,preprocessor_overrides={'device_processor':{'device':'xpu'}})
batch={'observation.state':torch.tensor([state],dtype=torch.float32),'task':[contract['training_task']]}
for key,file in [('camera1','overhead.jpg'),('camera2','usb-camera.jpg')]:batch['observation.images.'+key]=torch.from_numpy(np.array(Image.open(r/file))).permute(2,0,1).unsqueeze(0).float()/255
results=[]
for i in range(3):
 policy.reset();torch.manual_seed(42);t=time.monotonic()
 with torch.inference_mode():actions=post(policy.predict_action_chunk(pre(batch))).float().cpu().numpy()[0]
 assert actions.shape==(50,6) and np.isfinite(actions).all()
 results.append({'elapsed_s':time.monotonic()-t,'first_action':actions[0].tolist(),'min':actions.min(0).tolist(),'max':actions.max(0).tolist()})
np.save(r/'preflight-actions.npy',actions)
report={'hashes_verified':True,'device':'xpu','state':state,'finite':True,'shape':list(actions.shape),'results':results,'hardware_commands_sent':False}
(r/'offline-preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
