from pathlib import Path
import sys,json
import numpy as np
from PIL import Image
import av,mujoco
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'submission/assets'
sys.path.insert(0,str(ROOT/'simulation'))
from sorting_env import BrickSortingEnv
env=BrickSortingEnv(render=False);env.reset(seed=0,destination='blue')
env.model.vis.global_.offwidth=1600;env.model.vis.global_.offheight=1000
renderer=mujoco.Renderer(env.model,height=900,width=1440)
cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[0,-0.10,0.08];cam.distance=.8;cam.azimuth=130;cam.elevation=-35
renderer.update_scene(env.data,camera=cam);Image.fromarray(renderer.render()).save(OUT/'hero.png')
for camera in ['overhead','wrist_mounted']:
 renderer.update_scene(env.data,camera=camera);Image.fromarray(renderer.render()).save(OUT/(camera+'.png'))
renderer.close()
source=ROOT/'simulation/outputs/sorting-150/attempt-0000'
for name in ['overhead','wrist']:
 with av.open(str(source/(name+'.mp4'))) as c:
  for n,frame in enumerate(c.decode(video=0)):
   if n in [0,140,240,390]: frame.to_image().save(OUT/f'expert-{name}-{n}.png')
sys.path.insert(0,str(ROOT/'training/blue_plate'))
from variants import transform
im=np.array(Image.open(ROOT/'simulation/outputs/sorting/overhead.png').convert('RGB'))
Image.fromarray(transform(im,0,np.random.default_rng(0),overhead=True,xy=[399,253])).save(OUT/'target-x.png')
(OUT/'provenance.json').write_text(json.dumps({'hero':'New MuJoCo render of the repository sorting environment, seed 0','camera_views':'New simulation renders, not real photographs','expert_frames':str(source.relative_to(ROOT)),'expert_metadata':json.loads((source/'metadata.json').read_text()),'target_x':'Illustration using the actual training transform on a simulated scene; point manually selected for this slide'},indent=2))
print('Rendered simulation assets')
