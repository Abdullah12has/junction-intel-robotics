"""Render the model's zero and positive joint directions for physical comparison."""
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from so101_env import BrickInspectionEnv, JOINTS

root=Path('outputs/joint-mapping');root.mkdir(parents=True,exist_ok=True)
env=BrickInspectionEnv()
env.model.geom_rgba[env.brick_geoms,3]=0
env.model.site('inspection_target').rgba[3]=0
renderer=mujoco.Renderer(env.model,height=480,width=640)
def render(q,azimuth=90,elevation=-10):
 env.data.qpos[env.qadr]=q;mujoco.mj_forward(env.model,env.data)
 camera=mujoco.MjvCamera();camera.lookat[:]=[.04,0,.18];camera.distance=.65;camera.azimuth=azimuth;camera.elevation=elevation
 renderer.update_scene(env.data,camera=camera)
 return Image.fromarray(renderer.render().copy())
out=Image.new('RGB',(1280,520),(22,26,32));d=ImageDraw.Draw(out)
for i,(az,el,label) in enumerate([(90,-10,'Model zero: side view'),(150,-25,'Model zero: angled view')]):
 out.paste(render(np.zeros(6),az,el),(640*i,40));d.text((640*i+15,12),label,fill='white')
out.save(root/'zero-reference.png')
for i,name in enumerate(JOINTS):
 frames=[]
 for angle in [0]*8+list(np.linspace(0,np.deg2rad(15),15))+[np.deg2rad(15)]*8+list(np.linspace(np.deg2rad(15),0,15)):
  q=np.zeros(6);q[i]=angle;im=render(q,150,-25)
  ImageDraw.Draw(im).text((12,12),f'{name}: model +{np.rad2deg(angle):.1f} degrees',fill='white');frames.append(im)
 frames[0].save(root/f'{name}-positive.gif',save_all=True,append_images=frames[1:],duration=80,loop=0)
renderer.close();env.close()
