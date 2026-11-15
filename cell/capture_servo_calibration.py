"""Station-only supervised local measurements. Requires operator-ready flag.

Never modifies motor calibration. Small pan/lift perturbations return to the
captured reference after every sample. Photos and measured joint states only;
no calibration is auto-approved. Stop file checked between bounded moves.
"""
import argparse
from collections import deque
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--camera',required=True)
    p.add_argument('--operator-ready',action='store_true')
    p.add_argument('--reference-only',action='store_true')
    a=p.parse_args()
    if not a.operator_ready:p.error('Station operator must be present and hands clear')
    if not Path(a.camera).exists():p.error('Camera missing')
    if subprocess.run(['fuser',a.camera],capture_output=True).returncode != 1:p.error('Camera owned or check failed')
    sys.path.insert(0,str(Path.home()/'intel-junction-2026/app/motion'))
    from so101_motion import claim_devices,device_ids_for,resolve_arm,open_follower_held,MotionPlayer
    a.output.mkdir(parents=True,exist_ok=False)
    stop=a.output.parent/'STOP'
    if stop.exists():raise RuntimeError('STOP file present')
    rows=[]
    reference_pixel=None
    def locate_cyan(path):
        # Conservative calibration-only cyan mask in the full wrist image.
        # Refuse low visibility; the offline connected-component review remains mandatory.
        raw=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(path),'-f','rawvideo','-pix_fmt','rgb24','-'],
                           check=True,capture_output=True,timeout=5).stdout
        image=np.frombuffer(raw,np.uint8).reshape(480,640,3).astype(float)
        r,g,b=image.transpose(2,0,1)
        mask=(g>45)&(b>45)&(r<.65*g)&(b>.8*g)
        seen=np.zeros_like(mask);components=[]
        for y,x in zip(*np.nonzero(mask)):
            if seen[y,x]:continue
            queue=deque([(int(y),int(x))]);seen[y,x]=True;points=[]
            while queue:
                yy,xx=queue.popleft();points.append((yy,xx))
                for ny,nx in ((yy-1,xx),(yy+1,xx),(yy,xx-1),(yy,xx+1)):
                    if 0<=ny<mask.shape[0] and 0<=nx<mask.shape[1] and mask[ny,nx] and not seen[ny,nx]:
                        seen[ny,nx]=True;queue.append((ny,nx))
            if len(points)>=1000:components.append(np.array(points))
        if len(components)!=1:raise RuntimeError('Cyan target missing or ambiguous in calibration ROI')
        y,x=components[0].T
        if len(x)>150000 or min(x)==0 or max(x)==639 or min(y)==0 or max(y)==479:
            raise RuntimeError('Cyan target clipped in calibration ROI')
        return np.array([x.mean(),y.mean()])
    def photo(label,player):
        nonlocal reference_pixel
        if stop.exists():raise RuntimeError('STOP requested')
        before=player.current()
        path=a.output/(label+'.png')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','v4l2','-video_size','640x480','-i',a.camera,
                        '-frames:v','1',str(path)],check=True,timeout=5,stdin=subprocess.DEVNULL)
        state=player.current()
        if not np.isfinite(state).all() or np.max(np.abs(state-before))>.3:raise RuntimeError('Pose moved during capture')
        (a.output/(label+'-state.json')).write_text(json.dumps({'state':state.tolist(),'image':path.name}))
        pixel=locate_cyan(path)
        if reference_pixel is None:reference_pixel=pixel.copy()
        if np.linalg.norm(pixel-reference_pixel)>25:raise RuntimeError('Target moved beyond small calibration region')
        rows.append(dict(label=label,state=state.tolist(),image=path.name,target_uv=pixel.tolist()))
        (a.output/'measurements.json').write_text(json.dumps(rows,indent=2)+'\n')
        return state
    port,cal=resolve_arm(studio_export=str(Path.home()/'intel-junction-2026/app/motion/project_robots.json'),name='SO101 Follower',follower=True)
    with claim_devices(device_ids_for(port),ports=(port,)):
        drv=open_follower_held(port,cal)
        player=MotionPlayer(drv,max_speed=.5,max_accel=1.,tolerance=.3,settle_s=.5)
        try:
            reference=photo('reference',player)
            if a.reference_only:return
            if np.any(reference[:5]<-98) or np.any(reference[:5]>98):raise RuntimeError('Insufficient joint-limit margin')
            # One normalized unit is approximately one degree for these calibrated joints.
            for phase,amount in [('fit',1.0),('validation',.7)]:
                for joint in (0,1):
                    for sign in (-1,1):
                        if stop.exists():raise RuntimeError('STOP requested')
                        target=reference.copy();target[joint]+=sign*amount
                        before=player.current()
                        result=player.move_to(target)
                        measured=player.current()
                        # Fit against measured encoder motion, never the requested target.
                        # A sub-degree command may fall within servo deadband. Preserve that
                        # residual as evidence; only bounded actual movement is a valid sample.
                        movement=measured-before
                        others=np.delete(movement,joint)
                        if (not np.isfinite(measured).all() or np.max(np.abs(measured-reference))>1.5
                            or np.max(np.abs(others))>.3 or sign*movement[joint]<.15
                            or sign*movement[joint]>1.5):
                            (a.output/'tracking-failure.json').write_text(json.dumps({'target':target.tolist(),'measured':measured.tolist(),'before':before.tolist(),'max_error':result.max_error}))
                            raise RuntimeError('Measured perturbation has insufficient, wrong-direction or excessive movement')
                        photo(f'{phase}-{joint}-{sign}',player)
                        rows[-1]['requested_target']=target.tolist()
                        rows[-1]['target_error']=result.max_error
                        if stop.exists():raise RuntimeError('STOP requested; hold instead of returning')
                        player.move_to(reference)
                        if np.max(np.abs(player.current()-reference))>.75:raise RuntimeError('Reference return drift exceeds local calibration region')
                        photo(f'return-{phase}-{joint}-{sign}',player)
        finally:
            try:player.hold()
            finally:drv.disconnect()

if __name__=='__main__':main()
