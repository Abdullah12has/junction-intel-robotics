"""Station-only supervised local measurements. Requires operator-ready flag.

Never modifies motor calibration. Small pan/lift perturbations return to the
captured reference after every sample. Photos and measured joint states only;
no calibration is auto-approved. Stop file checked between bounded moves.
"""
import argparse
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
    def photo(label,player):
        if stop.exists():raise RuntimeError('STOP requested')
        before=player.current()
        path=a.output/(label+'.png')
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-f','v4l2','-video_size','640x480','-i',a.camera,
                        '-frames:v','1',str(path)],check=True,timeout=5,stdin=subprocess.DEVNULL)
        state=player.current()
        if not np.isfinite(state).all() or np.max(np.abs(state-before))>.3:raise RuntimeError('Pose moved during capture')
        rows.append(dict(label=label,state=state.tolist(),image=path.name))
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
            # 0.4 normalized units is less than half a degree for these two calibrated joints.
            for phase,amount in [('fit',.4),('validation',.2)]:
                for joint in (0,1):
                    for sign in (-1,1):
                        if stop.exists():raise RuntimeError('STOP requested')
                        target=reference.copy();target[joint]+=sign*amount
                        result=player.move_to(target)
                        if not result.arrived:raise RuntimeError('Perturbation did not track')
                        photo(f'{phase}-{joint}-{sign}',player)
                        if stop.exists():raise RuntimeError('STOP requested; hold instead of returning')
                        if not player.move_to(reference).arrived:raise RuntimeError('Reference return failed')
                        photo(f'return-{phase}-{joint}-{sign}',player)
        finally:
            try:player.hold()
            finally:drv.disconnect()

if __name__=='__main__':main()
