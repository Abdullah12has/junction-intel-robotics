from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parents[2];build=r/'submission/build';out=r/'submission/output'
chapters=[
(1,'Prometheus is a robot learning and inspection workbench for the Intel Physical AI challenge. These visuals show our MuJoCo simulation.'),
(2,'The goal is to pick blocks from clutter and place them correctly. Our learning campaign focuses on blue plate placement. A separate application adds quality inspection.'),
(3,'An overhead camera shows the target and destination. A wrist camera provides local grasp geometry. Matching these views is essential before transferring simulation data.'),
(4,'We compare ordinary images with a causal X cue. A selected target is tracked only while confidence holds. Whether this improves learned behavior remains an experiment.'),
(5,'We prepared sixty four demonstrations and a thirty two experiment matrix across ACT, Smol VLA, target cues, augmentation and data sources. Real data training launched. Mixed simulation runs remain gated.'),
('sim','This is a recorded successful expert rollout in MuJoCo. The arm grasps, transports and releases the block onto the blue plate. This is simulation, not a real arm or learned policy success claim.'),
(7,'Physical AI Studio connects collection and policy deployment. We implemented OpenVINO inference checks, Intel station deployment tools, and an anomaly inspection application.'),
(8,'Real policy driven motion has been tested. Reliable physical pick and place is still unproven. We added output checks after reproducing out of range policy targets.'),
(9,'Prometheus makes the work inspectable through source, data contracts, simulation and test evidence. The next milestone is comparable real world trials across the same evaluation scenes.')]
segments=[];transcript=[]
for i,(slide,words) in enumerate(chapters):
 txt=build/f'voice-{i}.txt';audio=build/f'voice-{i}.aiff';txt.write_text(words);transcript.append(words)
 subprocess.run(['say','-v','Samantha','-r','170','-f',str(txt),'-o',str(audio)],check=True)
 d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(audio)]));duration=max(d+.35,13.47 if slide=='sim' else 2)
 dest=build/f'segment-{i}.mp4'
 if slide=='sim':
  inputs=['-i',str(r/'simulation/outputs/sorting-150/attempt-0000/overhead.mp4')]
  vf="scale=960:720,pad=1280:720:160:0:color=0x101820,tpad=stop_mode=clone:stop_duration=10,drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:text='MUJOCO EXPERT DEMO - NOT REAL ROBOT FOOTAGE':fontcolor=white:fontsize=24:box=1:boxcolor=black@0.8:x=(w-tw)/2:y=25"
 else:inputs=['-loop','1','-i',str(build/f'slide-{slide}.png')];vf='scale=1280:720'
 subprocess.run(['ffmpeg','-y','-v','error',*inputs,'-i',str(audio),'-vf',vf,'-af','apad','-t',str(duration),'-r','30','-c:v','libx264','-preset','fast','-crf','21','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',str(dest)],check=True)
 segments.append(dest);print(i,round(duration,2),flush=True)
manifest=build/'video-concat.txt';manifest.write_text(''.join("file '"+str(s)+"'\n" for s in segments))
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(out/'prometheus-pitch.mp4')],check=True)
(r/'submission/PITCH-SCRIPT.md').write_text('# Prometheus pitch video\n\nNarration uses local synthetic speech. Simulation footage is explicitly labeled.\n\n'+'\n\n'.join(transcript)+'\n')
