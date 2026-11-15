"""Build a sub-two-minute pitch with 3x physical footage and an expert sim clip."""
from pathlib import Path
import subprocess,json
r=Path(__file__).resolve().parents[2];build=r/'submission/build/revision';out=r/'submission/output'
chapters=[
(1,'Prometheus brings visual intelligence into physical action. Our SO one oh one robot inspects real parts and sorts them, with Smol V L M assisted vision.'),
('real','Here is the working physical demonstration, played at three times speed. Smol V L M assists the vision guided controller. The overhead camera shows the workspace, while wrist and close cameras show the inspection views. The system observes each brick, uses visual information to guide the task, then moves and releases it onto the blue plate. The supplied demo records three completed good to blue sorts, with a reported median cycle time of forty eight seconds. This is a selected three part demonstration, rather than a broad reliability benchmark. Inspection confidence is marked partial in the recording. What matters is that the system connects perception to physical action on the real station. We keep this Smol V L M assisted execution separate from our Smol V L A action policy training experiments.'),
('sim','We also built this MuJoCo environment. Watch the expert controller pick up the block, transport it, and release it onto the blue plate. This is successful simulated manipulation.'),
(6,'Our learning pipeline is implemented: sixty four demonstrations, grouped evaluation splits, and thirty two configurations comparing ACT, Smol V L A, target cues and augmentation.'),
(8,'Physical AI Studio connects the station workflows. Our Intel deployment tools include OpenVINO inference checks, inspection code, bounded outputs and explicit stop paths.'),
(9,'Our ambition is adaptable inspection for smaller factories. We have a working physical demo, a simulation environment, and a reproducible learning pipeline. The next validation expands to defects, clutter and learned policy transfer.')]
segments=[];records=[]
for i,(kind,words) in enumerate(chapters):
 txt=build/f'v2-voice-{i}.txt';audio=build/f'v2-voice-{i}.aiff';txt.write_text(words)
 subprocess.run(['say','-v','Samantha','-r','185','-f',str(txt),'-o',str(audio)],check=True)
 d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(audio)]));duration=d+.3
 dest=build/f'v2-segment-{i}.mp4'
 if kind in ('real','sim'):
  src=out/'prometheus-demo-3x.mp4' if kind=='real' else r/'simulation/outputs/sorting-150/attempt-0000/overhead.mp4'
  vd=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','default=nw=1:nk=1',str(src)]));duration=max(duration,vd)
  size='1152:648' if kind=='real' else '864:648';x=64 if kind=='real' else 208
  filters=f'[0:v]scale={size},tpad=stop_mode=clone:stop_duration=15[v];[2:v][v]overlay={x}:64[out]'
  inputs=['-i',str(src),'-i',str(audio),'-loop','1','-i',str(build/('video-'+kind+'.png')),'-filter_complex',filters,'-map','[out]','-map','1:a']
 else:
  inputs=['-loop','1','-i',str(build/f'slide-{kind}.png'),'-i',str(audio),'-vf','scale=1280:720']
 subprocess.run(['ffmpeg','-y','-v','error',*inputs,'-af','apad','-t',str(duration),'-r','30','-c:v','libx264','-preset','fast','-crf','23','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',str(dest)],check=True)
 segments.append(dest);records.append({'segment':kind,'narration':words,'duration_s':duration});print(kind,round(duration,2),flush=True)
manifest=build/'v2-concat.txt';manifest.write_text(''.join("file '"+str(s)+"'\n" for s in segments))
subprocess.run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(manifest),'-c','copy','-movflags','+faststart',str(out/'prometheus-demo-narrated.mp4')],check=True)
(r/'submission/video-v2-manifest.json').write_text(json.dumps({'real_source':'THIRD_PARTY_NOTICES.mp4','real_playback_speed':3,'real_controller':'SmolVLM-assisted vision-guided, per user clarification','simulation_controller':'MuJoCo expert','voice':'macOS Samantha synthetic narration at 185 words per minute','segments':records},indent=2)+'\n')
print('Total',sum(x['duration_s'] for x in records))
