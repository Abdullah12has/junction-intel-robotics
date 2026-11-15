import asyncio,json,time,uuid,math,urllib.request,pathlib
import websockets
import numpy as np, pyarrow.parquet as pq
BASE='http://127.0.0.1:7860/api'
PID='ab767473-d6fd-47ff-812e-8239c455dabb'
ENV='72206569-df16-4a8b-a98f-247bac4de63f'
MODEL='fa42ee76-bcfb-43ba-ba21-124e678ee785'
async def main():
 data=pathlib.Path('/home/intel/.local/share/physicalai/datasets/6d6bdce3-b382-478d-9697-fb9e573b97af/data')
 table=pq.read_table(sorted(data.rglob('*.parquet'))).to_pydict()
 values=np.concatenate([np.asarray(table[k],dtype=float) for k in ('observation.state','action')])
 names=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
 bounds={k:(max(0 if k=='gripper.pos' else -100,float(values[:,i].min())-5),min(100,float(values[:,i].max())+5)) for i,k in enumerate(names)}
 print('RANGE GUARDS',bounds,flush=True)
 frames_dir=pathlib.Path('/home/intel/junction-cell/evidence/act-full-test-frames');frames_dir.mkdir(exist_ok=True)
 last_saved={}

 report={'model_id':MODEL,'duration_limit_s':30,'hz':30,'joint_bounds':bounds,'samples':[],'events':[]}
 acks={}; state=None; received=0; cameras={}; failure=None
 async with websockets.connect(f'ws://127.0.0.1:7860/api/projects/{PID}/runtime/ws?hz=20',max_size=None) as ws:
  async def pump():
   nonlocal state,received,failure
   try:
    async for m in ws:
     if isinstance(m,bytes):
      n=int.from_bytes(m[:4],'big');h=json.loads(m[4:4+n]);now=time.monotonic()
      if h.get('state'):state=h['state'];received=now;report['samples'].append({'time':now,'state':state,'action':h.get('action')})
      offset=4+n
      for f in h.get('frames',[]):
       cameras[f['key']]=now
       if now-last_saved.get(f['key'],0)>1:
        (frames_dir/f"{f['key']}-{now:.2f}.jpg").write_bytes(m[offset:offset+f['size']]);last_saved[f['key']]=now
       offset+=f['size']
     else:
      j=json.loads(m)
      if j.get('type')=='ack':
       f=acks.get(j['request_id'])
       if f and not f.done():f.set_result(j)
      else:
       report['events'].append(j)
       if j.get('type')=='state':report['session_id']=j['data']['session_id']
       if j.get('type')=='error' or (j.get('type')=='status' and j.get('state')=='failed'):failure=str(j);print('ERROR',j,flush=True)
   except Exception as e:failure=repr(e)
  async def cmd(command,**kw):
   rid=uuid.uuid4().hex;f=asyncio.get_running_loop().create_future();acks[rid]=f
   await ws.send(json.dumps({'command':command,'request_id':rid,**kw}))
   r=await asyncio.wait_for(f,90 if command.startswith('load') else 5)
   if not r['ok']:raise RuntimeError(r)
   print(command,'OK',flush=True)
  p=asyncio.create_task(pump())
  try:
   envs=json.load(urllib.request.urlopen(f'{BASE}/projects/{PID}/environments'))
   env=next(e for e in envs if e['id']==ENV)
   env={k:env[k] for k in ('id','name','robots','cameras')}
   await cmd('load_environment',environment=env)
   await cmd('load_model',model_id=MODEL,backend='openvino',device='GPU')
   deadline=time.monotonic()+15
   while state is None or len(cameras)<2 or time.monotonic()-received>0.5:
    if failure:raise RuntimeError(failure)
    if time.monotonic()>deadline:raise TimeoutError('Live joint/camera preflight failed')
    await asyncio.sleep(.05)
   assert all(math.isfinite(v) and -100<=v<=100 for v in state.values()),state
   assert all(time.monotonic()-t<1 for t in cameras.values()),cameras
   initial=dict(state);report['initial_state']=initial;print('PREFLIGHT',initial,list(cameras),flush=True)
   started=time.monotonic()
   try:
    await cmd('set_control',control={'type':'model','task':'move the brick to the inspection spot','hz':30,'grace_s':3,'execution':{'mode':'sync','request_threshold':0.5}})
    while time.monotonic()-started<30:
     await asyncio.sleep(.05)
     if failure:raise RuntimeError(failure)
     if time.monotonic()-received>.5:raise RuntimeError('Joint feedback stale')
     if any(time.monotonic()-t>1 for t in cameras.values()):raise RuntimeError('Camera feedback stale')
     if any(not math.isfinite(v) or not bounds[k][0]<=v<=bounds[k][1] for k,v in state.items()):raise RuntimeError('Joint outside demonstrated range')
     action=report['samples'][-1].get('action') or {}
     if any(not math.isfinite(v) or not bounds[k][0]<=v<=bounds[k][1] for k,v in action.items()):raise RuntimeError('Target outside demonstrated range')
    report['result']='bounded test completed'
   finally:
    await cmd('set_control',control=None)
    report['control_stopped']=True
    report['elapsed_control_s']=time.monotonic()-started
   await asyncio.sleep(.5);report['final_state']=state
  except Exception as e:
   report['result']='stopped: '+repr(e);print(report['result'],flush=True)
   try:await cmd('set_control',control=None);report['control_stopped']=True
   except Exception as stop:report['stop_error']=repr(stop)
  finally:
   pathlib.Path('/home/intel/junction-cell/evidence/act-robot-full-test.json').write_text(json.dumps(report,indent=2))
   p.cancel()
 print(json.dumps({k:v for k,v in report.items() if k not in ('samples','events')},indent=2),flush=True)
asyncio.run(main())
