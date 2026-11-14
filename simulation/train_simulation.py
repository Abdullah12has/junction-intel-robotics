"""Launch a guarded, simulation-only SmolVLA fine-tune in the GPU environment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from station_coordinates import UNITS, fingerprint


def command(dataset, output, steps=2000, minimum=140):
    manifest=json.loads((dataset/'simulation_provenance.json').read_text())
    episodes=manifest['episodes']
    train=manifest['train_episodes'];val=manifest['validation_episodes']
    if len(episodes)<minimum or not val or len(val)*3>len(train):
        raise ValueError(f'Require at least {minimum} verified episodes with a held-out split of at most 25%')
    if set(train)&set(val) or set(train+val)!=set(range(len(episodes))):
        raise ValueError('Invalid episode split')
    if {episodes[i]['metadata']['seed'] for i in train}&{episodes[i]['metadata']['seed'] for i in val}:
        raise ValueError('Scene leakage between training and validation')
    if any(not all(e['metadata'][k] for k in ('accepted','success','physical_grasp_verified')) for e in episodes):
        raise ValueError('Unverified episode')
    if manifest.get('normalization_episodes')!=train:raise ValueError('Require training-only normalization statistics')
    if manifest['source']!='simulation_only' or manifest['action_units']!=UNITS or manifest.get('state_units')!=UNITS:
        raise ValueError('Unexpected source/action convention')
    if fingerprint(manifest.get('coordinate_mapping')) != manifest.get('mapping_sha256'):
        raise ValueError('Mapping does not match its recorded fingerprint')
    if any(e['metadata'].get('mapping_sha256') != manifest['mapping_sha256'] or
           e['metadata'].get('action_units') != UNITS or e['metadata'].get('state_units') != UNITS for e in episodes):
        raise ValueError('Mixed or mislabelled episode coordinates')
    return [str(Path(sys.executable).parent/'lerobot-train'),
            '--policy.path=lerobot/smolvla_base','--policy.device=cuda',
            '--policy.use_amp=true','--policy.push_to_hub=false',
            f'--policy.scheduler_warmup_steps={max(1,steps//10)}',
            f'--policy.scheduler_decay_steps={steps}',
            '--dataset.repo_id=local/junction-sim-sorting',f'--dataset.root={dataset}',
            '--dataset.video_backend=pyav',f'--dataset.episodes={json.dumps(train)}',
            '--dataset.eval_split=0','--eval_steps=0','--env_eval_freq=0',
            f'--steps={steps}','--batch_size=32','--num_workers=4',
            '--save_freq=1000','--log_freq=20','--wandb.enable=false',
            f'--output_dir={output}','--job_name=sim-sorting-150','--seed=42']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=2000)
    a=p.parse_args();cmd=command(a.dataset,a.output,a.steps)
    print(json.dumps({'command':cmd,'simulation_only':True}),flush=True)
    subprocess.run(cmd,check=True)
