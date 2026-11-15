"""Find diverse, contact-verified scenarios before expensive video recording."""
import argparse
from collections import Counter
import json
from pathlib import Path

from expert import PickPlaceExpert, PlanningError
from sorting_env import BrickSortingEnv, COLORS
from station_coordinates import StationCoordinates, constrain_simulation


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--episodes',type=int,default=150)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--max-seeds',type=int,default=3000)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mapping',type=Path,help='Joint mapping JSON; default is the nominal mapping')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    pairs=[(c,d) for c in COLORS for d in ('blue','green')]
    quotas={pair:a.episodes//len(pairs) for pair in pairs}
    # Spread remainder across colors and destinations.
    extras=[(c,('blue','green')[i%2]) for i,c in enumerate(COLORS)]
    extras += [pair for pair in pairs if pair not in extras]
    for pair in extras[:a.episodes%len(pairs)]:quotas[pair]+=1
    counts=Counter();results=[]
    env=BrickSortingEnv(horizon=1800)
    coordinates=StationCoordinates(env.limits,mapping=a.mapping)
    constrain_simulation(env,coordinates)
    try:
        for seed in range(a.seed,a.seed+a.max_seeds):
            env.reset(seed)
            # Alternate target geometry while allowing all colors and layouts.
            index=3 if seed%2==0 else next(i for i in env.active_indices if i!=3)
            color=env.block_colors[index]
            # Station rule: squares (2x2) are good and go to blue; rectangles (2x4) are defective and go to green.
            for destination in ({'2x2':'blue','2x4':'green'}[env.blocks[index]['type']],):
                if counts[color,destination]>=quotas[color,destination]:continue
                env.reset(seed,target_color=color,destination=destination)
                expert=PickPlaceExpert(env)
                error=None
                try:
                    success=bool(expert.run())
                    success=success and any(len(t['gripper_contact_bodies'])==2 and t['object_xyz'][2]>.06 for t in expert.trace)
                except PlanningError as exc:success=False;error=str(exc)
                row=dict(seed=seed,color=color,destination=destination,type=env.blocks[env.target_index]['type'],success=success,error=error)
                results.append(row)
                if success:counts[color,destination]+=1
                a.output.write_text(json.dumps(results,indent=2)+'\n')
                if success or len(results)%50==0:
                    print(f'{sum(counts.values())}/{a.episodes} verified; {len(results)} attempts; seed {seed}; {color} to {destination}: {success}',flush=True)
                if sum(counts.values())==a.episodes:return
        raise RuntimeError(f'Only {sum(counts.values())} successes; increase seed budget')
    finally:env.close()

if __name__=='__main__':main()
