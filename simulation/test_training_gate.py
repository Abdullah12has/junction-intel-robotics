import json
import pytest
from train_simulation import command
from station_coordinates import UNITS, fingerprint


def manifest():
    train=list(range(120))
    from station_coordinates import NOMINAL as mapping
    digest=fingerprint(mapping)
    return dict(source='simulation_only',action_units=UNITS,state_units=UNITS,
                joint_mapping_verified=False,coordinate_mapping=mapping,mapping_sha256=digest,
                episodes=[{'metadata':dict(seed=i//2,accepted=True,success=True,physical_grasp_verified=True,
                                               action_units=UNITS,state_units=UNITS,mapping_sha256=digest)} for i in range(150)],
                train_episodes=train,validation_episodes=list(range(120,150)),normalization_episodes=train)


def test_training_command_excludes_validation_and_publishing(tmp_path):
    data=manifest();(tmp_path/'simulation_provenance.json').write_text(json.dumps(data))
    cmd=command(tmp_path,tmp_path/'model')
    assert '--policy.push_to_hub=false' in cmd
    episodes=json.loads(next(s.split('=',1)[1] for s in cmd if s.startswith('--dataset.episodes=')))
    assert set(episodes)==set(range(120))


@pytest.mark.parametrize('problem',['missing_episode','scene_leak','failed_grasp','validation_stats','radians','mixed_mapping'])
def test_training_rejects_incomplete_or_leaked_data(tmp_path,problem):
    data=manifest()
    if problem=='missing_episode':data['episodes'].pop(0)
    if problem=='scene_leak':data['episodes'][120]['metadata']['seed']=0
    if problem=='failed_grasp':data['episodes'][0]['metadata']['physical_grasp_verified']=False
    if problem=='validation_stats':data['normalization_episodes']=list(range(150))
    if problem=='radians':data['action_units']='absolute_joint_radians'
    if problem=='mixed_mapping':data['episodes'][3]['metadata']['mapping_sha256']='different'
    (tmp_path/'simulation_provenance.json').write_text(json.dumps(data))
    with pytest.raises(ValueError):command(tmp_path,tmp_path/'model')
