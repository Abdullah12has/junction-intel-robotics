import json, pathlib, shutil
import numpy as np, openvino as ov
from openvino import opset13 as ops
p=pathlib.Path('/home/intel/.local/share/physicalai/models/6540166d-c6c7-4087-ba07-ec5571c37d3b/exports/openvino')
d=json.loads((p/'manifest.json').read_text());post=d['model']['postprocessors'];assert len(post)==1 and post[0]['type']=='denormalize' and post[0]['mode']=='mean_std'
stats=post[0]['stats']['action'];mean=np.array(stats['mean']);std=np.array(stats['std']);assert np.all(std>0)
lo=np.array([-99.9]*5+[0.1]);hi=np.array([99.9]*6)
low=((lo-mean)/std).astype(np.float32);high=((hi-mean)/std).astype(np.float32)
backup=p/'original';assert backup.exists(), 'Expected preserved original'
core=ov.Core();m=core.read_model(backup/'smolvla.xml');assert len(m.outputs)==1 and m.output().partial_shape[-1].get_length()==6
r=m.get_results()[0];old=r.input_value(0);names=set(m.output().get_names());bounded=ops.minimum(ops.maximum(old,ops.constant(low)),ops.constant(high));bounded.output(0).set_names(names);r.input(0).replace_source_output(bounded.output(0));m.validate_nodes_and_infer_types()
ov.save_model(m,p/'smolvla-bounded.xml',compress_to_fp16=False)
for ext in ('xml','bin'):shutil.copy2(p/f'smolvla-bounded.{ext}',p/f'smolvla.{ext}')
print('Bounded normalized outputs to denormalized calibrated range with 0.1-unit margin. Original retained.')
