import json,pathlib,shutil,numpy as np,openvino as ov
from openvino import opset13 as ops
p=pathlib.Path('/home/intel/.local/share/physicalai/models/fa42ee76-bcfb-43ba-ba21-124e678ee785/exports/openvino')
core=ov.Core();m=core.read_model(p/'act.xml');assert len(m.outputs)==1 and m.output().partial_shape[-1].get_length()==6
low=np.array([-100]*5+[0],np.float32);high=np.array([100]*6,np.float32)
result=m.get_results()[0];old=result.input_value(0);names=set(m.output().get_names())
bounded=ops.minimum(ops.maximum(old,ops.constant(low)),ops.constant(high));bounded.output(0).set_names(names);result.input(0).replace_source_output(bounded.output(0));m.validate_nodes_and_infer_types()
ov.save_model(m,p/'act-bounded.xml',compress_to_fp16=False)
# Verify the added broadcast bounds with deliberately invalid test values.
x=ops.parameter([1,1,6],np.float32);y=ops.minimum(ops.maximum(x,ops.constant(low)),ops.constant(high));c=core.compile_model(ov.Model([y],[x]),'CPU');got=c([np.array([[[-101,101,-50,0,200,-.48]]],np.float32)])[0];np.testing.assert_array_equal(got,[[[-100,100,-50,0,100,0]]])
core.compile_model(p/'act-bounded.xml','GPU')
manifest=p/'manifest.json';backup=p/'original'
assert not backup.exists(),'Already patched; inspect before repeating'
backup.mkdir()
for name in ('act.xml','act.bin','manifest.json'):
 shutil.copy2(p/name,backup/name)
for ext in ('xml','bin'):
 shutil.copy2(p/f'act-bounded.{ext}',p/f'act.{ext}')
d=json.loads(manifest.read_text());d['model']['artifacts']['openvino']='act.xml';manifest.write_text(json.dumps(d,indent=2))
print('Bounded canonical export compiled on GPU; boundary test passed; originals retained')
