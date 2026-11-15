#!/usr/bin/env python3
"""P3-TWIN-DYN E1: probe MuJoCo 4.0.1 APIs (rollout, MjSpec, XML save) and step speed on the no-geometry URDF."""
import time

import mujoco

URDF = "/home/intel/intel-junction-2026/work/P3-TWIN-DYN/E1/urdf/so101_nogeom.urdf"
print("mujoco", mujoco.__version__)
print("has mujoco.rollout:", hasattr(mujoco, "rollout"))
try:
    from mujoco import rollout  # noqa: F401
    print("from mujoco import rollout: ok", rollout.rollout.__doc__.strip().splitlines()[0][:200])
except Exception as e:  # noqa: BLE001
    print("rollout import failed:", repr(e))
print("has MjSpec:", hasattr(mujoco, "MjSpec"))
if hasattr(mujoco, "MjSpec"):
    spec = mujoco.MjSpec.from_file(URDF)
    print("spec has add_actuator:", hasattr(spec, "add_actuator"), "| spec.compile:", hasattr(spec, "compile"))
    m_spec = spec.compile()
    print("spec compile nq", m_spec.nq, "nbody", m_spec.nbody)
m = mujoco.MjModel.from_xml_path(URDF)
try:
    mujoco.mj_saveLastXML("/home/intel/intel-junction-2026/work/P3-TWIN-DYN/E1/tools/probe_last.xml", m)
    txt = open("/home/intel/intel-junction-2026/work/P3-TWIN-DYN/E1/tools/probe_last.xml").read()
    print("mj_saveLastXML ok, length", len(txt), "| has <joint:", "<joint" in txt, "| has <body:", "<body" in txt)
except Exception as e:  # noqa: BLE001
    print("mj_saveLastXML failed:", repr(e))
d = mujoco.MjData(m)
t0 = time.time()
for _ in range(10000):
    mujoco.mj_step(m, d)
print("10000 mj_step on URDF model: %.3f s" % (time.time() - t0), "| integrator", m.opt.integrator)
