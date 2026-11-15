"""Launch the native MuJoCo viewer for closed-loop learned-policy evaluation."""
import os
from pathlib import Path
import subprocess
import sys

if __name__=='__main__':
    env=os.environ.copy();executable=Path(sys.executable)
    if sys.platform=='darwin':
        executable=executable.parent/'mjpython'
        old=env.get('DYLD_LIBRARY_PATH')
        env['DYLD_LIBRARY_PATH']=str(Path(sys.base_prefix)/'lib')+(os.pathsep+old if old else '')
    raise SystemExit(subprocess.call([str(executable),'-u',str(Path(__file__).with_name('rollout_policy.py')),*sys.argv[1:]],env=env))
