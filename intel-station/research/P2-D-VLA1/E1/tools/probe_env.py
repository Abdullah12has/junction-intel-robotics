import importlib.metadata as md
import shutil
import sys

print("python", sys.version.split()[0], sys.executable)
for p in ["lerobot", "av", "pyarrow", "numpy", "pandas", "opencv-python", "opencv-python-headless",
          "imageio", "imageio-ffmpeg", "torch", "huggingface-hub", "pillow", "pyyaml"]:
    try:
        print("pkg", p, md.version(p))
    except Exception as exc:
        print("pkg", p, "MISSING", type(exc).__name__)
for b in ["ffmpeg", "ffprobe"]:
    print("bin", b, shutil.which(b))
try:
    import av
    print("pyav", av.__version__)
    for c in ["libsvtav1", "libaom-av1", "libdav1d", "av1", "libx264"]:
        print("codec", c, c in av.codecs_available)
except Exception as exc:
    print("pyav import failed", repr(exc))
for mod in ["lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"]:
    try:
        m = __import__(mod, fromlist=["CODEBASE_VERSION"])
        print("module", mod, "CODEBASE_VERSION", getattr(m, "CODEBASE_VERSION", None))
    except Exception as exc:
        print("module", mod, "failed", type(exc).__name__, str(exc)[:200])
