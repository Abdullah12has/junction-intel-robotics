import os
import sys

import av
import PIL

print("av", av.__version__, "pillow", PIL.__version__)
try:
    import matplotlib
    print("matplotlib", matplotlib.__version__)
except Exception as exc:
    print("matplotlib missing:", type(exc).__name__)
from importlib.metadata import version
print("lerobot", version("lerobot"))
from lerobot.configs.video import VideoEncoderConfig
print("opts none", VideoEncoderConfig().get_codec_options(None, as_strings=True))
print("opts lp3", VideoEncoderConfig().get_codec_options(3, as_strings=True))

c = av.open(sys.argv[1])
s = c.streams.video[0]
print("codec", s.codec_context.name, "time_base", s.time_base, "container start_time_us", c.start_time)
print("has thread_count", hasattr(s, "thread_count"), "has thread_type", hasattr(s, "thread_type"))
try:
    s.thread_count = 1
    print("stream.thread_count ok", s.thread_count)
except Exception as exc:
    print("stream.thread_count error", type(exc).__name__, str(exc)[:120])
    s.codec_context.thread_count = 1
    print("codec_context.thread_count ok", s.codec_context.thread_count)
n = 0
for pkt in c.demux(s):
    n += len(pkt.decode())
    if n >= 60:
        break
print("decoded", n, "frames; OS threads now", len(os.listdir("/proc/self/task")))
c.close()
