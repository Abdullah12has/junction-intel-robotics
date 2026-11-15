"""Replace narration using six locally synthesized WAV files, preserving video timing.

Usage: python replace_narration.py AUDIO_DIRECTORY
Files are named abdullah-0.wav through abdullah-5.wav. Voice profiles stay private.
Requires the segment videos from make_video_v2.py and FFmpeg.
"""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
build = root / 'submission/build/revision'
audio_dir = Path(sys.argv[1])
manifest = json.loads((root / 'submission/video-v2-manifest.json').read_text())
def duration(path):
    return float(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', str(path)]))

parts = []
for i, record in enumerate(manifest['segments']):
    video = build / f'v2-segment-{i}.mp4'
    audio = audio_dir / f'abdullah-{i}.wav'
    seconds = duration(video)
    tempo = max(1.0, duration(audio) / (seconds - .2))
    if tempo > 1.5:
        raise ValueError(f'Segment {i} needs shorter narration: tempo {tempo:.2f}')
    dest = build / f'pocket-segment-{i}.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(video), '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-af', f'atempo={tempo},loudnorm=I=-16:TP=-1.5:LRA=11,apad', '-t', str(seconds), '-c:a', 'aac', '-b:a', '192k', str(dest)], check=True)
    parts.append(dest)
    record['voice_tempo'] = tempo
    print(i, 'seconds', seconds, 'tempo', round(tempo, 3), flush=True)
concat = build / 'pocket-concat.txt'
concat.write_text(''.join(f"file '{p}'\n" for p in parts))
out = root / 'submission/output/prometheus-demo-narrated.mp4'
subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(concat), '-c', 'copy', '-movflags', '+faststart', str(out)], check=True)
manifest['voice'] = 'Pocket TTS, saved Abdullah voice; pitch-preserving timing adjustment and loudness normalization'
manifest['duration_seconds'] = duration(out)
(root / 'submission/video-pocket-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Final seconds', duration(out))
