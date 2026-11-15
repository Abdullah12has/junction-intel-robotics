#!/bin/bash
# Opus real-time monitor (I-58, v2 I-59 17:05Z): both RealSense colour streams recorded to MKV at the native
# 30 fps with ABSOLUTE wall-clock timestamps (-ts abs + -copyts: packet pts = Unix time), so robot episodes can be
# cut exactly from the joint logs for the VLA datasets; mirrored as JPEGs (4 fps) for the watchdog and the loop.
# Stop: kill $(cat /home/intel/intel-junction-2026/run/monitor_cams.pid)
R=/home/intel/intel-junction-2026
mkdir -p "$R/logs/monitor" "$R/run"
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo $$ > "$R/run/monitor_cams.pid"
trap 'kill $P1 $P2 2>/dev/null; rm -f "$R/run/monitor_cams.pid"; exit 0' TERM INT
ffmpeg -hide_banner -loglevel error -f v4l2 -ts abs -video_size 640x480 -framerate 30 -i /dev/video8 \
  -map 0 -copyts -c:v libx264 -preset veryfast -crf 23 -g 30 "$R/logs/monitor/top_$TS.mkv" \
  -map 0 -vf fps=4 -update 1 -y "$R/run/live_top.jpg" >> "$R/logs/monitor/ffmpeg_top.log" 2>&1 &
P1=$!
ffmpeg -hide_banner -loglevel error -f v4l2 -ts abs -video_size 640x480 -framerate 30 -i /dev/video14 \
  -map 0 -copyts -c:v libx264 -preset veryfast -crf 23 -g 30 "$R/logs/monitor/close_$TS.mkv" \
  -map 0 -vf fps=4 -update 1 -y "$R/run/live_close.jpg" >> "$R/logs/monitor/ffmpeg_close.log" 2>&1 &
P2=$!
wait
