#!/bin/bash
# Gripper-camera recorder for the Junction recording cycle: MJPEG 640x480@30 copied into 60 s MKV segments with
# absolute wall-clock timestamps, UTC file names. Restarts ffmpeg if it exits. Stops at END or when the stop file exists.
# Stop with: touch /home/intel/junction-cell/recording/STOP
R=/home/intel/junction-cell/recording
OUT=$R/wrist
mkdir -p "$OUT"
END=$(date -d '+4 hours' +%s)
DEV=$(for d in /sys/class/video4linux/video*; do case "$(cat $d/name)" in *USB2.0_CAM1*) [ "$(cat $d/index)" = 0 ] && echo /dev/$(basename $d);; esac; done | head -1)
echo "$(date -u +%FT%TZ) device $DEV" >> "$R/wrist_rec.log"
while [ "$(date +%s)" -lt "$END" ] && [ ! -e "$R/STOP" ]; do
  TZ=UTC ffmpeg -hide_banner -loglevel warning -f v4l2 -ts abs -input_format mjpeg -video_size 640x480 -framerate 30 \
    -i "$DEV" -map 0 -c copy -copyts -f segment -segment_time 60 -strftime 1 "$OUT/w_%Y%m%dT%H%M%SZ.mkv" \
    -map 0 -vf fps=2 -update 1 -y "$R/live_wrist.jpg" >> "$R/wrist_rec.log" 2>&1 &
  FF=$!
  while kill -0 $FF 2>/dev/null; do
    if [ -e "$R/STOP" ] || [ "$(date +%s)" -ge "$END" ]; then kill $FF; break; fi
    sleep 2
  done
  wait $FF 2>/dev/null
  echo "$(date -u +%FT%TZ) ffmpeg ended" >> "$R/wrist_rec.log"
  sleep 3
done
