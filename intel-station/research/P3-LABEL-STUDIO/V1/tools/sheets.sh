#!/bin/bash
# usage: sheets.sh <windows.tsv>; writes frames/<WID>_sheet.png (row1 cam1 243722064316, row2 cam2 243622061421; 4 time points each)
set -u
TSV="$1"
FR=/home/intel/intel-junction-2026/work/P3-LABEL-STUDIO/V1/frames
TMP=$(mktemp -d /tmp/v1sheets.XXXXXX)
tail -n +2 "$TSV" | while IFS=$'\t' read -r wid c1f c1t c2f c2t c1path entries; do
  c1p="/home/intel/intel-junction-2026/data/$c1path"
  c2p="${c1p/243722064316/243622061421}"
  # cam1 path = the 243722 file; cam2 path must be the 243622 file of the same episode window
  c2p=$(echo "$c1p" | sed 's#observation.images.realsense 243722064316#observation.images.realsense 243622061421#')
  ok=1
  for fr in 0.10 0.35 0.65 0.90; do
    t1=$(python3 -I -c "print(round($c1f+($c1t-$c1f)*$fr,3))")
    t2=$(python3 -I -c "print(round($c2f+($c2t-$c2f)*$fr,3))")
    ffmpeg -v error -y -ss "$t1" -i "$c1p" -frames:v 1 -vf "scale=480:360" "$TMP/${wid}_c1_${fr}.png" || ok=0
    ffmpeg -v error -y -ss "$t2" -i "$c2p" -frames:v 1 -vf "scale=480:360" "$TMP/${wid}_c2_${fr}.png" || ok=0
  done
  if [ $ok -eq 1 ]; then
    ffmpeg -v error -y \
      -i "$TMP/${wid}_c1_0.10.png" -i "$TMP/${wid}_c1_0.35.png" -i "$TMP/${wid}_c1_0.65.png" -i "$TMP/${wid}_c1_0.90.png" \
      -i "$TMP/${wid}_c2_0.10.png" -i "$TMP/${wid}_c2_0.35.png" -i "$TMP/${wid}_c2_0.65.png" -i "$TMP/${wid}_c2_0.90.png" \
      -filter_complex "[0][1][2][3]hstack=inputs=4[t];[4][5][6][7]hstack=inputs=4[b];[t][b]vstack=inputs=2" \
      -frames:v 1 "$FR/${wid}_sheet.png" && echo "$wid sheet ok"
  else
    echo "$wid FAILED extraction"
  fi
done
rm -rf "$TMP"
