#!/usr/bin/env bash
# E1 FRAGILITY-AUDIT grep. Read-only: grep, awk, sha256sum, git --no-optional-locks log/status, ls, file, head, wc.
# Writes ONE result file: R/work/FRAGILITY-AUDIT/E1/grep_consts.txt
set -u
export LC_ALL=C
R=/home/intel/intel-junction-2026
T=$R/app/tools
W=$R/work/FRAGILITY-AUDIT/E1
OUT=$W/grep_consts.txt
TMPH=$W/.hits_tmp.txt
mkdir -p "$W"
: > "$OUT"
emit() { printf '%s\n' "$*" >> "$OUT"; }

run_class() { # $1 file  $2 tag  $3 regex  $4 include filter (ERE, -i, may be empty)  $5 exclude filter (ERE, -i, may be empty)
  local a
  a=$(grep -n -E -e "$3" -- "$1" 2>/dev/null)
  if [ -n "${4:-}" ]; then a=$(printf '%s\n' "$a" | grep -i -E -e "$4" 2>/dev/null); fi
  if [ -n "${5:-}" ]; then a=$(printf '%s\n' "$a" | grep -v -i -E -e "$5" 2>/dev/null); fi
  printf '%s\n' "$a" | grep -E '^[0-9]+:' | cut -d: -f1 | sed "s/\$/ $2/"
}

collect() { # $1 file -> "LINE TAG" pairs
  local f="$1"
  run_class "$f" TUP '(\(|\[)[[:space:]]*-?[0-9]+(\.[0-9]+)?[[:space:]]*,[[:space:]]*-?[0-9]+(\.[0-9]+)?([[:space:]]*,[[:space:]]*-?[0-9]+(\.[0-9]+)?){0,2}[[:space:]]*(\)|\])' 'px|pix|roi|spot|target|plate|cent|crop|point|xy|img|image|frame|cv2|\bmm\b|pos|origin|offset|\btip|jaw|hover|\bmat\b|MAT_|grasp|\brest\b|scan|pose|station|colou?r|bgr|rgb|hsv|circle|rect|\bbox|cell|goal|size|shape|depth|dist|yaw|angle|tape|brick' ''
  run_class "$f" LIST6 '(\(|\[)[[:space:]]*-?[0-9.]+([[:space:]]*,[[:space:]]*-?[0-9.]+){4,}[[:space:]]*(\)|\])' 'rest|scan|pose|goal|home|hover|station|park|pick|place|wrist|close|cam|view|grasp|sort|jaw|tip|plate|pos' ''
  run_class "$f" ID '\b(ROI|SPOT|TARGET|PLATES?|MAT_Z|HOVER|CROP|ORIGIN|AREA|THRESH|HSV|HUE|SAT|VAL|PX|PIX|FX|FY|CX|CY|JAW|TIP|WRIST|CLOSE|STATION|SCAN|REST|DEPTH)[A-Z0-9_]*\b|\b[a-z0-9_]*(roi|spot|target|plate|hover|crop|origin|thresh|hsv|hue|area|_px|px_|pix)[a-z0-9_]*\b|\b(fx|fy|cx|cy)\b' '' ''
  run_class "$f" HSV 'inRange|threshold\(|adaptiveThreshold|[HSV]_(MIN|MAX|LO|HI)\b|\b[hsv]_?(lo|hi|min|max)\b|\b[HSV] *(<|>|<=|>=)' '' ''
  run_class "$f" HSVARR 'np\.array\(\[[^]]*[0-9]' 'hsv|lower|upper|low|high|range|colo|blue|green|red|pink|white|mask|thr' ''
  run_class "$f" CMPIDX '\][[:space:]]*(<|>|<=|>=|==)[[:space:]]*-?[0-9]+(\.[0-9]+)?' 'st\[|px|pix|img|frame|mask|hsv|rgb|depth|roi|area|cent|jaw|tip|plate|brick|blob|contour|count|cam|wrist|close|top|sum' ''
  run_class "$f" AREA 'contourArea|countNonZero|count_nonzero|np\.sum\(|\.sum\(|\bblob|BLOB|MIN_PIX|MAX_PIX|min_px|max_px|min_area|max_area|MIN_AREA|MAX_AREA|minArea|maxArea|\barea\b|AREA' '' ''
  run_class "$f" CMPNUM '(>|<)=?[[:space:]]*[0-9]+(\.[0-9]+)?' 'px|pix|area|mask|blob|contour|cv2|img|frame|roi|crop|cent|sum|count|hsv|depth|\bmm\b|dist' '(print\(|log|logger|raise|Error)'
  run_class "$f" SLICE '\[[^][]*[0-9][^][]*:[^][]*\]|\[[^][]*:[^][]*[0-9][^][]*\]' 'img|frame|\bim\b|image|crop|mask|rgb|bgr|gray|hsv|frm|vis|depth|roi|patch|win|raw|jpg|png|\bcap|\bpx\b|pix|top|close|wrist|\bsub|\bsrc|\bdst|cam|hand' ''
  run_class "$f" XY0 '\b(x0|y0|x1|y1|x2|y2|cx|cy|ox|oy|px|py|u0|v0|xc|yc|x_c|y_c|x_min|y_min|x_max|y_max|xmin|ymin|xmax|ymax|x_lo|y_lo|x_hi|y_hi)\b[[:space:]]*(=|:)[[:space:]]*-?[0-9]+(\.[0-9]+)?|\b[a-zA-Z]{0,6}_?[xyXY][0-9]?\b[[:space:]]*(=|:)[[:space:]]*-?[0-9]{2,}(\.[0-9]+)?' '' ''
  run_class "$f" RES '\b(640|480|1280|720|1920|1080|424|848|960|540|320|240|1600|1200)\b' 'cam|frame|img|image|size|shape|resiz|width|height|stream|rs\.|config|enable|cv2|crop|roi|px|res|fps|video|color|depth' ''
  run_class "$f" MMASSIGN '\b[A-Za-z_]*(_Z|_z|Z_|z_|_X|_Y|_x|_y|_mm|MM|DZ|DX|DY|dz|dx|dy)[A-Za-z_0-9]*\b[[:space:]]*=[[:space:]]*-?[0-9]+(\.[0-9]+)?' '' ''
  run_class "$f" MMUNIT '-?[0-9]+(\.[0-9]+)?[[:space:]]*(mm|cm|px|pixels?|deg|degrees?)\b' '' ''
  run_class "$f" COLOUR '\b(blue|green|red|pink|white|black|grey|gray|purple|yellow|orange|BLUE|GREEN|RED|PINK|WHITE|BLACK|GREY|GRAY|dark|DARK|bright|BRIGHT)\b' '' '(print\(|log|logger|raise|Error)'
  run_class "$f" SIZE '2x2|2 ?x ?2|\bstud|\bSTUD|DUPLO|duplo|\bbrick_|BRICK_|\bw_px|\bh_px|aspect|ASPECT|\bratio\b|RATIO|orient|ORIENT|yaw|YAW|\bangle\b|ANGLE' '' ''
  run_class "$f" LIGHT 'brightness|BRIGHT|exposure|EXPOS|gamma|GAMMA|\blum|LUM|\bgain|GAIN|DIFF_T|diff_t|DELTA_T|delta_t|changed_frac|CHANGED|\bchange_|mean_?[a-z]*\(|np\.mean\(|np\.std\(|\.std\(' '' ''
  run_class "$f" THRCONST '\b[A-Za-z_]*(THR|TOL|THRESH|LIMIT|_LIM|MIN_|MAX_|_MIN|_MAX|_TH)[A-Za-z_0-9]*\b[[:space:]]*=[[:space:]]*-?[0-9]+(\.[0-9]+)?' 'px|pix|area|hsv|h_|s_|v_|jaw|tip|plate|brick|blob|contour|cent|cam|hand|tape|depth|dist|mm|xy|tol|thr|close|wrist|station|mat|roi|spot|grasp|pos|scan|sat|val|hue' ''
  run_class "$f" INTR 'depth_scale|DEPTH_SCALE|intrinsics|INTRINSICS|principal|focal|ppx|ppy|camera_matrix|cameraMatrix|distCoeffs|dist_coeffs|get_intrinsics|\bK\b[[:space:]]*=' '' ''
  run_class "$f" CALIB 'homography|findHomography|estimateAffine|affine|solvePnP|lstsq|polyfit|np\.linalg|LinearRegression|RANSAC|ransac|residual|resid' '' ''
}

FILES=""
MISSING=""
EXTRA=""
for f in sort_any.py sort_loop.py grasp_auto.py pick_any.py kin.py handdet.py hand_watch.py jaw_check.py jaw_calib.py closeloc.py scan_calib.py rs_height.py; do
  if [ -f "$T/$f" ]; then FILES="$FILES $T/$f"; else MISSING="$MISSING $f"; fi
done
if [ -f "$R/app/anomaly/det2_f2/inspector.py" ]; then FILES="$FILES $R/app/anomaly/det2_f2/inspector.py"; else MISSING="$MISSING inspector.py"; fi
for f in "$T"/*.py; do
  [ -f "$f" ] || continue
  b=$(basename "$f")
  case "$b" in
    *scan*|*map*|*calib*|*station*|*closeloc*|*plate*|*cell*|*view*|*servo*|*jaw*)
      case " $FILES " in
        *" $f "*) ;;
        *) FILES="$FILES $f"; EXTRA="$EXTRA $b";;
      esac;;
  esac
done

emit "# E1 grep_consts  utc=$(date -u +%Y-%m-%dT%H:%MZ)  root=$R"
emit "# context = 2 lines before/after each hit, merged; '>' marks a hit line; 'in' = enclosing def/class"
emit "# CLASSES: TUP=(x,y[,z[,w]]) 2-4 numbers with a geometry word; LIST6=5+ numbers with pose/goal word; ID=ROI/SPOT/TARGET/PLATE/MAT_Z/HOVER/CROP/AREA/HSV/_px/fx/cx names"
emit "#   HSV=inRange/threshold/H,S,V comparisons; HSVARR=np.array of numbers near colour words; CMPIDX=x[...] compared with a number; AREA=area/blob/pixel-count code"
emit "#   CMPNUM=comparison with a number in px/area/mask/mm context; SLICE=numpy slice with numbers in image context; XY0=x0/y0/cx/cy style assignments; RES=camera resolution literal"
emit "#   MMASSIGN=*_Z/*_X/*_Y/*_mm constant; MMUNIT=number with mm/px/deg; COLOUR=colour names; SIZE=stud/2x2/yaw/angle/aspect; LIGHT=brightness/exposure/gain/diff thresholds"
emit "#   THRCONST=*_THR/_TOL/_MIN/_MAX constant; INTR=intrinsics/depth scale; CALIB=homography/affine/PnP/least-squares/residual code"
emit "# git R/app HEAD: $(git -C "$R/app" --no-optional-locks log -1 --format='%h %cI %s' 2>/dev/null)"
emit "# git R/app status (uncommitted = working tree differs from HEAD):"
git -C "$R/app" --no-optional-locks status --short 2>/dev/null | head -40 | sed 's/^/#   /' >> "$OUT"
emit "# PY FILES: $(for f in $FILES; do basename "$f"; done | tr '\n' ' ')"
emit "# EXTRA (name match scan|map|calib|station|closeloc|plate|cell|view|servo|jaw): ${EXTRA:-none}"
emit "# MISSING named files: ${MISSING:-none}"

for f in $FILES; do
  b=$(basename "$f")
  collect "$f" > "$TMPH"
  nhits=$(awk '{print $1}' "$TMPH" | sort -u | wc -l)
  tagsum=$(awk '{c[$2]++} END {for (k in c) printf "%s=%d ", k, c[k]}' "$TMPH")
  emit ""
  emit "===== FILE $b  lines=$(wc -l < "$f")  sha256=$(sha256sum "$f" | cut -c1-16)  hit_lines=$nhits ====="
  emit "# tags: $tagsum"
  emit "## module-level constant assignments (full inventory, no context):"
  grep -n -E '^[A-Z][A-Z0-9_]{2,}[[:space:]]*(:[^=]*)?=[[:space:]]*[-(\[{0-9]' "$f" 2>/dev/null | cut -c1-150 | sed 's/^/  /' >> "$OUT"
  emit "## context blocks:"
  awk -v FN="$b" -v HF="$TMPH" '
function trim(x) { gsub(/^[ \t]+|[ \t]+$/, "", x); return x }
BEGIN {
  while ((getline ln < HF) > 0) {
    n = split(ln, a, " ")
    if (n >= 2) { k = a[1] + 0; if (index(" " t[k] " ", " " a[2] " ") == 0) t[k] = t[k] " " a[2] }
  }
  close(HF)
}
{
  L[NR] = $0
  if ($0 ~ /^[ \t]*(def|class) /) { c = $0; sub(/^[ \t]+/, "", c); sub(/\(.*$/, "", c); sub(/:[ \t]*$/, "", c); cur = c }
  F[NR] = cur
}
END {
  n = 0
  for (i = 1; i <= NR; i++) if (i in t) H[++n] = i
  k = 1
  while (k <= n) {
    s = H[k] - 2; if (s < 1) s = 1
    e = H[k] + 2; if (e > NR) e = NR
    m = k
    while (m < n && H[m + 1] - 2 <= e + 1) { m++; e = H[m] + 2; if (e > NR) e = NR }
    hdr = ""
    for (q = k; q <= m; q++) hdr = hdr sprintf(" %d:%s", H[q], trim(t[H[q]]))
    printf "## %s:%d-%d | in %s |%s\n", FN, s, e, (F[H[k]] == "" ? "module" : F[H[k]]), hdr
    for (j = s; j <= e; j++) printf "%s%5d| %s\n", ((j in t) ? ">" : " "), j, substr(L[j], 1, 130)
    k = m + 1
  }
}' "$f" >> "$OUT"
  echo "$b hit_lines=$nhits blocks=$(grep -c "^## $b:" "$OUT") tags: $tagsum"
done

DATA_HDR="===== DATA FILES: motion poses (R/app/motion/gm) and tool json/yaml/png (first 60 lines each, cut at 150 chars) ====="
emit ""
emit "$DATA_HDR"
for d in "$R/app/motion/gm" "$T"; do
  if [ ! -d "$d" ]; then emit "MISSING DIR $d"; continue; fi
  for x in "$d"/*.yaml "$d"/*.yml "$d"/*.json; do
    [ -f "$x" ] || continue
    emit "--- $x  lines=$(wc -l < "$x")  bytes=$(wc -c < "$x")"
    head -n 60 "$x" | cut -c1-150 >> "$OUT"
  done
done
emit "--- PNG / npz / csv in tools:"
for x in "$T"/*.png "$T"/*.npz "$T"/*.csv; do
  [ -f "$x" ] || continue
  emit "  $x  bytes=$(wc -c < "$x")  $(file -b "$x" 2>/dev/null | cut -c1-100)"
done

emit ""
emit "===== END ====="
rm -f "$TMPH"
echo "GREP_CONSTS_DONE utc=$(date -u +%Y-%m-%dT%H:%MZ) out=$OUT lines=$(wc -l < "$OUT") bytes=$(wc -c < "$OUT")"
