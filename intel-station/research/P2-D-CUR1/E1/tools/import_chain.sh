#!/bin/bash
# Import PP-GREEN and PP-BOTH in turn (prepare, upload, review, finalize) and log each step in the laptop folder.
cd /home/intel/intel-junction-2026/work/P2-D-CUR1/E1 || exit 1
PY=/home/intel/intel-junction-2026/envs/data-env/bin/python3
Z=/home/intel/intel-junction-2026/data/curated
bash tools/import_ds.sh "$Z/PP-GREEN.zip" "PP-GREEN v1 red brick to green plate" "move the red brick to the green plate" > import_PP-GREEN.log 2>&1
J=$(grep -o 'IMPORT_JOB=[^ ]*' import_PP-GREEN.log | cut -d= -f2)
$PY -I tools/import_wait.py "$J" "move the red brick to the green plate" > import_wait_PP-GREEN.log 2>&1
bash tools/import_ds.sh "$Z/PP-BOTH.zip" "PP-BOTH v1 red brick to either plate" "move the red brick to the blue plate" > import_PP-BOTH.log 2>&1
J=$(grep -o 'IMPORT_JOB=[^ ]*' import_PP-BOTH.log | cut -d= -f2)
$PY -I tools/import_wait.py "$J" "move the red brick to the blue plate" > import_wait_PP-BOTH.log 2>&1
echo CHAIN_DONE >> import_wait_PP-BOTH.log
