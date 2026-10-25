#!/bin/bash
# Confirmation 3 (§11, prereg a7cb3e6) or drift test (§12, prereg a4111de), per dataset:
# ESN grid (CPU, background) | GPU: QRC pass g = 0.35 (FID trace saved) -> LSTM | offline eval.
# Usage: run_step5.sh confirm3|drift TAG dataset...
MODE=$1; TAG=$2; shift 2
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
for ds in "$@"; do
  R=$HOME/cira-qrc/runs/step5/$ds; mkdir -p $R
  ( OMP_NUM_THREADS=1 $PY scripts/qrc_realworld_bench.py esn --dataset $ds --run-dir $R --workers 32 > $R/esn.log 2>&1 ) &
  EP=$!
  $PY scripts/qrc_design_c1.py --dataset $ds --run-root $R --gains 0.35 > $R/c1.log 2>&1
  $PY scripts/qrc_realworld_bench.py lstm --dataset $ds --run-dir $R > $R/lstm.log 2>&1
  wait $EP
  OMP_NUM_THREADS=8 $PY scripts/qrc_design_r.py $MODE --dataset $ds --run-root $R > $R/$MODE.log 2>&1
  echo "$ds exit $? $(date -u)" >> $HOME/cira-qrc/runs/step5/PROGRESS_$TAG
done
echo "all done $(date -u)" >> $HOME/cira-qrc/runs/step5/PROGRESS_$TAG
