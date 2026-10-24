#!/bin/bash
# Confirmation 2 per city (QRC_Design_Concepts.md §10, pre-registration 7965cde): ESN grid (CPU,
# background) | GPU: QRC passes g = 0.35 (final design) and g = 1 -> LSTM | confirm eval.
TAG=$1; shift
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
for c in "$@"; do
  ds=era5_$c; R=$HOME/cira-qrc/runs/confirm2/$ds; mkdir -p $R
  ( OMP_NUM_THREADS=1 $PY scripts/qrc_realworld_bench.py esn --dataset $ds --run-dir $R --workers 32 > $R/esn.log 2>&1 ) &
  EP=$!
  $PY scripts/qrc_design_c1.py --dataset $ds --run-root $R --gains 0.35,1.0 > $R/c1.log 2>&1
  $PY scripts/qrc_realworld_bench.py lstm --dataset $ds --run-dir $R > $R/lstm.log 2>&1
  wait $EP
  $PY scripts/qrc_design_confirm.py --dataset $ds --run-dir $R --grid 0.35,1 --prereg 7965cde > $R/confirm.log 2>&1
  echo "$ds done $(date -u)" >> $HOME/cira-qrc/runs/confirm2/PROGRESS_$TAG
done
echo "all done $(date -u)" >> $HOME/cira-qrc/runs/confirm2/PROGRESS_$TAG
