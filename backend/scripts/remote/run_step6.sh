#!/bin/bash
# Step 6 (§13, prereg d022533): per dev dataset, GPU pass (raw traces saved) -> VALIDATION-only scoring.
# Usage: run_step6.sh A|B TAG dataset...
PART=$1; TAG=$2; shift 2
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
for ds in "$@"; do
  R=$HOME/cira-qrc/runs/step6/$ds; mkdir -p $R
  $PY scripts/qrc_design_p.py pass --part $PART --dataset $ds --run-root $R > $R/pass_$PART.log 2>&1
  e1=$?
  OMP_NUM_THREADS=8 $PY scripts/qrc_design_p.py dev --part $PART --dataset $ds --run-root $R > $R/dev_$PART.log 2>&1
  echo "$ds pass $e1 dev $? $(date -u)" >> $HOME/cira-qrc/runs/step6/PROGRESS_$TAG
done
echo "all done $(date -u)" >> $HOME/cira-qrc/runs/step6/PROGRESS_$TAG
