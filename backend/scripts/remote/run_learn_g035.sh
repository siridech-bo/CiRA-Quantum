#!/bin/bash
# Step 4 dev (QRC_Design_Concepts.md §9): learned encoding on top of g=0.35, seeds 0-2,
# then validation-only scoring. One GPU job at a time. Usage: run_learn_g035.sh TAG ds...
TAG=$1; shift
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
for ds in "$@"; do
  R=$HOME/cira-qrc/runs/realworld/$ds
  for s in 0 1 2; do
    [ -f $R/design_learn/g0.35_seed$s/X_learned.npz ] && continue
    $PY scripts/qrc_realworld_bench.py learn --dataset $ds --run-dir $R --gain 0.35 --seed $s > $R/learn_g035_seed$s.log 2>&1
  done
  $PY scripts/qrc_design_learn.py dev --dataset $ds --run-root $R > $R/learn_dev.log 2>&1
  echo "$ds done $(date -u)" >> $HOME/cira-qrc/runs/realworld/LEARN_PROGRESS_$TAG
done
echo "all done $(date -u)" >> $HOME/cira-qrc/runs/realworld/LEARN_PROGRESS_$TAG
