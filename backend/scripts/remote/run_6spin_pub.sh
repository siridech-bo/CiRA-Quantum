#!/bin/bash
# 6-spin publication runs (QRC_Next_Stage_Status 2026-10-24): 6spin_pub then 6spin_retrace,
# shard K of 2 per host. One GPU job at a time. Usage: run_6spin_pub.sh K
K=$1
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
R=$HOME/cira-qrc/runs/6spin-pub
mkdir -p $R
for plan in 6spin_pub 6spin_retrace; do
  D=$R/${plan}_shard$K
  [ -f $D/campaign.json ] && continue
  echo "$plan shard $K start $(date -u)" >> $R/PROGRESS_$K
  $PY scripts/qrc_learnable_campaign.py --plan $plan --shard $K/2 --run-dir $D --out $D/campaign.json \
      --budget-hours 40 > $D.log 2>&1
  echo "$plan shard $K exit $? $(date -u)" >> $R/PROGRESS_$K
done
echo "all done $(date -u)" >> $R/PROGRESS_$K
