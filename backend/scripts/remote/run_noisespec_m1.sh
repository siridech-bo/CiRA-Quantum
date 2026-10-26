#!/bin/bash
# Milestone 1: mechanism gate, then raw FID traces (pre-registered; see QRC_NoiseSpec_Milestone1.md).
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
R=$HOME/cira-qrc/runs/noisespec_m1; mkdir -p $R
$PY scripts/qrc_noisespec_m1.py mech --device cuda > $R/mech.log 2>&1
echo "mech exit $? $(date -u)" >> $R/PROGRESS
for t in a b; do for s in dev confirm; do
  $PY scripts/qrc_noisespec_m1.py gen --task $t --split $s --device cuda > $R/gen_${t}_${s}.log 2>&1
  echo "$t $s exit $? $(date -u)" >> $R/PROGRESS
done; done
echo "all done $(date -u)" >> $R/PROGRESS
