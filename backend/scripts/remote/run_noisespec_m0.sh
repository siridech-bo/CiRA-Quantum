#!/bin/bash
# Milestone 0 noise spectroscopy: generate raw FID traces (pre-registered 09e3bfd).
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
R=$HOME/cira-qrc/runs/noisespec_m0; mkdir -p $R
for t in a b; do for s in dev confirm; do
  $PY scripts/qrc_noisespec_m0.py gen --task $t --split $s --device cuda --batch 300 > $R/gen_${t}_${s}.log 2>&1
  echo "$t $s exit $? $(date -u)" >> $R/PROGRESS
done; done
echo "all done $(date -u)" >> $R/PROGRESS
