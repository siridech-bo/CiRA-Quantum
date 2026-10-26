#!/bin/bash
# PSR-driven training loop vs autograd (methods demo, qrc_psr_train.py). Usage: run_psr_train.sh SYSTEM STEPS
SYS=$1; STEPS=$2
cd $HOME/cira-qrc/backend
PY=$HOME/.venvs/qrc/bin/python
R=$HOME/cira-qrc/runs/psr_train; mkdir -p $R
$PY scripts/qrc_psr_train.py --system $SYS --steps $STEPS --device cuda --splits 30 900 300 288 \
  --out $R/qrc_psr_train > $R/train_${SYS}spin.log 2>&1
echo "${SYS}spin exit $? $(date -u)" >> $R/PROGRESS
