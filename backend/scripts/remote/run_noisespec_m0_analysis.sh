#!/bin/bash
# Milestone 0 analysis (pre-registered 09e3bfd). Usage: run_noisespec_m0_analysis.sh a|b
cd $HOME/cira-qrc/backend
R=$HOME/cira-qrc/runs/noisespec_m0; mkdir -p $R
OMP_NUM_THREADS=8 $HOME/.venvs/qrc/bin/python scripts/qrc_noisespec_m0_analysis.py --task $1 --jobs 32 > $R/analysis_$1.log 2>&1
echo "analysis $1 exit $? $(date -u)" >> $R/PROGRESS_ANALYSIS
