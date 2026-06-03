#!/bin/bash
export PYTHONPATH="/home/my/.local/lib/python3.12/site-packages"
PYTHON="/usr/bin/python3"
BASE="/mnt/c/Users/user/Desktop/vrp/vrp_gnu"
LOG="$BASE/logs"
mkdir -p "$LOG"

ts() { date '+%H:%M:%S'; }

echo "[$(ts)] START suite" | tee "$LOG/suite.log"

for sc in run_main_benchmark_v6.py run_dlinear_benchmark.py run_patchtst_benchmark.py; do
    echo "[$(ts)] === $sc ===" | tee -a "$LOG/suite.log"
    "$PYTHON" -u "$BASE/$sc" 2>&1 | tee "$LOG/${sc%.py}.log"
    echo "[$(ts)] DONE $sc exit=$?" | tee -a "$LOG/suite.log"
done

echo "[$(ts)] ALL DONE" | tee -a "$LOG/suite.log"