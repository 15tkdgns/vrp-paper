#!/bin/bash
export PYTHONPATH="/home/my/.local/lib/python3.12/site-packages"
PYTHON="/usr/bin/python3"
BASE="/mnt/c/Users/user/Desktop/vrp/vrp_gnu"
LOG="$BASE/logs"
mkdir -p "$LOG"

SCRIPT="${1:-run_main_benchmark_v6.py}"
LOGFILE="$LOG/$(basename $SCRIPT .py).log"

echo "Starting: $SCRIPT"
echo "Log: $LOGFILE"
nohup "$PYTHON" -u "$BASE/$SCRIPT" > "$LOGFILE" 2>&1 &
echo "PID: $!"
echo "$!" > "$LOG/$(basename $SCRIPT .py).pid"
