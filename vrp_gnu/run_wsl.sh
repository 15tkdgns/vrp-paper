#!/bin/bash
# WSL launcher for VRP benchmarks
# Usage: bash run_wsl.sh [main|dlinear|patchtst|all]

SCRIPT_DIR="/mnt/c/Users/user/Desktop/vrp/vrp_gnu"
PYTHON="/usr/bin/python3"
PY_PATH="/home/$USER/.local/lib/python3.12/site-packages"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

export PYTHONPATH="$PY_PATH"
export PYTHONUSERBASE="/home/$USER/.local"

cd "$SCRIPT_DIR"

TARGET="${1:-all}"

run_script() {
    local script="$1"
    local log="$2"
    echo "============================================"
    echo "Running: $script"
    echo "Log: $log"
    echo "Started: $(date)"
    echo "============================================"
    "$PYTHON" "$SCRIPT_DIR/$script" 2>&1 | tee "$LOG_DIR/$log"
    echo "Finished: $(date)"
}

case "$TARGET" in
    main)
        run_script "run_main_benchmark_v6.py" "main_v6.log"
        ;;
    dlinear)
        run_script "run_dlinear_benchmark.py" "dlinear.log"
        ;;
    patchtst)
        run_script "run_patchtst_benchmark.py" "patchtst.log"
        ;;
    all)
        run_script "run_main_benchmark_v6.py"   "main_v6.log"
        run_script "run_dlinear_benchmark.py"   "dlinear.log"
        run_script "run_patchtst_benchmark.py"  "patchtst.log"
        ;;
    *)
        echo "Usage: bash run_wsl.sh [main|dlinear|patchtst|all]"
        exit 1
        ;;
esac

echo ""
echo "All done. Results in: $SCRIPT_DIR/results/"
