#!/bin/bash
cd "$(dirname "$0")"
python3 src/experiments/verification/v66_v50_robustness_hparam_cv.py 2>&1
