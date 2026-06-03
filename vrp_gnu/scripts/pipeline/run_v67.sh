#!/bin/bash
cd "$(dirname "$0")"
python3 src/experiments/verification/v67_v36_robustness_hparam_cv.py 2>&1
