import json
import re
import glob

with open('/root/vrp/src/experiments/creative/multi_horizon_benchmark_results.json', 'r') as f:
    d = json.load(f)
    print('Available in JSON:', list(d['results'].keys()))

pattern = r" horizons\s*=\s*\[1d 5d 22d 60d 90d 120d 180d 252d ]repl
