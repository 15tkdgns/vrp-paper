"""Quick 22d-only test of v5 logic to verify WEns >= 0.803."""
import sys
# Patch HORIZONS before running
import importlib.util, types

# Load v5 source and patch HORIZONS
spec = importlib.util.spec_from_file_location("v5", "/root/vrp/run_main_benchmark_v5.py")
src = open("/root/vrp/run_main_benchmark_v5.py").read()
# Replace HORIZONS list with only 22d, and skip saves
src = src.replace(
    "HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]",
    "HORIZONS = [22]"
)
# Redirect output files to /tmp to avoid overwriting
src = src.replace(
    "out_json = '/root/vrp/results/main_benchmark_v5_results.json'",
    "out_json = '/tmp/test_v5_22d.json'"
)
src = src.replace(
    "out_csv = '/root/vrp/paper/csv/main_benchmark_v5_performance.csv'",
    "out_csv = '/tmp/test_v5_22d.csv'"
)
exec(compile(src, "run_main_benchmark_v5.py", "exec"))
