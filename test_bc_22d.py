"""Quick 22d-only test of baseline comparison script."""
src = open("/root/vrp/run_baseline_comparison.py").read()
src = src.replace("HORIZONS          = [1, 5, 22, 60, 90, 120, 180, 252]", "HORIZONS          = [22]")
src = src.replace("out_json = '/root/vrp/results/baseline_comparison_results.json'", "out_json = '/tmp/bc_test.json'")
src = src.replace("out_perf = '/root/vrp/paper/csv/baseline_comparison_performance.csv'", "out_perf = '/tmp/bc_test_perf.csv'")
src = src.replace("out_dm = '/root/vrp/paper/csv/dm_test_results.csv'", "out_dm = '/tmp/bc_test_dm.csv'")
exec(compile(src, "run_baseline_comparison.py", "exec"))
