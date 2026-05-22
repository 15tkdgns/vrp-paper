"""Quick 22d-only test of v6 logic to verify WEns >= 0.803."""
src = open("/root/vrp/run_main_benchmark_v6.py").read()
src = src.replace("HORIZONS = [1, 5, 22, 60, 90, 120, 180, 252]", "HORIZONS = [22]")
src = src.replace("out_json = '/root/vrp/results/main_benchmark_v6_results.json'", "out_json = '/tmp/test_v6_22d.json'")
src = src.replace("out_csv = '/root/vrp/paper/csv/main_benchmark_v6_performance.csv'", "out_csv = '/tmp/test_v6_22d.csv'")
exec(compile(src, "run_main_benchmark_v6.py", "exec"))
