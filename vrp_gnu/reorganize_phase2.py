import os
import shutil

base_dir = r"\\wsl.localhost\Ubuntu-24.04\root\vrp"
scripts_dir = os.path.join(base_dir, 'scripts')
src_exp_dir = os.path.join(base_dir, 'src', 'experiments')

# Target directories
analysis_dir = os.path.join(scripts_dir, 'analysis')
experiments_dir = os.path.join(scripts_dir, 'experiments')
utils_dir = os.path.join(scripts_dir, 'utils')
pipeline_dir = os.path.join(scripts_dir, 'pipeline')

# Ensure target directories exist
for d in [analysis_dir, experiments_dir, utils_dir, pipeline_dir]:
    os.makedirs(d, exist_ok=True)

# 1. Reorganize scripts
def move_file(filename, target_dir):
    src = os.path.join(scripts_dir, filename)
    if os.path.exists(src):
        dst = os.path.join(target_dir, filename)
        try:
            shutil.move(src, dst)
            print(f"Moved {filename} to {os.path.basename(target_dir)}")
        except Exception as e:
            print(f"Failed to move {filename}: {e}")

# Analysis scripts
move_file('gen_median_table.py', analysis_dir)
move_file('compute_missing_metrics.py', analysis_dir)
move_file('run_enet_feature_selection.py', analysis_dir)

# Experiment scripts
move_file('tmp_benchmark.py', experiments_dir)
move_file('tmp_calc_wens_multi_horizon.py', experiments_dir)
move_file('tmp_multi_horizon_nonoverlap.py', experiments_dir)
move_file('tmp_wens_weight_experiment.py', experiments_dir)
move_file('run_garch_benchmark.py', experiments_dir)
move_file('run_trading_strategy.py', experiments_dir)

# Utility scripts
move_file('json_to_csv.py', utils_dir)
move_file('patch_qmds.py', utils_dir)
move_file('cleanup.py', utils_dir)
move_file('run_server.sh', utils_dir)

# 2. Cleanup empty directories
if os.path.exists(src_exp_dir):
    try:
        os.rmdir(src_exp_dir)
        print(f"Removed directory: {src_exp_dir}")
    except OSError as e:
        print(f"Warning: Could not remove {src_exp_dir} (may not be empty) - {e}")
