import os
import shutil
import glob

base_dir = r"\\wsl.localhost\Ubuntu-24.04\root\vrp"

# 1. Move paper to docs/paper
paper_dir = os.path.join(base_dir, 'paper')
docs_paper_dir = os.path.join(base_dir, 'docs', 'paper')
os.makedirs(docs_paper_dir, exist_ok=True)
if os.path.exists(paper_dir):
    for item in os.listdir(paper_dir):
        src = os.path.join(paper_dir, item)
        dst = os.path.join(docs_paper_dir, item)
        try:
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Failed to move {src}: {e}")

# 2. Move src/experiments to experiments
src_exp_dir = os.path.join(base_dir, 'src', 'experiments')
exp_dir = os.path.join(base_dir, 'experiments')
os.makedirs(exp_dir, exist_ok=True)
if os.path.exists(src_exp_dir):
    for item in os.listdir(src_exp_dir):
        src = os.path.join(src_exp_dir, item)
        dst = os.path.join(exp_dir, item)
        try:
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Failed to move {src}: {e}")

# 3. Move scripts/run_v*.sh to scripts/pipeline
scripts_dir = os.path.join(base_dir, 'scripts')
pipeline_dir = os.path.join(scripts_dir, 'pipeline')
os.makedirs(pipeline_dir, exist_ok=True)
for sh_file in glob.glob(os.path.join(scripts_dir, 'run_v*.sh')):
    dst = os.path.join(pipeline_dir, os.path.basename(sh_file))
    try:
        shutil.move(sh_file, dst)
        print(f"Moved {sh_file} to {dst}")
    except Exception as e:
        print(f"Failed to move {sh_file}: {e}")

# 4. Move unused models to archive/models
models_dir = os.path.join(base_dir, 'src', 'models')
archive_models_dir = os.path.join(base_dir, 'archive', 'models')
os.makedirs(archive_models_dir, exist_ok=True)
if os.path.exists(models_dir):
    for item in os.listdir(models_dir):
        src = os.path.join(models_dir, item)
        if os.path.isfile(src) and not item.startswith('v25_champion') and not item.startswith('v29_champion') and not item.startswith('v50_champion') and item != '__init__.py':
            dst = os.path.join(archive_models_dir, item)
            try:
                shutil.move(src, dst)
                print(f"Moved {src} to {dst}")
            except Exception as e:
                print(f"Failed to move {src}: {e}")
