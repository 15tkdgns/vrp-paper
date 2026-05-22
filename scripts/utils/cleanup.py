import os
import glob

def cleanup():
    files_to_remove = [
        'json_to_csv.py',
        'v60_ablation.log',
        'v60_ablation_retry.log',
        'v61_feature_importance.log'
    ]
    
    print("Cleaning up temporary files...")
    for f in files_to_remove:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed {f}")
            except Exception as e:
                print(f"Failed to remove {f}: {e}")
        else:
            print(f"Not found {f}")

if __name__ == "__main__":
    cleanup()
