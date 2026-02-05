import os
from pathlib import Path

class Paths:
    """Project path constants"""
    ROOT = Path(__file__).parent.parent.parent
    DATA = ROOT / "data"
    RAW_DATA = DATA / "raw"
    PROCESSED_DATA = DATA / "processed"
    MODELS = DATA / "models"
    RESULTS = DATA / "results"

    @staticmethod
    def get_experiment_dir(exp_id):
        return Paths.RESULTS / str(exp_id)

def ensure_dirs():
    """Ensure key directories exist"""
    for path in [Paths.RAW_DATA, Paths.PROCESSED_DATA, Paths.MODELS, Paths.RESULTS]:
        path.mkdir(parents=True, exist_ok=True)
