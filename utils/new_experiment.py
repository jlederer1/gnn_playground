# utils/new_experiment.py

"""
copies configs/example.yaml -> configs/new_experiment{n}.yaml, 
with n being the next available integer based on existing configs.
"""

import glob 
import shutil
from pathlib import Path 

def main():
    # Get highest existing experiment index
    root = Path(__file__).parent.parent
    config_dir = root / "configs"
    pattern = str(config_dir / "new_experiment*.yaml")
    existing_files = glob.glob(pattern)
    n = len(existing_files)  # Start indexing from 1

    # copy example configuration 
    src = config_dir / "example.yaml"
    destination = config_dir / f"new_experiment{n+1}.yaml" # Start from 1
    shutil.copy(src, destination)
    print(f"Created new experiment configuration: {destination}")


if __name__ == "__main__":
    main()