import subprocess
import sys
from pathlib import Path


def run_script(script_name):
    src_dir = Path(__file__).resolve().parent
    script_path = src_dir / script_name

    print(f"\nRunning {script_name}...")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main():
    run_script("train_models.py")
    run_script("make_figures.py")

    print("\nProject pipeline completed.")


if __name__ == "__main__":
    main()