"""
stratified_evaluate.py

Convenience wrapper for Step 5 analysis once predictions are collected.
"""

import subprocess
import sys


def main():
    cmd = [sys.executable, "analysis/zone_horizon_breakdown.py"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
