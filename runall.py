#!/usr/bin/env python3
"""
run_all.py — Master script: execute all four phases end-to-end.

Usage:
    python run_all.py

Phases:
    1 — Data generation + model architecture summary
    2 — Sanity check (forward pass) + adversarial training + metric logging
    3 — (Handled by hyperparameters in train.py)
    4 — Final evaluation + demo + plots
"""

import subprocess, sys, os

BASE = os.path.dirname(os.path.abspath(__file__))
PY   = sys.executable

def run(script, label):
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")
    result = subprocess.run([PY, os.path.join(BASE, script)], check=True)
    return result

if __name__ == "__main__":
    run("generate_data.py",  "Phase 1 — Data Generation")
    run("models.py",         "Phase 1 — Model Architecture Summary")
    run("train.py",          "Phase 2 & 3 — Adversarial Training")
    run("evaluate.py",       "Phase 4 — Evaluation & Demo")
    print("\n✓ All phases complete.")