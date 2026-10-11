#!/usr/bin/env python3
"""
scripts/print_checkpoint_q_diffs.py

Extracts raw git diffs of scripts/check_secrets.py and scripts/claim_check.py
from commit 53a29b8 and writes them to results/checkpoint_q_diffs.txt.
"""

import subprocess
from pathlib import Path

def print_diffs(out_path="results/checkpoint_q_diffs.txt"):
    cmd = ["git", "show", "--format=commit %h (%s)", "53a29b8", "--", "scripts/check_secrets.py", "scripts/claim_check.py"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(res.stdout)
        
    print(res.stdout)

if __name__ == "__main__":
    print_diffs()

