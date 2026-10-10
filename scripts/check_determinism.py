#!/usr/bin/env python3
"""
Test determinism of run_thermal_sweep.py:
Runs run_thermal_sweep twice with identical seeds into two temporary CSV files
and asserts byte-identical output.
"""
import tempfile
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_thermal_sweep import run_thermal_sweep

def main():
    t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    t1.close()
    t2.close()
    try:
        print("Running run_thermal_sweep (pass 1)...")
        run_thermal_sweep(num_seeds=2, shots=100, output_csv=t1.name, max_cells=4)
        print("Running run_thermal_sweep (pass 2)...")
        run_thermal_sweep(num_seeds=2, shots=100, output_csv=t2.name, max_cells=4)

        with open(t1.name, "rb") as f1, open(t2.name, "rb") as f2:
            bytes1 = f1.read()
            bytes2 = f2.read()

        identical = (bytes1 == bytes2)
        print(f"Pass 1 size: {len(bytes1)} bytes")
        print(f"Pass 2 size: {len(bytes2)} bytes")
        print(f"CSVs byte-identical: {identical}")
        assert identical, "CSVs are NOT byte-identical!"
    finally:
        if os.path.exists(t1.name):
            os.remove(t1.name)
        if os.path.exists(t2.name):
            os.remove(t2.name)
        sidecar1 = t1.name.replace(".csv", ".provenance.json")
        sidecar2 = t2.name.replace(".csv", ".provenance.json")
        if os.path.exists(sidecar1):
            os.remove(sidecar1)
        if os.path.exists(sidecar2):
            os.remove(sidecar2)

if __name__ == "__main__":
    main()
