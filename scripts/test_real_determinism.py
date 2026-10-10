#!/usr/bin/env python3
"""
Real determinism test for thermal sweep:
Runs run_thermal_sweep.py with committed settings (10 seeds, 8192 shots)
into a temporary CSV and compares with results/sim/thermal_sweep.csv.
Prints whether the CSVs are byte-identical, and if not, prints the max
absolute difference per column.
"""
import os
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_thermal_sweep import run_thermal_sweep

def main():
    target_csv = Path("results/sim/thermal_sweep.csv")
    if not target_csv.exists():
        print(f"Error: {target_csv} not found.")
        sys.exit(1)

    t_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    temp_path = Path(t_file.name)
    t_file.close()

    try:
        print("Starting real determinism sweep run (10 seeds, 8192 shots)...")
        run_thermal_sweep(num_seeds=10, shots=8192, output_csv=str(temp_path))

        with open(target_csv, "rb") as f1, open(temp_path, "rb") as f2:
            b1 = f1.read()
            b2 = f2.read()

        is_byte_identical = (b1 == b2)
        print("\n" + "=" * 80)
        print("REAL DETERMINISM TEST RESULTS")
        print("=" * 80)
        print(f"Target CSV size:  {len(b1)} bytes ({target_csv})")
        print(f"Rerun CSV size:   {len(b2)} bytes ({temp_path})")
        print(f"Byte-identical:   {is_byte_identical}")

        df_target = pd.read_csv(target_csv)
        df_rerun = pd.read_csv(temp_path)

        if is_byte_identical:
            print("CSVs are 100% byte-identical across all 168 cells and 15 columns!")
        else:
            print("\nComputing column-wise differences:")
            numeric_cols = df_target.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                diff = np.abs(df_target[col].values - df_rerun[col].values)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f"  {col:<26}: max_abs_diff = {max_diff:.8e}, mean_abs_diff = {mean_diff:.8e}")

            str_cols = [c for c in df_target.columns if c not in numeric_cols]
            for col in str_cols:
                mismatch = (df_target[col] != df_rerun[col]).sum()
                print(f"  {col:<26}: string mismatches = {mismatch}")
        print("=" * 80)
    finally:
        if temp_path.exists():
            temp_path.unlink()
        sidecar = temp_path.with_suffix(".provenance.json")
        if sidecar.exists():
            sidecar.unlink()

if __name__ == "__main__":
    main()

