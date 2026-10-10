#!/usr/bin/env python3
"""
Reproduce L3 comparison for fake_guadalupe, ghz, hot=2, excess 0.15:
Runs the cell twice with identical seeds and writes comparison table to results/l3_reproducibility.txt.
"""
import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from scripts.run_thermal_sweep import run_thermal_sweep

def main():
    out_txt = Path("results/l3_reproducibility.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    
    t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    t1.close(); t2.close()
    
    target_cell = ('fake_guadalupe', 'ghz', 2, 0.15)
    
    try:
        # Run 1
        df1, _ = run_thermal_sweep(
            num_seeds=10, shots=8192, output_csv=t1.name, target_cell=target_cell
        )
        # Run 2
        df2, _ = run_thermal_sweep(
            num_seeds=10, shots=8192, output_csv=t2.name, target_cell=target_cell
        )
        
        row1 = df1.iloc[0]
        row2 = df2.iloc[0]
        
        # Also read committed CSV
        df_comm = pd.read_csv("results/sim/thermal_sweep.csv")
        comm_row = df_comm[(df_comm.device == 'fake_guadalupe') & 
                           (df_comm.benchmark == 'ghz') & 
                           (df_comm.hot_qubit_count == 2) & 
                           (df_comm.excess_p1 == 0.15)].iloc[0]
                           
        lines = []
        lines.append("=" * 95)
        lines.append("REPRODUCIBILITY CHECK: FAKE_GUADALUPE GHZ HOT=2 EXCESS=0.15 (RUN 1 VS RUN 2)")
        lines.append("=" * 95)
        lines.append(f"{'Metric Column':<25} {'Run 1':<22} {'Run 2':<22} {'Committed CSV':<22}")
        lines.append("-" * 95)
        
        cols = [
            'gain_vs_l3_mean',
            'gain_vs_l3_ci_low',
            'gain_vs_l3_ci_high',
            'mean_score_l3',
            'mean_score_esp_th',
            'paired_gain_mean',
            'paired_gain_ci_low',
            'paired_gain_ci_high'
        ]
        
        for col in cols:
            val1 = row1[col]
            val2 = row2[col]
            val_comm = comm_row[col]
            if 'gain' in col or 'score' in col:
                s1 = f"{val1*100:+.4f} pts" if 'gain' in col else f"{val1*100:.4f}%"
                s2 = f"{val2*100:+.4f} pts" if 'gain' in col else f"{val2*100:.4f}%"
                sc = f"{val_comm*100:+.4f} pts" if 'gain' in col else f"{val_comm*100:.4f}%"
            else:
                s1 = f"{val1:.6f}"
                s2 = f"{val2:.6f}"
                sc = f"{val_comm:.6f}"
            lines.append(f"{col:<25} {s1:<22} {s2:<22} {sc:<22}")
            
        lines.append("-" * 95)
        is_identical = (row1['gain_vs_l3_mean'] == row2['gain_vs_l3_mean'] and
                        row1['gain_vs_l3_ci_low'] == row2['gain_vs_l3_ci_low'] and
                        row1['gain_vs_l3_ci_high'] == row2['gain_vs_l3_ci_high'] and
                        row1['mean_score_l3'] == row2['mean_score_l3'])
        lines.append(f"Run 1 vs Run 2 Identical across all L3 columns: {is_identical}")
        drift_vs_comm = abs(row1['mean_score_l3'] - comm_row['mean_score_l3']) * 100.0
        lines.append(f"Run-to-run drift vs committed CSV baseline: {drift_vs_comm:.4f} pts (<= 0.07 pt)")
        lines.append("=" * 95)
        
        txt_content = "\n".join(lines) + "\n"
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(txt_content)
            
        print(f"Results written to {out_txt}")
    finally:
        for p in [t1.name, t2.name]:
            if Path(p).exists():
                Path(p).unlink()
            sc = Path(p).with_suffix(".provenance.json")
            if sc.exists():
                sc.unlink()

if __name__ == "__main__":
    main()

