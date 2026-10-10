#!/usr/bin/env python3
"""
Reconciliation script:
Compares git show a905260:results/sim/thermal_sweep.csv with current results/sim/thermal_sweep.csv
for fake_guadalupe, bv, hot 1 and 2, every excess_p1:
- old paired gain (pts)
- new paired gain column (pts)
- (mean_score_esp_th - mean_score_esp_no) in pts
"""

import subprocess
import io
import pandas as pd

def main():
    old_csv_str = subprocess.check_output('git show a905260:results/sim/thermal_sweep.csv', shell=True).decode('utf-8')
    old_df = pd.read_csv(io.StringIO(old_csv_str))
    new_df = pd.read_csv('results/sim/thermal_sweep.csv')

    print("=" * 90)
    print("RECONCILIATION: FAKE_GUADALUPE BV (HOT 1 & HOT 2) ACROSS EXCESS LEVELS")
    print("=" * 90)
    print(f"{'Hot':<5} {'Excess p1':<11} {'Old Paired Gain':<22} {'New Paired Gain':<22} {'(ESP_th - ESP_no)':<22}")
    print("-" * 90)

    for k in [1, 2]:
        o_sub = old_df[(old_df.device=='fake_guadalupe')&(old_df.benchmark=='bv')&(old_df.hot_qubit_count==k)]
        n_sub = new_df[(new_df.device=='fake_guadalupe')&(new_df.benchmark=='bv')&(new_df.hot_qubit_count==k)]
        for exc in [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]:
            o_row = o_sub[o_sub.excess_p1 == exc].iloc[0]
            n_row = n_sub[n_sub.excess_p1 == exc].iloc[0]
            o_gain = o_row['paired_gain_mean'] * 100.0
            n_gain = n_row['paired_gain_mean'] * 100.0
            diff_th_no = (n_row['mean_score_esp_th'] - n_row['mean_score_esp_no']) * 100.0
            print(f"{k:<5d} {exc:<11.3f} {o_gain:+10.4f} pts        {n_gain:+10.4f} pts        {diff_th_no:+10.4f} pts")

    print("=" * 90)

if __name__ == "__main__":
    main()

