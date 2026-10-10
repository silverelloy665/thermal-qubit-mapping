#!/usr/bin/env python3
"""
Print task 3 table from results/sim/thermal_sweep.csv:
For fake_guadalupe, print gain of esp_thermal vs qiskit_L3 (points, 95% CI)
for every benchmark, hot count, excess_p1, and absolute mean success of
qiskit_L3, esp_no_thermal, and esp_thermal in each cell.
"""

import pandas as pd

def main():
    df = pd.read_csv("results/sim/thermal_sweep.csv")
    sub = df[df['device'] == 'fake_guadalupe']

    print("=" * 115)
    print("FAKE_GUADALUPE: GAIN VS QISKIT_L3 AND ABSOLUTE MEAN SUCCESS PER CELL")
    print("=" * 115)
    print(f"{'Benchmark':<10} {'Hot':<5} {'Excess p1':<10} {'Gain vs L3 (pts) [95% CI]':<35} {'Qiskit L3':<14} {'ESP Plain':<14} {'ESP Thermal':<14}")
    print("-" * 115)

    for _, r in sub.iterrows():
        b = r['benchmark']
        k = int(r['hot_qubit_count'])
        exc = r['excess_p1']
        gain_pts = r['gain_vs_l3_mean'] * 100.0
        low_pts = r['gain_vs_l3_ci_low'] * 100.0
        high_pts = r['gain_vs_l3_ci_high'] * 100.0
        gain_str = f"{gain_pts:+7.4f} pts [{low_pts:+7.4f}, {high_pts:+7.4f}]"
        l3_mean = r.get('mean_score_l3', float('nan'))
        esp_no = r.get('mean_score_esp_no', float('nan'))
        esp_th = r.get('mean_score_esp_th', float('nan'))
        print(f"{b:<10} {k:<5d} {exc:<10.3f} {gain_str:<35} {l3_mean:<14.4f} {esp_no:<14.4f} {esp_th:<14.4f}")

    print("=" * 115)

if __name__ == "__main__":
    main()

