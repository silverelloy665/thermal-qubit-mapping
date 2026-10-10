#!/usr/bin/env python3
"""
Hardware statistical analysis script.
Pools the 2 repetitions from raw JSON count files in results/hardware/raw/.
Computes:
- Two-proportion z-test, Newcombe CI, and Cohen's h for success_prob benchmarks (GHZ, BV, Mirror)
- Multinomial bootstrap (10,000 resamples) of Hellinger fidelity for QFT
- Holm-Bonferroni correction across the 16 comparisons
- Outputs results/hardware_stats.csv and writes provenance JSON sidecar.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as stats

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_all_benchmarks
from src.metrics.outcome import get_ideal_support, compute_ideal_distribution
from scripts.make_provenance_and_summary import write_provenance

def load_pooled_counts(raw_dir: Path, benchmark: str, method: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in ['r0', 'r1']:
        fn = raw_dir / f"{benchmark}_{method}_{r}.json"
        if not fn.exists():
            raise FileNotFoundError(f"Missing raw count file: {fn}")
        with open(fn, "r", encoding="utf-8") as f:
            c = json.load(f)
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + int(v)
    return counts

def wilson_bounds(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2.0 * n)) / denom
    delta = (z / denom) * np.sqrt(p * (1.0 - p) / n + (z**2) / (4.0 * (n**2)))
    return float(center - delta), float(center + delta)

def newcombe_ci(k1: float, n1: float, k2: float, n2: float, z: float = 1.96) -> tuple[float, float]:
    l1, u1 = wilson_bounds(k1, n1, z)
    l2, u2 = wilson_bounds(k2, n2, z)
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2
    lower = diff - np.sqrt((p1 - l1)**2 + (u2 - p2)**2)
    upper = diff + np.sqrt((u1 - p1)**2 + (p2 - l2)**2)
    return float(lower), float(upper)

def two_proportion_z(k1: float, n1: float, k2: float, n2: float) -> tuple[float, float]:
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2
    p_pool = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return float(diff), 1.0
    z = diff / se
    p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    return float(diff), p_val

def cohens_h(p1: float, p2: float) -> float:
    phi1 = 2.0 * np.arcsin(np.sqrt(np.clip(p1, 0.0, 1.0)))
    phi2 = 2.0 * np.arcsin(np.sqrt(np.clip(p2, 0.0, 1.0)))
    return float(phi1 - phi2)

def holm_bonferroni(p_vals: np.ndarray) -> np.ndarray:
    m = len(p_vals)
    order = np.argsort(p_vals)
    p_adj = np.zeros(m, dtype=float)
    cur_max = 0.0
    for rank, idx in enumerate(order):
        mult = m - rank
        val = min(1.0, float(p_vals[idx]) * mult)
        cur_max = max(cur_max, val)
        p_adj[idx] = cur_max
    return p_adj

def analyze_hardware(raw_dir_str: str = "results/hardware/raw", output_csv_str: str = "results/hardware_stats.csv", seed: int = 42) -> pd.DataFrame:
    raw_dir = Path(raw_dir_str)
    output_csv = Path(output_csv_str)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    benchmarks = get_all_benchmarks(5)
    baselines = ['random', 'qiskit_L1', 'qiskit_L3', 'qiskit_L3_best3']
    
    print("=" * 80)
    print("HARDWARE STATISTICAL ANALYSIS: ibm_marrakesh (156q)")
    print("Note: Repetitions r0 and r1 are executed using identical compiled circuits;")
    print("      all reported confidence intervals quantify shot noise only.")
    print("=" * 80)
    
    records = []
    
    # 1. Success probability benchmarks
    for b in ['ghz', 'bv', 'mirror']:
        qc = benchmarks[b]
        support = get_ideal_support(b, qc)
        c_esp = load_pooled_counts(raw_dir, b, 'esp_no_thermal')
        k_esp = sum(c_esp.get(s, 0) for s in support)
        n_esp = sum(c_esp.values())
        p_esp = k_esp / n_esp
        
        for base in baselines:
            c_base = load_pooled_counts(raw_dir, b, base)
            k_base = sum(c_base.get(s, 0) for s in support)
            n_base = sum(c_base.values())
            p_base = k_base / n_base
            
            diff, p_val = two_proportion_z(k_esp, n_esp, k_base, n_base)
            ci_low, ci_high = newcombe_ci(k_esp, n_esp, k_base, n_base)
            eff = cohens_h(p_esp, p_base)
            
            records.append({
                'benchmark': b,
                'comparison': f'esp_vs_{base}',
                'metric_type': 'success_prob',
                'p_esp': p_esp,
                'p_baseline': p_base,
                'diff_points': diff * 100.0,
                'ci_95_low_pts': ci_low * 100.0,
                'ci_95_high_pts': ci_high * 100.0,
                'p_value_raw': p_val,
                'effect_size': eff,
                'effect_type': "cohens_h"
            })
            
    # 2. Fidelity benchmark: QFT
    qc_qft = benchmarks['qft']
    ideal_dist = compute_ideal_distribution(qc_qft)
    c_esp = load_pooled_counts(raw_dir, 'qft', 'esp_no_thermal')
    n_esp = sum(c_esp.values())
    
    np.random.seed(seed)
    n_resamples = 10000
    
    for base in baselines:
        c_base = load_pooled_counts(raw_dir, 'qft', base)
        n_base = sum(c_base.values())
        
        all_keys = sorted(list(set(c_esp) | set(c_base) | set(ideal_dist)))
        ideal_vec = np.array([ideal_dist.get(k, 0.0) for k in all_keys], dtype=float)
        
        cnt_esp = np.array([c_esp.get(k, 0) for k in all_keys], dtype=float)
        cnt_base = np.array([c_base.get(k, 0) for k in all_keys], dtype=float)
        
        p_esp = cnt_esp / n_esp
        p_base = cnt_base / n_base
        
        fid_esp_point = float((np.sum(np.sqrt(p_esp * ideal_vec)))**2)
        fid_base_point = float((np.sum(np.sqrt(p_base * ideal_vec)))**2)
        point_diff = fid_esp_point - fid_base_point
        
        draws_esp = np.random.multinomial(n_esp, p_esp, size=n_resamples) / float(n_esp)
        draws_base = np.random.multinomial(n_base, p_base, size=n_resamples) / float(n_base)
        
        boot_fid_esp = (np.sum(np.sqrt(draws_esp * ideal_vec[np.newaxis, :]), axis=1))**2
        boot_fid_base = (np.sum(np.sqrt(draws_base * ideal_vec[np.newaxis, :]), axis=1))**2
        boot_diffs = boot_fid_esp - boot_fid_base
        
        ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
        p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
        p_val = max(p_val, 1.0 / n_resamples) if p_val == 0 else p_val
        std_boot = float(np.std(boot_diffs))
        std_effect = point_diff / std_boot if std_boot > 0 else 0.0
        
        records.append({
            'benchmark': 'qft',
            'comparison': f'esp_vs_{base}',
            'metric_type': 'fidelity',
            'p_esp': fid_esp_point,
            'p_baseline': fid_base_point,
            'diff_points': point_diff * 100.0,
            'ci_95_low_pts': float(ci_low) * 100.0,
            'ci_95_high_pts': float(ci_high) * 100.0,
            'p_value_raw': p_val,
            'effect_size': std_effect,
            'effect_type': "standardized_diff"
        })
        
    df_res = pd.DataFrame(records)
    
    # 3. Holm-Bonferroni correction across the 16 comparisons
    df_res['p_value_holm'] = holm_bonferroni(df_res['p_value_raw'].values)
    
    # Save CSV
    df_res.to_csv(output_csv, index=False)
    print(f"Hardware statistical table written to {output_csv} ({len(df_res)} rows).")
    
    # Write provenance sidecar
    sidecar_path = write_provenance(output_csv)
    print(f"Provenance sidecar written to {sidecar_path}")
    
    # Pretty print summary table
    print("\nSUMMARY OF 16 HARDWARE COMPARISONS (ESP vs Baselines):")
    fmt_df = df_res[['benchmark', 'comparison', 'diff_points', 'ci_95_low_pts', 'ci_95_high_pts', 'p_value_raw', 'p_value_holm', 'effect_size', 'effect_type']]
    print(fmt_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:10.4f}"))
    print("=" * 80 + "\n")
    
    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze hardware statistical outcomes.")
    parser.add_argument("--hardware", action="store_true", help="Run hardware statistical analysis.")
    parser.add_argument("--raw-dir", default="results/hardware/raw", help="Path to raw JSON count directory.")
    parser.add_argument("--output", default="results/hardware_stats.csv", help="Output CSV path.")
    args = parser.parse_args()
    
    analyze_hardware(raw_dir_str=args.raw_dir, output_csv_str=args.output)

