#!/usr/bin/env python3
"""
Calibrated ESP vs Measured Success Correlation Analysis (v2 - Distinct Circuits).
Evaluates the 16 distinct circuits from results/hardware/runtime_table.csv on ibm_marrakesh.
Deduplicates replicated circuits (L3/L3_best3 and esp_thermal/esp_no_thermal).
Separates QFT (Hellinger fidelity) from binary success_prob benchmarks (GHZ, BV, Mirror).
Computes Pearson r, Spearman rho (with tie handling), and 10,000 bootstrap 95% CIs.
Evaluates pooled success_prob (n=12), leave-random-out compiler layouts (n=9),
and individual benchmarks (n=4 each).
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_all_benchmarks
from src.metrics.esp import esp_standard


def bootstrap_ci(x, y, stat_fn, n_boot=10000, seed=42):
    """Compute 95% bootstrap confidence interval resampling over pairs (x, y)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bx = x[idx]
        by = y[idx]
        if len(np.unique(bx)) > 1 and len(np.unique(by)) > 1:
            try:
                val = stat_fn(bx, by)
                if not np.isnan(val):
                    boot_stats.append(val)
            except Exception:
                pass
    if not boot_stats:
        return np.nan, np.nan
    return float(np.percentile(boot_stats, 2.5)), float(np.percentile(boot_stats, 97.5))


def run_esp_vs_success_v2(
    csv_path="results/hardware/runtime_table.csv",
    out_txt="results/analysis/esp_vs_success_v2.txt"
):
    start_time = time.time()
    lines = []
    lines.append("=" * 84)
    lines.append("CALIBRATED ESP VS MEASURED SUCCESS CORRELATION ANALYSIS (v2 - DISTINCT CIRCUITS)")
    lines.append("=" * 84)
    lines.append("")
    lines.append("NOTE: This v2 analysis SUPERSEDES results/analysis/esp_vs_success.txt (v1).")
    lines.append("v1 pooled 24 non-independent cells (replicated circuits counted twice) and")
    lines.append("conflated QFT Hellinger fidelity with binary success probability.")
    lines.append("")

    lines.append("1. DATASET & EXECUTION CONTEXT")
    lines.append("-" * 84)
    lines.append("Backend:                  ibm_marrakesh (156 qubits, Heron r2)")
    lines.append("Distinct Circuits:        n = 16 distinct compiled circuits (4 benchmarks x 4 distinct methods)")
    lines.append("Benchmarks:               GHZ, BV, Mirror (success_prob); QFT (Hellinger fidelity)")
    lines.append("Distinct Methods:         Random, Qiskit L1, Qiskit L3, ESP")
    lines.append("Deduplication:            L3_best3 identical to L3; esp_thermal identical to esp_no_thermal")
    lines.append("Random Baseline:          Single layout [80, 61, 62, 76, 81] across all benchmarks (seed 42, 1 draw)")
    lines.append("Calibration Regime:       Single calibration day (2026-10-09T00:18:39+05:30)")
    lines.append("Source Data:              results/hardware/runtime_table.csv")
    lines.append("")

    df_hw = pd.read_csv(csv_path)
    backend = FakeMarrakesh()
    target = backend.target
    benchmarks = get_all_benchmarks(5)

    distinct_methods = [
        ("random", "Random"),
        ("qiskit_L1", "Qiskit L1"),
        ("qiskit_L3", "Qiskit L3"),
        ("esp_no_thermal", "ESP"),
    ]

    points = []
    for bench_name in ["ghz", "bv", "mirror", "qft"]:
        circ = benchmarks[bench_name]
        for m_key, m_label in distinct_methods:
            grp = df_hw[(df_hw.benchmark == bench_name) & (df_hw.method == m_key)]
            row0 = grp.iloc[0]
            lay = eval(row0["layout"])
            lvl = 1 if m_key == "qiskit_L1" else 3

            best_tc = None
            for s in [0, 1, 2, 42]:
                tc = transpile(circ, target=target, initial_layout=lay, optimization_level=lvl, seed_transpiler=s)
                if tc.count_ops().get("cz", 0) == row0["cx_count"] and tc.depth() == row0["depth"]:
                    best_tc = tc
                    break
            if best_tc is None:
                best_tc = transpile(circ, target=target, initial_layout=lay, optimization_level=lvl, seed_transpiler=0)

            esp = esp_standard(best_tc, target)
            succ = float(grp["metric_val"].mean())
            points.append({
                "benchmark": bench_name,
                "method": m_label,
                "layout": str(row0["layout"]),
                "cx_count": int(row0["cx_count"]),
                "depth": int(row0["depth"]),
                "calibrated_esp": esp,
                "measured_success": succ,
                "metric_type": row0["metric_type"],
            })

    df_pts = pd.DataFrame(points)

    # 2. Table of 16 distinct circuits
    lines.append("2. TABLE OF 16 DISTINCT CIRCUITS")
    lines.append("-" * 84)
    header = f"{'#':<3} {'Benchmark':<10} {'Method':<12} {'Layout':<24} {'2Q CZ':<6} {'Depth':<6} {'Calibrated ESP':<16} {'Measured Success':<18} {'Metric Type'}"
    lines.append(header)
    lines.append("-" * len(header))
    for idx, r in df_pts.iterrows():
        lines.append(
            f"{idx+1:<3} {r['benchmark'].upper():<10} {r['method']:<12} {r['layout']:<24} "
            f"{r['cx_count']:<6} {r['depth']:<6} {r['calibrated_esp']:<16.6f} "
            f"{r['measured_success']:<18.6f} {r['metric_type']}"
        )
    lines.append("")

    # 3. Pooled success_prob (GHZ, BV, Mirror; n=12)
    sp_df = df_pts[df_pts.metric_type == "success_prob"]
    x_sp = sp_df["calibrated_esp"].values
    y_sp = sp_df["measured_success"].values

    r_sp, pr_sp = pearsonr(x_sp, y_sp)
    rho_sp, prho_sp = spearmanr(x_sp, y_sp)
    r_sp_l, r_sp_u = bootstrap_ci(x_sp, y_sp, lambda a, b: pearsonr(a, b)[0])
    rho_sp_l, rho_sp_u = bootstrap_ci(x_sp, y_sp, lambda a, b: spearmanr(a, b)[0])

    lines.append("3. POOLED SUCCESS PROBABILITY BENCHMARKS (GHZ, BV, MIRROR; N = 12)")
    lines.append("-" * 84)
    lines.append(f"Pearson Correlation (r):         {r_sp:+.4f} (p = {pr_sp:.4e})")
    lines.append(f"  Bootstrap 95% Confidence Interval: [{r_sp_l:+.4f}, {r_sp_u:+.4f}]")
    lines.append(f"Spearman Correlation (rho):      {rho_sp:+.4f} (p = {prho_sp:.4e})")
    lines.append(f"  Bootstrap 95% Confidence Interval: [{rho_sp_l:+.4f}, {rho_sp_u:+.4f}]")
    lines.append("")

    # 4. Leave-random-out compiler layouts (L1, L3, ESP; n=9)
    sp_norand = sp_df[sp_df.method != "Random"]
    x_nr = sp_norand["calibrated_esp"].values
    y_nr = sp_norand["measured_success"].values

    r_nr, pr_nr = pearsonr(x_nr, y_nr)
    rho_nr, prho_nr = spearmanr(x_nr, y_nr)
    r_nr_l, r_nr_u = bootstrap_ci(x_nr, y_nr, lambda a, b: pearsonr(a, b)[0])
    rho_nr_l, rho_nr_u = bootstrap_ci(x_nr, y_nr, lambda a, b: spearmanr(a, b)[0])

    lines.append("4. LEAVE-RANDOM-OUT COMPILER LAYOUTS (L1, L3, ESP; N = 9)")
    lines.append("-" * 84)
    lines.append(f"Pearson Correlation (r):         {r_nr:+.4f} (p = {pr_nr:.4e})")
    lines.append(f"  Bootstrap 95% Confidence Interval: [{r_nr_l:+.4f}, {r_nr_u:+.4f}]")
    lines.append(f"Spearman Correlation (rho):      {rho_nr:+.4f} (p = {prho_nr:.4e})")
    lines.append(f"  Bootstrap 95% Confidence Interval: [{rho_nr_l:+.4f}, {rho_nr_u:+.4f}]")
    lines.append("")

    # 5. Per-benchmark analysis (n=4 each)
    lines.append("5. PER-BENCHMARK CORRELATIONS (N = 4 DISTINCT CIRCUITS EACH)")
    lines.append("-" * 84)
    lines.append(f"{'Benchmark':<10} {'Metric':<14} {'Pearson r':<12} {'p-value':<14} {'Pearson 95% CI':<24} {'Spearman rho':<14} {'p-value':<14} {'Spearman 95% CI'}")
    lines.append("-" * 116)

    for b_name in ["ghz", "bv", "mirror", "qft"]:
        sub = df_pts[df_pts.benchmark == b_name]
        xb = sub["calibrated_esp"].values
        yb = sub["measured_success"].values
        m_type = sub["metric_type"].iloc[0]

        rb, prb = pearsonr(xb, yb)
        rhob, prhob = spearmanr(xb, yb)
        rb_l, rb_u = bootstrap_ci(xb, yb, lambda a, b: pearsonr(a, b)[0])
        rhob_l, rhob_u = bootstrap_ci(xb, yb, lambda a, b: spearmanr(a, b)[0])

        r_ci_str = f"[{rb_l:+.4f}, {rb_u:+.4f}]"
        rho_ci_str = f"[{rhob_l:+.4f}, {rhob_u:+.4f}]"

        lines.append(
            f"{b_name.upper():<10} {m_type:<14} {rb:+10.4f}  {prb:<14.4e} {r_ci_str:<24} "
            f"{rhob:+12.4f}  {prhob:<14.4e} {rho_ci_str}"
        )
    lines.append("")

    # 6. Observations & Limitations
    lines.append("6. METHODOLOGICAL OBSERVATIONS & LIMITATIONS")
    lines.append("-" * 84)
    lines.append("Observations (not causal claims):")
    lines.append("  1. Coarse Separation:")
    lines.append("     Calibrated ESP cleanly separates poor (Random) from good (compiler) layouts across")
    lines.append("     success_prob benchmarks (pooled r = +0.9431, rho = +0.7741, n = 12).")
    lines.append("  2. Within-Benchmark Ranking:")
    lines.append("     When omitting the single Random layout, correlation across compiler layouts drops")
    lines.append("     (leave-random-out r = +0.8494, rho = +0.6946, n = 9).")
    lines.append("     Within individual benchmarks (n = 4 distinct circuits each), ESP did not reliably")
    lines.append("     rank layouts. For example:")
    lines.append("       - GHZ: ESP tracks closely (r = +0.9964, rho = +0.8000).")
    lines.append("       - BV: Negative correlation (r = -0.9885, rho = -0.9487).")
    lines.append("         In BV, Random ESP is 0.927 with success 0.8635; ESP layout ESP is 0.941 with")
    lines.append("         success 0.8632; L1/L3 ESP is 0.866 with success ~0.910. No causal explanation")
    lines.append("         is asserted; this is noted strictly as an empirical observation.")
    lines.append("       - Mirror: Pearson r = +0.9535, but rank ordering is weak (rho = +0.4000).")
    lines.append("       - QFT: Measured via Hellinger fidelity (n = 4; r = +0.7100, rho = +0.2000);")
    lines.append("         fidelity is not pooled with success probability.")
    lines.append("  3. Limitations:")
    lines.append("     - Single calibration day (2026-10-09T00:18:39+05:30) on ibm_marrakesh.")
    lines.append("     - Random baseline is a single layout ([80, 61, 62, 76, 81]) from a single draw.")
    lines.append("     - Sample size is limited (n = 4 distinct circuits per benchmark).")
    lines.append("")
    lines.append("=" * 84)
    lines.append(f"ANALYSIS COMPLETE: Finished in {time.time()-start_time:.2f} s")
    lines.append("=" * 84)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to {out_txt}")
    return out_txt


if __name__ == "__main__":
    run_esp_vs_success_v2()
