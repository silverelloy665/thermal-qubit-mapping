#!/usr/bin/env python3
"""
scripts/analyze_hotspot.py

Pre-registered statistical analysis of hardware hot-spot stress test for Checkpoint Q.
Evaluates paired difference CLEAN - HOT per region and benchmark:
- Pools 2 repetitions (16,384 shots per cell)
- Computes Newcombe hybrid score 95% CI
- Two-proportion z-test with Holm-Bonferroni correction
- Fixed pre-registered decision rule:
  "hot-spot penalty confirmed" for a benchmark only if the pooled CLEAN - HOT
  difference has CI lower bound > 0 AND is >= 0.5 pt in at least 2 regions.
- Outputs results/hardware_extra_stats.csv and provenance JSON.
"""

import os
import sys
import json
import math
import argparse
import datetime
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
from scipy.stats import norm

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_ghz, get_bv_n, get_mirror
from src.metrics.outcome import compute_outcome_metric, compute_ideal_distribution, get_ideal_support

def wilson_score_interval(k, n, z=1.95996398454):
    if n == 0: return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z**2) / (4 * n**2))) / denom
    return max(0.0, center - half), min(1.0, center + half)

def newcombe_diff_ci(k1, n1, k2, n2, z=1.95996398454):
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2
    l1, u1 = wilson_score_interval(k1, n1, z)
    l2, u2 = wilson_score_interval(k2, n2, z)
    ci_low = diff - math.sqrt((p1 - l1)**2 + (u2 - p2)**2)
    ci_high = diff + math.sqrt((u1 - p1)**2 + (p2 - l2)**2)
    return diff, ci_low, ci_high

def two_prop_z_test(k1, n1, k2, n2):
    p1 = k1 / n1
    p2 = k2 / n2
    diff = p1 - p2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1.0 - p_pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0: return 0.0, 1.0
    z = diff / se
    pval = 2.0 * (1.0 - norm.cdf(abs(z)))
    return z, pval

def generate_synthetic_counts(plan_data, target_effect_pts=1.2, shots_per_rep=8192):
    """
    Generates synthetic counts for known-answer testing.
    CLEAN has higher success rate than HOT by target_effect_pts.
    """
    raw_synthetic = {}
    base_success_rates = {
        "ghz": 0.91,
        "bv": 0.90,
        "mirror": 0.72,
    }
    benchmarks_circs = {
        "ghz": get_ghz(5),
        "bv": get_bv_n(5),
        "mirror": get_mirror(5),
    }
    rng = np.random.RandomState(42)

    for r in plan_data.get("regions", []):
        r_id = r["region_id"]
        for b_name in r["benchmarks"]:
            b_circ = benchmarks_circs[b_name]
            base_p = base_success_rates.get(b_name, 0.85)
            clean_p = min(0.98, base_p + (target_effect_pts / 100.0) / 2.0)
            hot_p = max(0.02, base_p - (target_effect_pts / 100.0) / 2.0)

            supp = get_ideal_support(b_name, b_circ)
            primary_ideal = supp[0]
            n_bits = len(primary_ideal)
            non_ideal = "1" * n_bits if ("1" * n_bits) not in supp else "0" * (n_bits - 1) + "1"
            if non_ideal in supp:
                non_ideal = "01010"[:n_bits]

            for rep in range(2):
                # CLEAN
                clean_k = int(rng.binomial(shots_per_rep, clean_p))
                clean_counts = {primary_ideal: clean_k, non_ideal: shots_per_rep - clean_k}
                raw_synthetic[(r_id, b_name, "CLEAN", rep)] = clean_counts

                # HOT
                hot_k = int(rng.binomial(shots_per_rep, hot_p))
                hot_counts = {primary_ideal: hot_k, non_ideal: shots_per_rep - hot_k}
                raw_synthetic[(r_id, b_name, "HOT", rep)] = hot_counts

    return raw_synthetic

def analyze_hotspot(plan_path="results/hardware_extra/plan.json",
                    raw_dir="results/hardware_extra/raw",
                    out_csv="results/hardware_extra_stats.csv",
                    out_txt="results/hardware_extra/hotspot_analysis_summary.txt",
                    synthetic=False,
                    synthetic_effect_pts=1.2):
    plan_file = Path(plan_path)
    if not plan_file.exists():
        raise FileNotFoundError(f"Plan file '{plan_file}' not found. Run dry run first.")

    with open(plan_file, "r", encoding="utf-8") as f:
        plan_data = json.load(f)

    benchmarks_circs = {
        "ghz": get_ghz(5),
        "bv": get_bv_n(5),
        "mirror": get_mirror(5),
    }
    ideal_cache = {b: compute_ideal_distribution(circ) for b, circ in benchmarks_circs.items()}

    # Check for raw counts or synthetic mode
    raw_path = Path(raw_dir)
    has_real_counts = raw_path.exists() and len(list(raw_path.glob("*.json"))) >= 18

    if not has_real_counts:
        print(f"Notice: Real hardware raw counts not found in {raw_dir}. Using synthetic known-answer counts (effect = +{synthetic_effect_pts} pt).")
        raw_dict = generate_synthetic_counts(plan_data, target_effect_pts=synthetic_effect_pts)
        is_synthetic = True
    elif synthetic:
        print(f"Running synthetic known-answer evaluation (effect = +{synthetic_effect_pts} pt)...")
        raw_dict = generate_synthetic_counts(plan_data, target_effect_pts=synthetic_effect_pts)
        is_synthetic = True
    else:
        print(f"Loading real hardware execution counts from {raw_dir}...")
        raw_dict = {}
        for fn in raw_path.glob("r*.json"):
            # Format: r{r_id}_{benchmark}_{condition}_rep{rep}.json
            stem = fn.stem
            parts = stem.split("_")
            r_id = int(parts[0][1:])
            b_name = parts[1]
            cond = parts[2]
            rep = int(parts[3].replace("rep", ""))
            with open(fn, "r", encoding="utf-8") as f:
                raw_dict[(r_id, b_name, cond, rep)] = json.load(f)
        is_synthetic = False

    rows = []
    benchmarks_list = ["ghz", "bv", "mirror"]

    for r in plan_data.get("regions", []):
        r_id = r["region_id"]
        h_q = r["hot_qubit"]
        exc_q = r["excess_p1"]

        for b_name in benchmarks_list:
            if b_name not in r.get("benchmarks", {}):
                continue

            b_meta = r["benchmarks"][b_name]
            circ = benchmarks_circs[b_name]
            ideal = ideal_cache[b_name]

            # Pool repetition 0 and 1
            counts_clean_r0 = raw_dict.get((r_id, b_name, "CLEAN", 0), {})
            counts_clean_r1 = raw_dict.get((r_id, b_name, "CLEAN", 1), {})
            counts_hot_r0 = raw_dict.get((r_id, b_name, "HOT", 0), {})
            counts_hot_r1 = raw_dict.get((r_id, b_name, "HOT", 1), {})

            pool_clean = Counter(counts_clean_r0) + Counter(counts_clean_r1)
            pool_hot = Counter(counts_hot_r0) + Counter(counts_hot_r1)

            n_clean = sum(pool_clean.values()) or 16384
            n_hot = sum(pool_hot.values()) or 16384

            m_type, score_clean = compute_outcome_metric(b_name, pool_clean, circ, ideal, shots=n_clean)
            _, score_hot = compute_outcome_metric(b_name, pool_hot, circ, ideal, shots=n_hot)

            k_clean = int(round(score_clean * n_clean))
            k_hot = int(round(score_hot * n_hot))

            diff_val, ci_low, ci_high = newcombe_diff_ci(k_clean, n_clean, k_hot, n_hot)
            z_stat, p_val = two_prop_z_test(k_clean, n_clean, k_hot, n_hot)

            diff_pts = diff_val * 100.0
            ci_low_pts = ci_low * 100.0
            ci_high_pts = ci_high * 100.0

            is_confounded = bool(b_meta.get("confounded", False))
            conf_reason = str(b_meta.get("confounded_reason", "None"))

            rows.append({
                "region_id": r_id,
                "hot_qubit": h_q,
                "excess_p1": exc_q,
                "benchmark": b_name,
                "metric_name": m_type,
                "clean_shots": n_clean,
                "hot_shots": n_hot,
                "clean_successes": k_clean,
                "hot_successes": k_hot,
                "clean_score": score_clean,
                "hot_score": score_hot,
                "diff_pts": diff_pts,
                "ci_low_pts": ci_low_pts,
                "ci_high_pts": ci_high_pts,
                "z_stat": z_stat,
                "p_value_raw": p_val,
                "diff_2q": b_meta.get("diff_2q", 0),
                "confounded": is_confounded,
                "confounded_reason": conf_reason,
                "diff_depth": b_meta.get("diff_depth", 0),
                "hot_2q": b_meta.get("hot_2q", 0),
                "clean_2q": b_meta.get("clean_2q", 0),
                "hot_depth": b_meta.get("hot_depth", 0),
                "clean_depth": b_meta.get("clean_depth", 0),
                "esp_clean": b_meta.get("clean_esp", 0.0),
                "esp_hot": b_meta.get("hot_esp", 0.0),
                "diff_esp": b_meta.get("clean_esp", 0.0) - b_meta.get("hot_esp", 0.0),
                "thermal_equals_clean": b_meta.get("thermal_equals_clean", True),
                "penalty_confirmed_cell": (ci_low_pts > 0.0 and diff_pts >= 0.5 and not is_confounded)
            })

    df_stats = pd.DataFrame(rows)

    # Holm-Bonferroni correction across all comparisons
    if not df_stats.empty:
        df_stats = df_stats.sort_values("p_value_raw").reset_index(drop=True)
        m = len(df_stats)
        holm_pvals = []
        running_max = 0.0
        for i, r in df_stats.iterrows():
            factor = m - i
            adj_p = min(1.0, r["p_value_raw"] * factor)
            running_max = max(running_max, adj_p)
            holm_pvals.append(min(1.0, running_max))
        df_stats["p_value_holm"] = holm_pvals
        df_stats = df_stats.sort_values(["benchmark", "region_id"]).reset_index(drop=True)
    else:
        df_stats["p_value_holm"] = []

    # Write CSV
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_stats.to_csv(out_csv, index=False)

    # Write Provenance JSON
    prov_path = Path(out_csv).with_suffix(".provenance.json")
    prov_data = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend": plan_data.get("backend"),
        "calibration_timestamp": plan_data.get("calibration_timestamp"),
        "plan_hash": plan_data.get("plan_hash"),
        "is_synthetic": is_synthetic,
        "shots_pooled_per_cell": 16384,
        "repetitions": 2,
        "decision_rule": "region counts if own CLEAN-HOT Newcombe CI low > 0 and diff >= 0.5 pt; benchmark confirmed if >= 2 valid (non-confounded) regions count",
        "rows": len(df_stats)
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)

    # Generate Report Header & Text Summary
    lines = []
    lines.append("=" * 80)
    lines.append("PRE-REGISTERED HARDWARE HOT-SPOT STRESS TEST: STATISTICAL ANALYSIS REPORT")
    lines.append(f"Analysis Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("HEADER METHODOLOGICAL DISCLOSURES:")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("  1. Repetitions use identical circuits so CIs cover shot noise only.")
    lines.append("  2. One backend, one calibration day (strict single-epoch calibration isolation).")
    lines.append("  3. Regions chosen by measured excess excited-state population (excess_p1 > 0.05).")
    lines.append("  4. HOT and CLEAN differ in more than the hot qubit (native 2Q count and calibrated ESP differ).")
    lines.append(f"  5. Evaluation Mode: {'SYNTHETIC COUNTS (Known-Answer Test)' if is_synthetic else 'REAL HARDWARE COUNTS (IBM Quantum)'}")
    lines.append("")
    lines.append("SUMMARY OF REGIONAL DIFFERENCES IN CIRCUIT PROPERTIES (HOT vs CLEAN):")
    lines.append("--------------------------------------------------------------------------------")
    for r in plan_data.get("regions", []):
        lines.append(f"  Region {r['region_id']} (Hot Qubit: {r['hot_qubit']}, Excess p1: {r['excess_p1']:.4f}, Readout Err: {r['readout_error']:.4f}):")
        for b_name in benchmarks_list:
            if b_name in r["benchmarks"]:
                bm = r["benchmarks"][b_name]
                lines.append(
                    f"    - {b_name:<6}: Calibrated ESP HOT={bm['hot_esp']:.4f}, CLEAN={bm['clean_esp']:.4f} (diff: {bm['clean_esp']-bm['hot_esp']:+.4f}) | "
                    f"Native 2Q HOT={bm['hot_2q']}, CLEAN={bm['clean_2q']} (diff: {bm['diff_2q']:+d}) | "
                    f"Depth HOT={bm['hot_depth']}, CLEAN={bm['clean_depth']} (diff: {bm['diff_depth']:+d})"
                )
    lines.append("")
    lines.append("PAIRED DIFFERENCE RESULTS TABLE (CLEAN - HOT):")
    lines.append("-" * 105)
    hdr = f"{'Benchmark':<9} {'Region':<8} {'Clean %':<9} {'Hot %':<9} {'Diff (pts)':<12} {'95% Newcombe CI':<20} {'z-stat':<8} {'p (raw)':<11} {'p (Holm)':<11} {'Cell Pass':<9}"
    lines.append(hdr)
    lines.append("-" * 105)
    for idx, r in df_stats.iterrows():
        p_raw_s = f"{r['p_value_raw']:.4e}" if r['p_value_raw'] >= 1e-4 else "<1e-4"
        p_holm_s = f"{r['p_value_holm']:.4e}" if r['p_value_holm'] >= 1e-4 else "<1e-4"
        ci_s = f"[{r['ci_low_pts']:+.3f}, {r['ci_high_pts']:+.3f}]"
        lines.append(
            f"{r['benchmark']:<9} R{r['region_id']:<7} {r['clean_score']*100:<9.2f} {r['hot_score']*100:<9.2f} "
            f"{r['diff_pts']:<12.3f} {ci_s:<20} {r['z_stat']:<8.3f} {p_raw_s:<11} {p_holm_s:<11} "
            f"{str(r['penalty_confirmed_cell']):<9}"
        )
    lines.append("-" * 105)
    lines.append("")

    # Decision Rule Evaluation
    lines.append("DECISION RULE VERIFICATION (AMENDED IN CHECKPOINT S):")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("Pre-registered criterion: a region counts if its own CLEAN - HOT Newcombe CI lower bound > 0")
    lines.append("and difference >= 0.5 pt. The benchmark is confirmed if >= 2 valid (non-confounded) regions count.")
    lines.append("Cells with |native 2q difference HOT vs CLEAN| > 4 are designated as 'confounded'")
    lines.append("and excluded from contributing to the confirmation threshold, but reported.")
    lines.append("")
    benchmark_decisions = {}
    for b_name in benchmarks_list:
        sub = df_stats[df_stats["benchmark"] == b_name]
        valid_sub = sub[sub["confounded"] == False]
        passing_regions = valid_sub[valid_sub["penalty_confirmed_cell"] == True]
        num_valid = len(valid_sub)
        num_pass = len(passing_regions)
        if num_valid < 2:
            status_str = f"INCONCLUSIVE / UNCONFIRMED (only {num_valid}/3 regions are non-confounded; cannot reach >= 2 threshold)"
            is_confirmed = False
        elif num_pass >= 2:
            status_str = f"CONFIRMED ({num_pass}/{num_valid} valid non-confounded regions meet CI low > 0 and diff >= 0.5 pt)"
            is_confirmed = True
        else:
            status_str = f"NOT CONFIRMED ({num_pass}/{num_valid} valid non-confounded regions meet criterion)"
            is_confirmed = False
        benchmark_decisions[b_name] = (is_confirmed, num_pass, num_valid)
        lines.append(f"  Benchmark '{b_name:<6}': {status_str}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF STATISTICAL ANALYSIS")
    lines.append("=" * 80)

    report_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"Stats written to {out_csv} and report to {out_txt}.")

def main():
    parser = argparse.ArgumentParser(description="Analyze Hotspot Hardware Stress Test Results")
    parser.add_argument("--plan", default="results/hardware_extra/plan.json", help="Path to plan.json")
    parser.add_argument("--raw-dir", default="results/hardware_extra/raw", help="Path to raw JSON counts")
    parser.add_argument("--out-csv", default="results/hardware_extra_stats.csv", help="Path to output stats CSV")
    parser.add_argument("--out-txt", default="results/hardware_extra/hotspot_analysis_summary.txt", help="Path to output summary TXT")
    parser.add_argument("--synthetic", action="store_true", help="Run known-answer test with synthetic counts")
    parser.add_argument("--effect", type=float, default=1.2, help="Synthetic effect size in percentage points")
    args = parser.parse_args()

    analyze_hotspot(
        plan_path=args.plan,
        raw_dir=args.raw_dir,
        out_csv=args.out_csv,
        out_txt=args.out_txt,
        synthetic=args.synthetic,
        synthetic_effect_pts=args.effect
    )

if __name__ == "__main__":
    main()
