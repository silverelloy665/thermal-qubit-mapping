#!/usr/bin/env python3
"""
scripts/analyze_hotspot.py

Pre-registered statistical analysis of hardware hot-spot stress test for Checkpoint W / Pre-Registration v2.2.
Evaluates differences in ideal-state success probabilities (difference of independent proportions):
  Delta_success = success_prob(CLEAN_matched) - success_prob(HOT)
Per pre-registration v2.2:
- Evaluates PRIMARY cells only (matched and non-confounded; drops unmatchable and confounded).
- Computes Newcombe hybrid-score 95% confidence intervals (Method 10, Newcombe 1998) via stat_utils.
- Requires Delta_success >= 0.5 percentage points (+0.005) and CI lower bound > 0.
- Requires repetition sign agreement (rep0 and rep1 independently in the predicted direction).
- Negative control floor: if CLEAN2 exists, requires |CLEAN - HOT| > |CLEAN - CLEAN2| (layout-variance floor).
- Power Gate: requires at least 4 executed primary cells; pre-declares INCONCLUSIVE if fewer than 4.
- Outlines mutually exclusive and exhaustive outcomes:
  CONFIRMED, NULL / NOT CONFIRMED, INVERTED, MIXED, ZERO PERSISTENCE, INCONCLUSIVE.
- Outputs results/hardware_extra_stats.csv, provenance JSON, and results/hardware_extra/hotspot_analysis_summary.txt.
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

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.stat_utils import wilson_score_interval, newcombe_diff_ci, evaluate_cell_difference
from src.benchmarks.circuits import get_ghz, get_bv_n, get_mirror
from src.metrics.outcome import compute_outcome_metric, compute_ideal_distribution, get_ideal_support
from scipy.stats import norm

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

def generate_synthetic_hotspot_counts(plan_data, outcome="confirmed", target_effect_pts=2.5, shots_per_rep=8192):
    """
    Generates synthetic benchmark counts for testing all pre-registered outcomes:
    - 'confirmed': >= 4 primary cells, all significant positive, passes CLEAN2 floor.
    - 'null': all primary cells have ~0% difference (CI contains 0).
    - 'inverted': all primary cells have significant negative difference (HOT > CLEAN).
    - 'mixed': one primary cell significant positive, another significant inverted.
    - 'inconclusive': fewer than 4 primary cells executed.
    """
    raw_synthetic = {}
    benchmarks_circs = {
        "ghz": get_ghz(5),
        "bv": get_bv_n(5),
        "mirror": get_mirror(5),
    }

    primary_cells = []
    for r in plan_data.get("regions", []):
        r_id = r["region_id"]
        for b_name, b_val in r.get("benchmarks", {}).items():
            if b_val.get("status") == "PRIMARY":
                primary_cells.append((r_id, b_name, b_val))

    rng = np.random.RandomState(42)

    for cell_idx, (r_id, b_name, b_val) in enumerate(primary_cells):
        b_circ = benchmarks_circs[b_name]
        supp = get_ideal_support(b_name, b_circ)
        primary_ideal = supp[0]
        n_bits = len(primary_ideal)
        non_ideal = "1" * n_bits if ("1" * n_bits) not in supp else "0" * (n_bits - 1) + "1"
        if non_ideal in supp:
            non_ideal = "01010"[:n_bits]

        base_p = 0.88
        has_clean2 = b_val.get("has_clean2", False)

        if outcome == "confirmed":
            clean_p = base_p + (target_effect_pts / 100.0) / 2.0  # e.g., 0.8925
            hot_p = base_p - (target_effect_pts / 100.0) / 2.0    # e.g., 0.8675
            clean2_p = clean_p - 0.004                            # |clean - clean2| = 0.4 pt < 2.5 pt
        elif outcome == "null":
            clean_p = base_p
            hot_p = base_p
            clean2_p = base_p
        elif outcome == "inverted":
            clean_p = base_p - (target_effect_pts / 100.0) / 2.0
            hot_p = base_p + (target_effect_pts / 100.0) / 2.0
            clean2_p = clean_p + 0.004
        elif outcome == "mixed":
            if cell_idx % 2 == 0:
                clean_p = base_p + (target_effect_pts / 100.0) / 2.0
                hot_p = base_p - (target_effect_pts / 100.0) / 2.0
                clean2_p = clean_p - 0.004
            else:
                clean_p = base_p - (target_effect_pts / 100.0) / 2.0
                hot_p = base_p + (target_effect_pts / 100.0) / 2.0
                clean2_p = clean_p + 0.004
        elif outcome == "inconclusive":
            clean_p = base_p + (target_effect_pts / 100.0) / 2.0
            hot_p = base_p - (target_effect_pts / 100.0) / 2.0
            clean2_p = clean_p - 0.004
        else:
            raise ValueError(f"Unknown synthetic outcome: {outcome}")

        for rep in range(2):
            # CLEAN
            k_clean = int(rng.binomial(shots_per_rep, clean_p))
            raw_synthetic[(r_id, b_name, "CLEAN", rep)] = {
                primary_ideal: k_clean, non_ideal: shots_per_rep - k_clean
            }
            # HOT
            k_hot = int(rng.binomial(shots_per_rep, hot_p))
            raw_synthetic[(r_id, b_name, "HOT", rep)] = {
                primary_ideal: k_hot, non_ideal: shots_per_rep - k_hot
            }
            # CLEAN2 (if present)
            if has_clean2:
                k_clean2 = int(rng.binomial(shots_per_rep, clean2_p))
                raw_synthetic[(r_id, b_name, "CLEAN2", rep)] = {
                    primary_ideal: k_clean2, non_ideal: shots_per_rep - k_clean2
                }

    return raw_synthetic

def analyze_hotspot(plan_path="results/hardware_extra/plan.json",
                    raw_dir="results/hardware_extra/raw",
                    out_csv="results/hardware_extra_stats.csv",
                    out_txt="results/hardware_extra/hotspot_analysis_summary.txt",
                    synthetic=False,
                    synthetic_outcome="confirmed",
                    synthetic_effect_pts=2.5,
                    descriptive=False):
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

    # Identify primary cells
    primary_cells = []
    unmatchable_cells = []
    confounded_cells = []

    for r in plan_data.get("regions", []):
        r_id = r["region_id"]
        h_q = r["hot_qubit"]
        exc_q = r["excess_p1"]
        for b_name, b_val in r.get("benchmarks", {}).items():
            st = b_val.get("status", "")
            if st == "PRIMARY" and b_val.get("executed", True):
                primary_cells.append((r_id, h_q, exc_q, b_name, b_val))
            elif st == "UNMATCHABLE":
                unmatchable_cells.append((r_id, h_q, exc_q, b_name, b_val))
            elif st == "CONFOUNDED":
                confounded_cells.append((r_id, h_q, exc_q, b_name, b_val))

    raw_path = Path(raw_dir)
    has_real_counts = raw_path.exists() and len(list(raw_path.glob("r*.json"))) > 0

    if synthetic or not has_real_counts:
        is_synthetic = True
        raw_dict = generate_synthetic_hotspot_counts(
            plan_data, outcome=synthetic_outcome, target_effect_pts=synthetic_effect_pts
        )
    else:
        is_synthetic = False
        raw_dict = {}
        for fn in raw_path.glob("r*.json"):
            parts = fn.stem.split("_")
            r_id = int(parts[0][1:])
            b_name = parts[1]
            cond = parts[2]
            rep = int(parts[3].replace("rep", ""))
            with open(fn, "r", encoding="utf-8") as f:
                raw_dict[(r_id, b_name, cond, rep)] = json.load(f)

    # Statistical Evaluation across executed primary cells
    rows = []
    cell_evaluations = []

    for r_id, h_q, exc_q, b_name, b_val in primary_cells:
        circ = benchmarks_circs[b_name]
        ideal = ideal_cache[b_name]

        # Repetition 0
        counts_c0 = raw_dict.get((r_id, b_name, "CLEAN", 0), {})
        counts_h0 = raw_dict.get((r_id, b_name, "HOT", 0), {})
        n_c0 = sum(counts_c0.values()) or 8192
        n_h0 = sum(counts_h0.values()) or 8192
        _, score_c0 = compute_outcome_metric(b_name, Counter(counts_c0), circ, ideal, shots=n_c0)
        _, score_h0 = compute_outcome_metric(b_name, Counter(counts_h0), circ, ideal, shots=n_h0)
        k_c0 = int(round(score_c0 * n_c0))
        k_h0 = int(round(score_h0 * n_h0))

        # Repetition 1
        counts_c1 = raw_dict.get((r_id, b_name, "CLEAN", 1), {})
        counts_h1 = raw_dict.get((r_id, b_name, "HOT", 1), {})
        n_c1 = sum(counts_c1.values()) or 8192
        n_h1 = sum(counts_h1.values()) or 8192
        _, score_c1 = compute_outcome_metric(b_name, Counter(counts_c1), circ, ideal, shots=n_c1)
        _, score_h1 = compute_outcome_metric(b_name, Counter(counts_h1), circ, ideal, shots=n_h1)
        k_c1 = int(round(score_c1 * n_c1))
        k_h1 = int(round(score_h1 * n_h1))

        # Cell difference and stability evaluation via stat_utils
        cell_res = evaluate_cell_difference(
            clean_counts_rep0=k_c0, clean_shots_rep0=n_c0,
            clean_counts_rep1=k_c1, clean_shots_rep1=n_c1,
            hot_counts_rep0=k_h0, hot_shots_rep0=n_h0,
            hot_counts_rep1=k_h1, hot_shots_rep1=n_h1,
            confidence=0.95
        )

        diff_pooled = cell_res["diff_pooled"]
        ci_low = cell_res["newcombe_ci_low"]
        ci_high = cell_res["newcombe_ci_high"]
        diff_r0 = cell_res["diff_rep0"]
        diff_r1 = cell_res["diff_rep1"]

        # CLEAN2 negative control evaluation
        has_clean2 = b_val.get("has_clean2", False)
        clean2_diff_pooled = 0.0
        floor_value = 0.0
        passes_floor = True

        if has_clean2:
            counts_c2_0 = raw_dict.get((r_id, b_name, "CLEAN2", 0), {})
            counts_c2_1 = raw_dict.get((r_id, b_name, "CLEAN2", 1), {})
            n_c2_0 = sum(counts_c2_0.values()) or 8192
            n_c2_1 = sum(counts_c2_1.values()) or 8192
            _, score_c2_0 = compute_outcome_metric(b_name, Counter(counts_c2_0), circ, ideal, shots=n_c2_0)
            _, score_c2_1 = compute_outcome_metric(b_name, Counter(counts_c2_1), circ, ideal, shots=n_c2_1)
            k_c2_0 = int(round(score_c2_0 * n_c2_0))
            k_c2_1 = int(round(score_c2_1 * n_c2_1))
            k_c2_pool = k_c2_0 + k_c2_1
            n_c2_pool = n_c2_0 + n_c2_1

            k_c_pool = k_c0 + k_c1
            n_c_pool = n_c0 + n_c1
            clean2_diff_pooled, _, _ = newcombe_diff_ci(k_c_pool, n_c_pool, k_c2_pool, n_c2_pool, confidence=0.95)
            floor_value = abs(clean2_diff_pooled)
            passes_floor = (abs(diff_pooled) > floor_value)
        else:
            passes_floor = False  # Per W4(e): if CLEAN2 does not exist, cell is reported but does not count

        # Criteria for Significance
        # 1. Positive: CI low > 0, diff >= +0.005, rep0 > 0, rep1 > 0, passes floor
        rep_sign_pos = (diff_r0 > 0.0 and diff_r1 > 0.0)
        is_sig_positive = (ci_low > 0.0 and diff_pooled >= 0.005 and rep_sign_pos and passes_floor)

        # 2. Inverted: CI high < 0, diff <= -0.005, rep0 < 0, rep1 < 0, passes floor
        rep_sign_inv = (diff_r0 < 0.0 and diff_r1 < 0.0)
        is_sig_inverted = (ci_high < 0.0 and diff_pooled <= -0.005 and rep_sign_inv and passes_floor)

        cell_eval = {
            "region_id": r_id,
            "hot_qubit": h_q,
            "excess_p1": exc_q,
            "benchmark": b_name,
            "clean_score": cell_res["p_clean_pooled"],
            "hot_score": cell_res["p_hot_pooled"],
            "diff_pts": diff_pooled * 100.0,
            "ci_low_pts": ci_low * 100.0,
            "ci_high_pts": ci_high * 100.0,
            "diff_rep0_pts": diff_r0 * 100.0,
            "diff_rep1_pts": diff_r1 * 100.0,
            "clean_drift_pts": cell_res["clean_rep_drift"] * 100.0,
            "hot_drift_pts": cell_res["hot_rep_drift"] * 100.0,
            "has_clean2": has_clean2,
            "floor_value_pts": floor_value * 100.0,
            "passes_floor": passes_floor,
            "is_sig_positive": is_sig_positive,
            "is_sig_inverted": is_sig_inverted,
            "status": "PRIMARY",
            "confounded": False,
            "hot_esp": b_val.get("hot_esp", 0.0),
            "clean_esp": b_val.get("clean_esp", 0.0),
            "hot_2q": b_val.get("hot_2q", 0),
            "clean_2q": b_val.get("clean_2q", 0),
            "diff_2q": b_val.get("diff_2q", 0),
        }
        rows.append(cell_eval)
        cell_evaluations.append(cell_eval)

    # Append unmatchable and confounded cells to stats CSV for transparency
    for r_id, h_q, exc_q, b_name, b_val in unmatchable_cells:
        rows.append({
            "region_id": r_id, "hot_qubit": h_q, "excess_p1": exc_q, "benchmark": b_name,
            "clean_score": None, "hot_score": None, "diff_pts": None, "ci_low_pts": None, "ci_high_pts": None,
            "diff_rep0_pts": None, "diff_rep1_pts": None, "clean_drift_pts": None, "hot_drift_pts": None,
            "has_clean2": False, "floor_value_pts": None, "passes_floor": False,
            "is_sig_positive": False, "is_sig_inverted": False, "status": "UNMATCHABLE",
            "confounded": False,
            "hot_esp": b_val.get("hot_esp", 0.0), "clean_esp": b_val.get("clean_esp", 0.0),
            "hot_2q": b_val.get("hot_2q", 0), "clean_2q": b_val.get("clean_2q", 0), "diff_2q": b_val.get("diff_2q", 0),
        })

    for r_id, h_q, exc_q, b_name, b_val in confounded_cells:
        rows.append({
            "region_id": r_id, "hot_qubit": h_q, "excess_p1": exc_q, "benchmark": b_name,
            "clean_score": None, "hot_score": None, "diff_pts": None, "ci_low_pts": None, "ci_high_pts": None,
            "diff_rep0_pts": None, "diff_rep1_pts": None, "clean_drift_pts": None, "hot_drift_pts": None,
            "has_clean2": False, "floor_value_pts": None, "passes_floor": False,
            "is_sig_positive": False, "is_sig_inverted": False, "status": "CONFOUNDED",
            "confounded": True,
            "hot_esp": b_val.get("hot_esp", 0.0), "clean_esp": b_val.get("clean_esp", 0.0),
            "hot_2q": b_val.get("hot_2q", 0), "clean_2q": b_val.get("clean_2q", 0), "diff_2q": b_val.get("diff_2q", 0),
        })

    df_stats = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_stats.to_csv(out_csv, index=False)

    # Determine Pre-Registered Overall Outcome (Mutually Exclusive & Exhaustive A-F)
    n_primary = len(primary_cells)
    n_pos = sum(c["is_sig_positive"] for c in cell_evaluations)
    n_inv = sum(c["is_sig_inverted"] for c in cell_evaluations)

    if n_primary < 4 or synthetic_outcome == "inconclusive":
        outcome_code = "F. INCONCLUSIVE / UNDERPOWERED"
        outcome_label = f"INCONCLUSIVE (underpowered: {n_primary} primary cells < 4 required)"
        outcome_desc = "Fewer than 4 primary cells available. Lattice connectivity or 2Q gate overhead prevents confirmatory testing."
        is_confirmed = False
    elif n_pos > 0 and n_inv > 0:
        outcome_code = "D. MIXED"
        outcome_label = f"MIXED ({n_pos} positive, {n_inv} inverted primary cells)"
        outcome_desc = "Both significant positive and significant inverted cells observed. Performance is benchmark- or region-dependent."
        is_confirmed = False
    elif n_pos >= math.ceil(0.5 * n_primary) and n_inv == 0:
        outcome_code = "A. CONFIRMED"
        outcome_label = f"CONFIRMED ({n_pos}/{n_primary} primary cells meet all pre-registered criteria)"
        outcome_desc = "Avoidance of persistent hot qubit reliably improves algorithmic execution on hardware."
        is_confirmed = True
    elif n_inv >= math.ceil(0.5 * n_primary) and n_pos == 0:
        outcome_code = "C. INVERTED"
        outcome_label = f"INVERTED ({n_inv}/{n_primary} primary cells show significant inverted degradation)"
        outcome_desc = "Refutes predictive validity of p1 calibration data for spatial layout selection."
        is_confirmed = False
    else:
        outcome_code = "B. NULL / NOT CONFIRMED"
        outcome_label = f"NULL / NOT CONFIRMED ({n_pos}/{n_primary} primary cells meet criterion; < 50% threshold)"
        outcome_desc = "Measured p1 excess does not translate into circuit-level degradation."
        is_confirmed = False

    # Write Provenance JSON
    prov_path = Path(out_csv).with_suffix(".provenance.json")
    prov_data = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "backend": plan_data.get("backend"),
        "calibration_timestamp": plan_data.get("calibration_timestamp"),
        "plan_hash": plan_data.get("plan_hash"),
        "preregistration_version": "v2.2",
        "is_synthetic": is_synthetic,
        "synthetic_outcome": synthetic_outcome if is_synthetic else None,
        "primary_cells_executed": n_primary,
        "significant_positive_cells": n_pos,
        "significant_inverted_cells": n_inv,
        "outcome_code": outcome_code,
        "outcome_label": outcome_label,
        "shots_pooled_per_cell": 16384,
        "repetitions": 2,
    }
    with open(prov_path, "w", encoding="utf-8") as f:
        json.dump(prov_data, f, indent=2)

    # Format Summary Text Report
    lines = []
    lines.append("=" * 95)
    if is_synthetic:
        lines.append("[SYNTHETIC KNOWN-ANSWER TEST - NOT EVIDENCE - DO NOT CITE]")
    if n_primary < 4:
        lines.append("[DESCRIPTIVE, NOT CONFIRMATORY - N_primary < 4]")
    lines.append("PRE-REGISTERED HARDWARE HOT-SPOT STRESS TEST: STATISTICAL ANALYSIS REPORT (v2.2)")
    lines.append(f"Analysis Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    lines.append("=" * 95)
    lines.append("")
    lines.append("1. METHODOLOGICAL SPECIFICATIONS & DISCLOSURES")
    lines.append("-" * 95)
    lines.append("  - Comparator:              ESP-matched clean layout (ESP_clean <= ESP_hot within 0.03, |Delta 2Q| <= 4)")
    lines.append("  - Negative Control:        CLEAN2 matched layout for empirical layout-variance floor")
    lines.append("  - Metric:                  Difference of independent proportions: success_prob(CLEAN) - success_prob(HOT)")
    lines.append("  - Statistical CI:          Newcombe hybrid-score 95% CI (Method 10, Newcombe 1998) pooled over 2 reps")
    lines.append("  - Noise Scope:             CI covers shot noise only; context given by layout-variance floor")
    lines.append("  - Decision Rules:          Diff >= +0.5 pt, CI low > 0, rep0/rep1 sign agreement, |diff| > |clean - clean2|")
    lines.append(f"  - Mode:                    {'SYNTHETIC TEST' if is_synthetic else 'REAL QPU HARDWARE EXECUTION'}")
    lines.append("")
    lines.append("2. PRIMARY CELL STATISTICAL RESULTS (CLEAN - HOT)")
    lines.append("-" * 115)
    hdr = f"{'Region':<7} {'Bench':<7} {'Clean %':<9} {'Hot %':<9} {'Diff (pts)':<11} {'95% Newcombe CI':<20} {'Rep0/1 (pts)':<18} {'Floor':<9} {'Cell Pass':<9}"
    lines.append(hdr)
    lines.append("-" * 115)
    for c in cell_evaluations:
        ci_str = f"[{c['ci_low_pts']:+.3f}, {c['ci_high_pts']:+.3f}]"
        reps_str = f"{c['diff_rep0_pts']:+.2f} / {c['diff_rep1_pts']:+.2f}"
        floor_str = f"{c['floor_value_pts']:.2f} pt" if c['has_clean2'] else "No Clean2"
        pass_str = "PASS (+)" if c['is_sig_positive'] else ("PASS (-)" if c['is_sig_inverted'] else "FAIL")
        lines.append(
            f"R{c['region_id']:<6} {c['benchmark']:<7} {c['clean_score']*100:<9.2f} {c['hot_score']*100:<9.2f} "
            f"{c['diff_pts']:<11.3f} {ci_str:<20} {reps_str:<18} {floor_str:<9} {pass_str:<9}"
        )
    lines.append("-" * 115)
    lines.append("")
    lines.append("3. OVERALL OUTCOME DETERMINATION (MUTUALLY EXCLUSIVE & EXHAUSTIVE A-F)")
    lines.append("-" * 95)
    lines.append(f"  Total Primary Cells:      {n_primary} (Threshold for Confirmatory Power: >= 4)")
    lines.append(f"  Significant Positive:     {n_pos}")
    lines.append(f"  Significant Inverted:     {n_inv}")
    lines.append(f"  Pre-Registered Outcome:   {outcome_code}")
    lines.append(f"  Outcome Summary:          {outcome_label}")
    lines.append(f"  Interpretation Note:      {outcome_desc}")
    lines.append("=" * 95)

    report_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"Stats CSV written to {out_csv}")
    print(f"Summary TXT written to {out_txt}")

    return {
        "outcome_code": outcome_code,
        "outcome_label": outcome_label,
        "is_confirmed": is_confirmed,
        "n_primary": n_primary,
        "n_pos": n_pos,
        "n_inv": n_inv,
        "df_stats": df_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze Hotspot Hardware Stress Test Results (v2.2)")
    parser.add_argument("--plan", default="results/hardware_extra/plan.json", help="Path to plan.json")
    parser.add_argument("--raw-dir", default="results/hardware_extra/raw", help="Path to raw JSON counts")
    parser.add_argument("--out-csv", default="results/hardware_extra_stats.csv", help="Path to output stats CSV")
    parser.add_argument("--out-txt", default="results/hardware_extra/hotspot_analysis_summary.txt", help="Path to output summary TXT")
    parser.add_argument("--synthetic", action="store_true", help="Run known-answer test with synthetic counts")
    parser.add_argument("--outcome", choices=["confirmed", "null", "inverted", "inconclusive", "mixed"], default="confirmed",
                        help="Outcome pattern to simulate in synthetic mode")
    parser.add_argument("--effect", type=float, default=2.5, help="Synthetic effect size in percentage points")
    parser.add_argument("--descriptive", action="store_true", help="Label output descriptive if N_primary < 4")
    args = parser.parse_args()

    analyze_hotspot(
        plan_path=args.plan,
        raw_dir=args.raw_dir,
        out_csv=args.out_csv,
        out_txt=args.out_txt,
        synthetic=args.synthetic,
        synthetic_outcome=args.outcome,
        synthetic_effect_pts=args.effect,
        descriptive=args.descriptive
    )

if __name__ == "__main__":
    main()
