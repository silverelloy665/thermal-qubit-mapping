#!/usr/bin/env python3
"""
scripts/analyze_p1_v2.py

Offline analysis of enhanced p1v2 JSON characterization files (Checkpoint U).
Evaluates:
- excess_v2 using calibrated prob_meas1_prep0 (with fallback to symmetric ro if unexposed)
- persistence rule: HOT only if excess_v2 > 0.05 in BOTH A repeats AND isolated Block B
- flags: 'crosstalk-suspect' (hot in A, cold in B) and 'transient' (disagrees with 9 Oct file)
- 2-SE statistical significance margins at 0.01, 0.02, 0.05
- Outputs results/analysis_v2/p1v2_summary.txt
"""

import os
import sys
import glob
import json
import math
import argparse
from pathlib import Path

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))

def classify_qubit_persistence(exc_a1, exc_a2, exc_b, se_a1=0.01, se_a2=0.01, se_b=0.01, hist_exc=0.0):
    """
    Evaluates persistence and flags for a single qubit:
    - Persistent hot: excess > 0.05 in both A repeats and isolated B
    - Crosstalk suspect: hot in A, cold in B (excess_b <= 0.05)
    - Transient: disagrees with historical committed 9 Oct measurement
    """
    is_a_hot = (exc_a1 > 0.05 and exc_a2 > 0.05)
    is_b_hot = (exc_b is not None and exc_b > 0.05)
    is_persistent = is_a_hot and is_b_hot
    is_crosstalk = (is_a_hot and (exc_b is not None and exc_b <= 0.05))
    was_old_hot = (hist_exc > 0.05)
    is_transient = (was_old_hot != is_persistent)
    return {
        "is_persistent_hot": is_persistent,
        "is_crosstalk_suspect": is_crosstalk,
        "is_transient": is_transient,
    }

def analyze_p1_v2(p1v2_file=None, base_p1_file="results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json", out_file="results/analysis_v2/p1v2_summary.txt"):
    if p1v2_file is None:
        files = sorted(glob.glob("results/hardware_v2/p1v2_*.json"))
        if not files:
            raise FileNotFoundError("No p1v2 characterization files found in results/hardware_v2/.")
        p1v2_file = files[-1]

    with open(p1v2_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load 9 Oct base file for comparison
    with open(base_p1_file, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    base_p1 = base_data.get("p1_estimate", {})
    base_ro = base_data.get("target_readout_error", {})
    old_hot_set = set()
    for q_str, p in base_p1.items():
        if (p - base_ro.get(q_str, 0.0)) > 0.05:
            old_hot_set.add(int(q_str))

    backend = data.get("backend", "unknown")
    cal_ts = data.get("calibration_timestamp", "unknown")
    exec_ts = data.get("execution_timestamp", "unknown")
    shots = data.get("shots_per_pub", 4096)
    asym_exposed = data.get("calibrated_asymmetry_exposed", False)

    cal_p10 = data.get("target_calibrated_prob_meas1_prep0", {})
    cal_ro = data.get("target_calibrated_ro", {})
    measurements = data.get("measurements", {})

    total_qubits = len(measurements)
    fallback_used = not asym_exposed

    results_table = []
    persistent_hot = []
    crosstalk_suspect = []
    transient = []

    count_gt_001 = 0
    survive_2se_001 = 0
    count_gt_002 = 0
    survive_2se_002 = 0
    count_gt_005 = 0
    survive_2se_005 = 0

    for q_str, blocks in measurements.items():
        q = int(q_str)
        
        # Determine baseline subtraction (calibrated prob_meas1_prep0 or ro fallback)
        sub_val = cal_p10.get(q_str)
        if sub_val is None:
            sub_val = cal_ro.get(q_str, 0.0)
            q_fallback = True
        else:
            q_fallback = False

        # Block A: all qubits
        a_data = blocks.get("A", {})
        p10_a0 = a_data.get("rep0", {}).get("P10", 0.0)
        p10_a1 = a_data.get("rep1", {}).get("P10", 0.0)
        exc_a0 = max(0.0, p10_a0 - sub_val)
        exc_a1 = max(0.0, p10_a1 - sub_val)
        p10_a_mean = (p10_a0 + p10_a1) / 2.0
        exc_a_mean = max(0.0, p10_a_mean - sub_val)

        # Block B: isolated
        b_data = blocks.get("B", {})
        b_p10s = []
        for b_rep_k, b_rep_v in b_data.items():
            if "P10" in b_rep_v:
                b_p10s.append(b_rep_v["P10"])
        
        if b_p10s:
            p10_b_mean = sum(b_p10s) / len(b_p10s)
            exc_b_mean = max(0.0, p10_b_mean - sub_val)
            measured_in_b = True
        else:
            p10_b_mean = None
            exc_b_mean = None
            measured_in_b = False

        # Block C: single
        c_data = blocks.get("C", {})
        if c_data and "rep0" in c_data:
            p10_c = c_data["rep0"].get("P10", None)
            exc_c = max(0.0, p10_c - sub_val) if p10_c is not None else None
        else:
            p10_c = None
            exc_c = None

        # 2-SE margin calculation on Block A mean
        se = math.sqrt(p10_a_mean * (1.0 - p10_a_mean) / (shots * 2)) # 2 reps pooled
        if exc_a_mean > 0.01:
            count_gt_001 += 1
            if (exc_a_mean - 2 * se) > 0.01:
                survive_2se_001 += 1
        if exc_a_mean > 0.02:
            count_gt_002 += 1
            if (exc_a_mean - 2 * se) > 0.02:
                survive_2se_002 += 1
        if exc_a_mean > 0.05:
            count_gt_005 += 1
            if (exc_a_mean - 2 * se) > 0.05:
                survive_2se_005 += 1

        # Persistence Rule:
        # A qubit is HOT only if excess_v2 > 0.05 in BOTH A repeats AND in the isolated block B
        is_a_hot = (exc_a0 > 0.05 and exc_a1 > 0.05)
        is_b_hot = (exc_b_mean is not None and exc_b_mean > 0.05)
        
        is_persistent = is_a_hot and is_b_hot
        if is_persistent:
            persistent_hot.append(q)

        # Crosstalk suspect: hot in A, but cold in B
        if is_a_hot and (exc_b_mean is not None and exc_b_mean <= 0.05):
            crosstalk_suspect.append(q)

        # Transient: disagrees with 9 Oct file (was hot on 9 Oct but not persistent hot now, or vice versa)
        was_old_hot = (q in old_hot_set)
        if was_old_hot != is_persistent:
            transient.append(q)

        results_table.append({
            "qubit": q,
            "sub_val": sub_val,
            "fallback": q_fallback,
            "p10_a0": p10_a0,
            "p10_a1": p10_a1,
            "exc_a0": exc_a0,
            "exc_a1": exc_a1,
            "exc_a_mean": exc_a_mean,
            "exc_b_mean": exc_b_mean,
            "exc_c": exc_c,
            "is_persistent": is_persistent,
            "is_crosstalk_suspect": (is_a_hot and exc_b_mean is not None and exc_b_mean <= 0.05),
            "is_transient": (was_old_hot != is_persistent)
        })

    # Sort table by Block A mean excess
    results_table.sort(key=lambda x: x["exc_a_mean"], reverse=True)

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 85,
        "ENHANCED P1 (v2) CHARACTERIZATION AUDIT & PERSISTENCE REPORT",
        "=" * 85,
        f"Characterization File:    {p1v2_file}",
        f"Backend:                  {backend} ({total_qubits} qubits audited)",
        f"Calibration Timestamp:    {cal_ts}",
        f"Execution Timestamp:      {exec_ts}",
        f"Shots per PUB:            {shots} (pooled A shots: {shots * 2})",
        f"Calibrated Asymmetry:     {'EXPOSED' if asym_exposed else 'UNEXPOSED (fell back to symmetric readout error ro)'}",
        "",
        "1. PERSISTENCE & CLASSIFICATION SUMMARY",
        "-" * 85,
        f"- PERSISTENT HOT QUBITS:  {len(persistent_hot)} qubits {persistent_hot}",
        "  (Condition: excess_v2 > 0.05 in BOTH Block A repetitions AND Block B isolated)",
        f"- CROSSTALK-SUSPECT:      {len(crosstalk_suspect)} qubits {crosstalk_suspect}",
        "  (Condition: Block A repeats > 0.05, but Block B isolated <= 0.05)",
        f"- TRANSIENT DISAGREEMENTS: {len(transient)} qubits {transient}",
        "  (Condition: status disagrees with 9 Oct baseline classification)",
        "",
        "2. STATISTICAL SIGNIFICANCE (POOLED BLOCK A)",
        "-" * 85,
        f"- Excess > 0.01: {count_gt_001} qubits total | {survive_2se_001} clear 2-SE margin",
        f"- Excess > 0.02: {count_gt_002} qubits total | {survive_2se_002} clear 2-SE margin",
        f"- Excess > 0.05: {count_gt_005} qubits total | {survive_2se_005} clear 2-SE margin",
        "",
        "3. DETAILED TOP-15 QUBIT AUDIT TABLE",
        "-" * 85,
        f"{'Qubit':<6} {'BaseSub':<9} {'Type':<8} {'Exc A0':<9} {'Exc A1':<9} {'Exc B':<9} {'Exc C':<9} {'Status':<16}",
        "-" * 85
    ]

    for item in results_table[:15]:
        sub_type = "ro_fall" if item["fallback"] else "p10_cal"
        exc_b_str = f"{item['exc_b_mean']:.4f}" if item['exc_b_mean'] is not None else "N/A"
        exc_c_str = f"{item['exc_c']:.4f}" if item['exc_c'] is not None else "N/A"
        
        status_tags = []
        if item["is_persistent"]:
            status_tags.append("HOT_PERSISTENT")
        elif item["is_crosstalk_suspect"]:
            status_tags.append("CROSSTALK_SUSPECT")
        elif item["exc_a_mean"] > 0.05:
            status_tags.append("HOT_UNCONFIRMED")
        else:
            status_tags.append("CLEAN")

        if item["is_transient"]:
            status_tags.append("TRANSIENT")

        lines.append(
            f"{item['qubit']:<6} {item['sub_val']:<9.4f} {sub_type:<8} "
            f"{item['exc_a0']:<9.4f} {item['exc_a1']:<9.4f} {exc_b_str:<9} {exc_c_str:<9} {','.join(status_tags):<16}"
        )

    lines.append("=" * 85)

    report_text = "\n".join(lines) + "\n"
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"p1v2 analysis report written to {out_p}")
    return {
        "persistent_hot": persistent_hot,
        "crosstalk_suspect": crosstalk_suspect,
        "transient": transient,
        "counts": {
            "gt_001": count_gt_001,
            "survive_001": survive_2se_001,
            "gt_005": count_gt_005,
            "survive_005": survive_2se_005,
        }
    }

def main():
    parser = argparse.ArgumentParser(description="Analyze p1v2 JSON Characterization")
    parser.add_argument("--file", type=str, default=None, help="Path to p1v2 JSON file")
    args = parser.parse_args()

    analyze_p1_v2(args.file)

if __name__ == "__main__":
    main()

