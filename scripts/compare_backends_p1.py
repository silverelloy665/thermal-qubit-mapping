#!/usr/bin/env python3
"""
scripts/compare_backends_p1.py

Checkpoint Z (Z3): Descriptive cross-backend characterization comparison (IBM Marrakesh vs IBM Fez).
Loads Marrakesh characterization (p1v2 or committed 9 Oct baseline) and Fez survey (plus confirm if present).
Writes results/analysis_v2/backend_compare.txt.
"""

import os
import sys
import json
import glob
import math
import argparse
from pathlib import Path

def parse_marrakesh_data(m_path):
    p = Path(m_path)
    if not p.exists():
        raise FileNotFoundError(f"Marrakesh characterization file not found: {m_path}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if p1v2 format or legacy 9 Oct format
    is_v2 = "measurements" in data
    shots = int(data.get("shots_per_pub", 8192 if not is_v2 else 4096))
    cal_ts = data.get("calibration_timestamp", "2026-10-08T22:38:00+00:00")
    b_name = data.get("backend", "ibm_marrakesh")

    excess_dict = {}
    measured_ro_dict = {}
    calibrated_ro_dict = {}
    p01_gt_p10_count = 0
    total_asym_qubits = 0
    persistent_count = None

    if is_v2:
        cal_ro = data.get("target_calibrated_ro", {})
        cal_p10 = data.get("target_calibrated_prob_meas1_prep0", {})
        measurements = data.get("measurements", {})
        
        # Check persistence from isolated batches if present
        qubit_sets = data.get("qubit_sets", {})
        candidate_hot = qubit_sets.get("HOT", [])
        persistent_qubits = []

        for q_str, m_blocks in measurements.items():
            q = int(q_str)
            block_a = m_blocks.get("A", {})
            r0 = block_a.get("rep0", {})
            r1 = block_a.get("rep1", {})
            p10_0 = r0.get("P10", 0.0)
            p10_1 = r1.get("P10", 0.0)
            p01_0 = r0.get("P01", 0.0)
            p01_1 = r1.get("P01", 0.0)
            p10_avg = (p10_0 + p10_1) / 2.0
            p01_avg = (p01_0 + p01_1) / 2.0

            ro_cal = cal_ro.get(q_str, 0.0)
            base_p10 = cal_p10.get(q_str) if cal_p10.get(q_str) is not None else ro_cal
            exc_avg = max(0.0, p10_avg - base_p10)
            excess_dict[q] = {
                "excess": exc_avg,
                "p10_rep0": p10_0,
                "p10_rep1": p10_1,
                "base_p10": base_p10,
                "ro_cal": ro_cal,
            }
            measured_ro_dict[q] = (p10_avg + p01_avg) / 2.0
            calibrated_ro_dict[q] = ro_cal

            total_asym_qubits += 1
            if p01_avg > p10_avg:
                p01_gt_p10_count += 1

            # Persistence: excess > 0.05 in both rep0 and rep1 and in isolated block B
            if q in candidate_hot and (p10_0 - base_p10) > 0.05 and (p10_1 - base_p10) > 0.05 and ro_cal <= 0.20:
                block_b = m_blocks.get("B", {})
                b_pass = False
                for b_sub, b_vals in block_b.items():
                    if b_vals.get("P10", 0.0) - base_p10 > 0.05:
                        b_pass = True
                        break
                if b_pass:
                    persistent_qubits.append(q)
        persistent_count = len(persistent_qubits)
    else:
        # Legacy 9 Oct snapshot
        p1_est = data.get("p1_estimate", {})
        ro_err = data.get("target_readout_error", {})
        asym = data.get("readout_asymmetry", {})
        for q_str, p1_val in p1_est.items():
            q = int(q_str)
            ro_val = ro_err.get(q_str, 0.0)
            exc = max(0.0, p1_val - ro_val)
            excess_dict[q] = {
                "excess": exc,
                "p10_rep0": p1_val,
                "p10_rep1": p1_val,
                "base_p10": ro_val,
                "ro_cal": ro_val,
            }
            measured_ro_dict[q] = ro_val  # not separately isolated in single snapshot
            calibrated_ro_dict[q] = ro_val
            total_asym_qubits += 1
            asym_val = asym.get(q_str, 0.0)
            if asym_val > 0.0:  # P(0|1) > P(1|0)
                p01_gt_p10_count += 1
        persistent_count = None  # Single snapshot baseline

    return {
        "backend": b_name,
        "is_v2": is_v2,
        "calibration_timestamp": cal_ts,
        "shots": shots,
        "excess_dict": excess_dict,
        "measured_ro_dict": measured_ro_dict,
        "calibrated_ro_dict": calibrated_ro_dict,
        "total_asym_qubits": total_asym_qubits,
        "p01_gt_p10_count": p01_gt_p10_count,
        "persistent_count": persistent_count,
    }

def parse_fez_data(survey_path, confirm_path=None):
    p_s = Path(survey_path)
    if not p_s.exists():
        raise FileNotFoundError(f"Fez survey file not found: {survey_path}")
    with open(p_s, "r", encoding="utf-8") as f:
        s_data = json.load(f)

    b_name = s_data.get("backend", "ibm_fez")
    cal_ts = s_data.get("calibration_timestamp", "unknown")
    shots = int(s_data.get("shots_per_pub", 4096))
    cal_ro = s_data.get("target_calibrated_ro", {})
    cal_p10 = s_data.get("target_calibrated_prob_meas1_prep0", {})
    measurements = s_data.get("measurements", {})

    excess_dict = {}
    measured_ro_dict = {}
    calibrated_ro_dict = {}
    p01_gt_p10_count = 0
    total_asym_qubits = 0

    for q_str, m_blocks in measurements.items():
        q = int(q_str)
        block_a = m_blocks.get("A", {})
        r0 = block_a.get("rep0", {})
        r1 = block_a.get("rep1", {})
        p10_0 = r0.get("P10", 0.0)
        p10_1 = r1.get("P10", 0.0)
        p01_0 = r0.get("P01", 0.0)
        p01_1 = r1.get("P01", 0.0)
        p10_avg = (p10_0 + p10_1) / 2.0
        p01_avg = (p01_0 + p01_1) / 2.0

        ro_cal = cal_ro.get(q_str, 0.0)
        base_p10 = cal_p10.get(q_str) if cal_p10.get(q_str) is not None else ro_cal
        exc_avg = max(0.0, p10_avg - base_p10)
        excess_dict[q] = {
            "excess": exc_avg,
            "p10_rep0": p10_0,
            "p10_rep1": p10_1,
            "base_p10": base_p10,
            "ro_cal": ro_cal,
        }
        measured_ro_dict[q] = (p10_avg + p01_avg) / 2.0
        calibrated_ro_dict[q] = ro_cal

        total_asym_qubits += 1
        if p01_avg > p10_avg:
            p01_gt_p10_count += 1

    persistent_count = None
    if confirm_path and Path(confirm_path).exists():
        with open(confirm_path, "r", encoding="utf-8") as f:
            c_data = json.load(f)
        persistent_count = c_data.get("persistent_count", 0)

    return {
        "backend": b_name,
        "calibration_timestamp": cal_ts,
        "shots": shots,
        "excess_dict": excess_dict,
        "measured_ro_dict": measured_ro_dict,
        "calibrated_ro_dict": calibrated_ro_dict,
        "total_asym_qubits": total_asym_qubits,
        "p01_gt_p10_count": p01_gt_p10_count,
        "persistent_count": persistent_count,
    }

def compute_metrics(parsed):
    excess_dict = parsed["excess_dict"]
    shots = parsed["shots"]

    cnt_01 = sum(1 for q, d in excess_dict.items() if d["excess"] > 0.01)
    cnt_02 = sum(1 for q, d in excess_dict.items() if d["excess"] > 0.02)
    cnt_05 = sum(1 for q, d in excess_dict.items() if d["excess"] > 0.05)

    # 2-SE threshold: excess > 2 * sqrt(p*(1-p)/N)
    cleared_2se = 0
    for q, d in excess_dict.items():
        p = d["base_p10"]
        p = max(0.0001, min(0.9999, p))
        se = math.sqrt(p * (1.0 - p) / shots)
        if d["excess"] > 2.0 * se:
            cleared_2se += 1

    # Median measured readout error vs calibrated readout error
    meas_vals = sorted(parsed["measured_ro_dict"].values())
    cal_vals = sorted(parsed["calibrated_ro_dict"].values())
    med_meas = meas_vals[len(meas_vals) // 2] if meas_vals else 0.0
    med_cal = cal_vals[len(cal_vals) // 2] if cal_vals else 0.0

    # Fraction where P(0|1) > P(1|0)
    frac_asym = (
        parsed["p01_gt_p10_count"] / parsed["total_asym_qubits"]
        if parsed["total_asym_qubits"] > 0
        else 0.0
    )

    return {
        "total_qubits": len(excess_dict),
        "cnt_01": cnt_01,
        "cnt_02": cnt_02,
        "cnt_05": cnt_05,
        "cleared_2se": cleared_2se,
        "med_meas_ro": med_meas,
        "med_cal_ro": med_cal,
        "frac_p01_gt_p10": frac_asym,
        "persistent_count": parsed["persistent_count"],
    }

def generate_comparison_report(marrakesh_path, fez_survey_path, fez_confirm_path=None, out_txt=None):
    parsed_m = parse_marrakesh_data(marrakesh_path)
    parsed_f = parse_fez_data(fez_survey_path, fez_confirm_path)

    met_m = compute_metrics(parsed_m)
    met_f = compute_metrics(parsed_f)

    lines = []
    lines.append("=" * 95)
    lines.append("DESCRIPTIVE CROSS-BACKEND COMPARISON ONLY -- NOT A HYPOTHESIS TEST")
    lines.append("DESCRIPTIVE, TWO DEVICES, ONE DAY EACH - NOT A HYPOTHESIS TEST")
    lines.append("=" * 95)
    lines.append("")
    lines.append("1. SOURCES & DEVICE METADATA")
    lines.append("-" * 95)
    lines.append(f"  Marrakesh Source:         {marrakesh_path}")
    lines.append(f"  Marrakesh Backend:        {parsed_m['backend']} ({parsed_m['calibration_timestamp']})")
    if not parsed_m["is_v2"]:
        lines.append("  [METHODOLOGICAL NOTE]     Marrakesh source is single snapshot (9 Oct 2026):")
        lines.append("                            snapshot vs repeated, not like for like.")
    lines.append(f"  Fez Survey Source:        {fez_survey_path}")
    lines.append(f"  Fez Backend:              {parsed_f['backend']} ({parsed_f['calibration_timestamp']})")
    if fez_confirm_path:
        lines.append(f"  Fez Confirm Source:       {fez_confirm_path}")
    else:
        lines.append("  Fez Confirm Source:       None (survey stage only; confirmation pending)")
    lines.append("")

    lines.append("2. PER-BACKEND ANOMALY METRICS TABLE")
    lines.append("-" * 95)
    hdr = f"{'Metric':<45} | {'ibm_marrakesh':<20} | {'ibm_fez':<20}"
    lines.append(hdr)
    lines.append("-" * 95)
    lines.append(f"{'Total Evaluated Qubits':<45} | {met_m['total_qubits']:<20} | {met_f['total_qubits']:<20}")
    lines.append(f"{'Qubits with Excess P(1|0) > 0.01':<45} | {met_m['cnt_01']:<20} | {met_f['cnt_01']:<20}")
    lines.append(f"{'Qubits with Excess P(1|0) > 0.02':<45} | {met_m['cnt_02']:<20} | {met_f['cnt_02']:<20}")
    lines.append(f"{'Qubits with Excess P(1|0) > 0.05':<45} | {met_m['cnt_05']:<20} | {met_f['cnt_05']:<20}")
    lines.append(f"{'Qubits Clearing 2-SE Binomial Margin':<45} | {met_m['cleared_2se']:<20} | {met_f['cleared_2se']:<20}")

    pers_m_str = (
        str(met_m["persistent_count"])
        if met_m["persistent_count"] is not None
        else "not measured (single snapshot)"
    )
    pers_f_str = (
        str(met_f["persistent_count"])
        if met_f["persistent_count"] is not None
        else "not measured (confirm pending)"
    )
    lines.append(f"{'Persistent Hot Qubits (v2.4 Rule)':<45} | {pers_m_str:<20} | {pers_f_str:<20}")
    lines.append(f"{'Median Measured Readout Error':<45} | {met_m['med_meas_ro']:<20.4f} | {met_f['med_meas_ro']:<20.4f}")
    lines.append(f"{'Median Calibrated Readout Error':<45} | {met_m['med_cal_ro']:<20.4f} | {met_f['med_cal_ro']:<20.4f}")
    lines.append(f"{'Fraction Qubits with P(0|1) > P(1|0)':<45} | {met_m['frac_p01_gt_p10']:<20.2%} | {met_f['frac_p01_gt_p10']:<20.2%}")
    lines.append("-" * 95)
    lines.append("")
    lines.append("3. SCIENTIFIC INTERPRETATION BOUNDARIES")
    lines.append("-" * 95)
    lines.append("  - This cross-backend comparison is strictly descriptive and exploratory.")
    lines.append("  - No hypothesis regarding cryostat thermal gradients or chip layout is evaluated.")
    lines.append("  - No algorithmic hot-spot stress test is executed on ibm_fez.")
    lines.append("=" * 95)

    report_text = "\n".join(lines) + "\n"
    if out_txt:
        p_out = Path(out_txt)
        p_out.parent.mkdir(parents=True, exist_ok=True)
        with open(p_out, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"[SUCCESS] Backend comparison report written to {out_txt}")
    return report_text

def main():
    parser = argparse.ArgumentParser(description="Descriptive Cross-Backend P1 Comparison")
    parser.add_argument("--marrakesh", type=str, default=None, help="Marrakesh p1 characterization path")
    parser.add_argument("--fez-survey", type=str, default=None, help="Fez survey JSON path")
    parser.add_argument("--fez-confirm", type=str, default=None, help="Fez confirmation JSON path (optional)")
    parser.add_argument("--out-txt", type=str, default="results/analysis_v2/backend_compare.txt", help="Output text report path")
    args = parser.parse_args()

    m_path = args.marrakesh
    if not m_path:
        v2_m = sorted(glob.glob("results/hardware_v2/p1v2_ibm_marrakesh_*.json"))
        if v2_m:
            m_path = v2_m[-1]
        else:
            m_path = "results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json"

    f_survey = args.fez_survey
    if not f_survey:
        f_surveys = sorted(glob.glob("results/hardware_v2/p1v2_survey_ibm_fez_*.json"))
        if f_surveys:
            f_survey = f_surveys[-1]
        else:
            raise FileNotFoundError("No Fez survey file found. Specify with --fez-survey.")

    f_confirm = args.fez_confirm
    if not f_confirm:
        f_confirms = sorted(glob.glob("results/hardware_v2/p1v2_confirm_ibm_fez_*.json"))
        if f_confirms:
            f_confirm = f_confirms[-1]

    generate_comparison_report(m_path, f_survey, f_confirm, args.out_txt)

if __name__ == "__main__":
    main()

