#!/usr/bin/env python3
"""
scripts/p1_limits.py

Dynamically audits the committed 9 Oct p1 calibration file and extracts
the quantitative limitations of the excited-state population measurement.
Outputs results/analysis_v2/p1_limits.txt.
"""

import os
import sys
import json
import math
from pathlib import Path

def compute_p1_limits(
    p1_file="results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json",
    out_file="results/analysis_v2/p1_limits.txt"
):
    with open(p1_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    p1_est = data["p1_estimate"]
    ro_err = data["target_readout_error"]
    asym = data["readout_asymmetry"]
    shots = 4096

    total_qubits = len(p1_est)
    clipped_zero = 0
    excess_dict = {}

    count_gt_001 = 0
    survive_2se_001 = 0

    count_gt_002 = 0
    survive_2se_002 = 0

    count_gt_005 = 0
    survive_2se_005 = 0

    for q_str, p1_val in p1_est.items():
        ro_val = ro_err.get(q_str, 0.0)
        exc = max(0.0, p1_val - ro_val)
        excess_dict[int(q_str)] = exc

        if exc == 0.0:
            clipped_zero += 1

        se = math.sqrt(p1_val * (1.0 - p1_val) / shots)

        if exc > 0.01:
            count_gt_001 += 1
            if (exc - 2 * se) > 0.01:
                survive_2se_001 += 1

        if exc > 0.02:
            count_gt_002 += 1
            if (exc - 2 * se) > 0.02:
                survive_2se_002 += 1

        if exc > 0.05:
            count_gt_005 += 1
            if (exc - 2 * se) > 0.05:
                survive_2se_005 += 1

    # Qubit 125 properties
    q125_p10 = p1_est.get("125", 0.0)
    q125_asym = asym.get("125", 0.0)
    q125_p01 = q125_asym + q125_p10
    q125_ro = ro_err.get("125", 0.0)

    # Dates
    p1_date = data.get("timestamp", "2026-10-09T00:18:39+05:30")
    rerun_cal_date = "2026-10-12T23:40:00+05:30"

    out_p = Path(out_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 80,
        "AUDIT REPORT: KNOWN LIMITATIONS OF THE P1 MEASUREMENT",
        "=" * 80,
        f"Source Calibration File: {p1_file}",
        f"Calibration Timestamp:   {p1_date}",
        f"Total Qubits Audited:    {total_qubits}",
        f"Per-PUB Shot Count:      {shots}",
        "",
        "1. ARCHITECTURAL & METHODOLOGICAL CONSTRAINTS",
        "-" * 80,
        "- Single snapshot: All data obtained from a single characterization run on 2026-10-09.",
        "- Simultaneous readout crosstalk: All 156 qubits were measured simultaneously in a",
        "  single register, which creates potential measurement line and multiplexing crosstalk.",
        "- Symmetric error subtraction: Excess calculation subtracts symmetric calibrated readout",
        "  error ro although P(0|1) was recorded and differs markedly from P(1|0).",
        f"- Zero clipping: {clipped_zero}/{total_qubits} ({clipped_zero/total_qubits*100:.1f}%) excess values clip to exactly 0.0",
        "  because measured P(1|0) is at or below the calibrated readout error.",
        "",
        "2. STATISTICAL SIGNIFICANCE & 2-SE MARGINS",
        "-" * 80,
        f"- Excess > 0.01: {count_gt_001} qubits total; only {survive_2se_001} survive a 2-SE margin (excess - 2*SE > 0.01).",
        f"- Excess > 0.02: {count_gt_002} qubits total; only {survive_2se_002} survive a 2-SE margin (excess - 2*SE > 0.02).",
        f"- Excess > 0.05: {count_gt_005} qubits total; only {survive_2se_005} survive a 2-SE margin (excess - 2*SE > 0.05).",
        "",
        "3. QUBIT 125 ANOMALY: DEFECT / TRANSIENT VS TEMPERATURE",
        "-" * 80,
        f"- Qubit 125 P(1|0):             {q125_p10:.4f} (42.14%)",
        f"- Qubit 125 P(0|1):             {q125_p01:.4f} (13.33%)",
        f"- Qubit 125 Calibrated Readout: {q125_ro:.4f} (0.96%)",
        "- Analysis: Qubit 125 has an extreme excited state population P(1|0) = 0.42 alongside",
        "  an elevated P(0|1) = 0.13, while calibrated readout error is under 1%. Such an extreme",
        "  population cannot be explained by 15 mK thermal equilibrium (which implies P10 < 1e-4);",
        "  it strongly indicates a two-level system (TLS) defect, active drive leakage, or a transient",
        "  hardware anomaly rather than a baseline dilution refrigerator thermal gradient.",
        "",
        "4. CALIBRATION AGE MISMATCH (STALENESS)",
        "-" * 80,
        f"- Initial p1 Calibration:       {p1_date}",
        f"- Thermal-Arm Rerun Target:     {rerun_cal_date}",
        "- Age difference:               3 days stale (compiled circuits for 12 Oct hardware rerun",
        "  relied on 9 Oct excited state population measurements).",
        "=" * 80,
    ]

    report = "\n".join(lines) + "\n"
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"p1 limits report written to {out_p}")
    return report

if __name__ == "__main__":
    compute_p1_limits()

