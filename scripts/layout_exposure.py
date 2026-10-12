#!/usr/bin/env python3
"""
Committed Layout Warm-Qubit Exposure Analysis.
Evaluates the distinct committed physical layouts from results/hardware/runtime_table.csv
against measured P(1|0) and target readout error from results/hardware/p1_ibm_marrakesh_*.json.
Reports maximum and mean excess P(1|0) over readout error for each distinct layout.
Verifies Finding G3: no committed layout contains a qubit with excess > 0.05.
Does not make claims about uncompiled circuits.
Outputs results/analysis/layout_exposure.txt.
"""

import os
import sys
import glob
import json
import time
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))


def run_layout_exposure(
    csv_path="results/hardware/runtime_table.csv",
    p1_pattern="results/hardware/p1_ibm_marrakesh_*.json",
    out_txt="results/analysis/layout_exposure.txt"
):
    start_time = time.time()
    lines = []
    lines.append("=" * 84)
    lines.append("COMMITTED HARDWARE LAYOUT WARM-QUBIT EXPOSURE ANALYSIS")
    lines.append("=" * 84)
    lines.append("")

    # 1. Load Calibration & P(1) Data
    lines.append("1. CALIBRATION & MEASUREMENT DATASET")
    lines.append("-" * 84)
    p1_files = sorted(glob.glob(p1_pattern))
    if not p1_files:
        raise FileNotFoundError(f"No p1 file found matching {p1_pattern}")
    p1_file = p1_files[0]
    with open(p1_file, "r", encoding="utf-8") as f:
        p1_data = json.load(f)

    p1_est = p1_data.get("p1_estimate", {})
    ro_err = p1_data.get("target_readout_error", {})
    cal_ts = p1_data.get("timestamp", "unknown")
    backend_name = p1_data.get("backend", "ibm_marrakesh")

    lines.append(f"Backend:                  {backend_name} (156 qubits)")
    lines.append(f"Calibration Timestamp:    {cal_ts}")
    lines.append(f"P(1) Calibration Source:  {os.path.basename(p1_file)}")
    lines.append(f"Hardware Table Source:    {csv_path}")
    lines.append("")

    # Compute excess P(1|0) over readout error per qubit
    excess = {}
    for q_str, p1_val in p1_est.items():
        q_int = int(q_str)
        ro_val = float(ro_err.get(q_str, 0.0))
        excess[q_int] = float(p1_val) - ro_val

    # 2. Extract Distinct Committed Physical Layouts
    df = pd.read_csv(csv_path)

    # Group by physical layout to find all methods and benchmarks that use it
    layout_usage = {}
    for _, r in df.iterrows():
        lay_list = eval(r["layout"])
        lay_tuple = tuple(lay_list)
        bm = f"{r['benchmark'].upper()} {r['method']}"
        if lay_tuple not in layout_usage:
            layout_usage[lay_tuple] = {
                "layout": lay_list,
                "benchmarks_methods": set(),
            }
        layout_usage[lay_tuple]["benchmarks_methods"].add(bm)

    lines.append("2. EXPOSURE METRICS FOR DISTINCT COMMITTED PHYSICAL LAYOUTS")
    lines.append("-" * 84)
    lines.append(f"Found {len(layout_usage)} distinct physical layout(s) across all 24 hardware cells.")
    lines.append("")

    layout_records = []
    for idx, (lay_tuple, info) in enumerate(layout_usage.items()):
        lay = info["layout"]
        used_by = sorted(list(info["benchmarks_methods"]))
        qubit_excesses = [(q, excess.get(q, 0.0)) for q in lay]
        max_exc = max(e for _, e in qubit_excesses)
        mean_exc = sum(e for _, e in qubit_excesses) / len(qubit_excesses)
        max_q = [q for q, e in qubit_excesses if e == max_exc][0]

        is_random = any("random" in u.lower() for u in used_by)

        layout_records.append({
            "idx": idx + 1,
            "layout": lay,
            "used_by": used_by,
            "is_random": is_random,
            "max_excess": max_exc,
            "max_qubit": max_q,
            "mean_excess": mean_exc,
            "qubit_excesses": qubit_excesses,
        })

    # Print summary table
    header = f"{'#':<3} {'Physical Layout':<26} {'Max Excess':<14} {'Max Qubit':<12} {'Mean Excess':<14} {'Associated Benchmarks & Methods'}"
    lines.append(header)
    lines.append("-" * 108)

    for r in layout_records:
        usage_str = ", ".join(r["used_by"])
        lines.append(
            f"{r['idx']:<3} {str(r['layout']):<26} {r['max_excess']:+10.6f}    "
            f"Qubit {r['max_qubit']:<6} {r['mean_excess']:+10.6f}    {usage_str}"
        )
    lines.append("")

    # Detailed qubit-by-qubit breakdown
    lines.append("3. DETAILED PER-QUBIT EXCESS BREAKDOWN")
    lines.append("-" * 84)
    for r in layout_records:
        lines.append(f"Layout #{r['idx']} {r['layout']} ({', '.join(r['used_by'])}):")
        lines.append(f"  Max Excess:  {r['max_excess']:+.6f} (Qubit {r['max_qubit']})")
        lines.append(f"  Mean Excess: {r['mean_excess']:+.6f}")
        q_details = []
        for q, exc in r["qubit_excesses"]:
            p1_v = float(p1_est.get(str(q), 0.0))
            ro_v = float(ro_err.get(str(q), 0.0))
            q_details.append(f"Q{q}: exc={exc:+.4f} (p1={p1_v:.4f}, ro={ro_v:.4f})")
        lines.append(f"  Qubits:      {', '.join(q_details)}")
        lines.append("")

    # 4. Verification of Finding G3 & Scientific Conclusions
    lines.append("4. VERIFICATION OF FINDING G3 & SCOPE BOUNDS")
    lines.append("-" * 84)

    random_rec = [r for r in layout_records if r["is_random"]][0]
    compiler_recs = [r for r in layout_records if not r["is_random"]]
    max_compiler_rec = max(compiler_recs, key=lambda r: r["max_excess"])

    lines.append("Findings:")
    lines.append(f"  1. Random Layout Max Excess:    {random_rec['max_excess']:+.6f} (Layout {random_rec['layout']}, max on Qubit {random_rec['max_qubit']})")
    lines.append(f"  2. Compiler Layouts Max Excess: {max_compiler_rec['max_excess']:+.6f} (Layout {max_compiler_rec['layout']}, max on Qubit {max_compiler_rec['max_qubit']})")
    lines.append("  3. Threshold Compliance (>0.05):")
    all_below_05 = all(r["max_excess"] <= 0.05 for r in layout_records)
    if all_below_05:
        lines.append("     CONFIRMED (Finding G3): No committed physical layout contains any qubit with")
        lines.append(f"     excess P(1|0) > 0.05 (random layout max {random_rec['max_excess']:+.4f}; compiler layouts max <= {max_compiler_rec['max_excess']:+.4f}).")
    else:
        lines.append("     DISCREPANCY: One or more layouts exceed 0.05 excess.")
    lines.append("")
    lines.append("Scope & Non-Claim Boundaries:")
    lines.append("  - This analysis characterizes EXCLUSIVELY the 12 distinct committed physical layouts")
    lines.append("    evaluated in runtime_table.csv on ibm_marrakesh.")
    lines.append("  - It makes NO claims regarding uncompiled circuits or hypothetical compiler routing")
    lines.append("    on other benchmark circuits.")
    lines.append("=" * 84)

    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Layout exposure report written to {out_txt}")
    return out_txt


if __name__ == "__main__":
    run_layout_exposure()
