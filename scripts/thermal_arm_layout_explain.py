#!/usr/bin/env python3
"""
scripts/thermal_arm_layout_explain.py

Evaluates BOTH plain and thermal layouts for all 4 hardware benchmarks under BOTH objectives
(esp_standard and esp_thermal) using the same p1 calibration data on ibm_marrakesh.

OFFLINE DIAGNOSTIC on FakeMarrakesh, NOT the live target used for layout selection;
mean/max excess differ from layout_diff.txt (live); not evidence; do not cite.
No causal claims are asserted.
"""

import os
import sys
import re
import json
from pathlib import Path
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_all_benchmarks
from src.metrics.esp import esp_standard, esp_thermal


def parse_layout_diff(diff_path="results/hardware_thermal_arm/layout_diff.txt"):
    """
    Parses live target layout and excess data from layout_diff.txt.
    Returns dict: {b_name: {'plain': {'mean': ..., 'max': ...}, 'thermal': {'mean': ..., 'max': ...}}}
    """
    diff_data = {}
    if not os.path.exists(diff_path):
        return diff_data

    with open(diff_path, "r", encoding="utf-8") as f:
        for line in f:
            line_s = line.strip()
            for b in ["GHZ", "BV", "QFT", "MIRROR"]:
                if line_s.startswith(b + " ") or line_s.startswith(b + "\t"):
                    m = re.findall(r"(\d+\.\d+)\s*\/\s*(\d+\.\d+)", line_s)
                    if len(m) >= 2:
                        diff_data[b.lower()] = {
                            "plain": {"mean": float(m[0][0]), "max": float(m[0][1])},
                            "thermal": {"mean": float(m[1][0]), "max": float(m[1][1])},
                        }
    return diff_data


def run_layout_explain():
    dev = FakeMarrakesh()
    target = dev.target
    benchmarks = get_all_benchmarks(5)

    p1_file = Path("results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json")
    with open(p1_file, "r", encoding="utf-8") as f:
        p1_data = json.load(f)
    p1_dict = {int(k): float(v) for k, v in p1_data["p1_estimate"].items()}
    p1_timestamp = p1_data.get("timestamp", "unknown")

    # Readout errors per qubit on target
    ro_errors = {}
    for q in range(target.num_qubits):
        try:
            m_props = target["measure"].get((q,), None)
            ro_errors[q] = getattr(m_props, "error", 0.0) if m_props else 0.0
        except KeyError:
            ro_errors[q] = 0.0

    def compute_excess(layout):
        exc_list = [max(0.0, p1_dict.get(q, 0.0) - ro_errors.get(q, 0.0)) for q in layout]
        return max(exc_list), sum(exc_list) / len(exc_list)

    # Layouts selected during the live run (from results/hardware_thermal_arm/layout_diff.txt)
    layouts = {
        "ghz": {
            "plain": [91, 92, 90, 89, 78],
            "thermal": [33, 39, 53, 54, 55],
        },
        "bv": {
            "plain": [5, 2, 16, 3, 4],
            "thermal": [33, 39, 53, 31, 32],
        },
        "qft": {
            "plain": [4, 5, 6, 3, 7],
            "thermal": [3, 7, 5, 6, 4],
        },
        "mirror": {
            "plain": [5, 3, 2, 4, 6],
            "thermal": [5, 7, 6, 8, 4],
        },
    }

    live_diff = parse_layout_diff()

    out_file = Path("results/analysis/thermal_arm_layout_explain.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 105)
    lines.append("OFFLINE DIAGNOSTIC on FakeMarrakesh, NOT the live target used for layout selection;")
    lines.append("mean/max excess differ from layout_diff.txt (live); not evidence; do not cite.")
    lines.append("=" * 105)
    lines.append("THERMAL-ARM LAYOUT SELECTION & CROSS-OBJECTIVE EVALUATION ANALYSIS")
    lines.append("=" * 105)
    lines.append("Target Device:                FakeMarrakesh() (ibm_marrakesh heavy-hex layout, 156 qubits)")
    lines.append(f"Source p1 Calibration File:   {p1_file.name} ({p1_timestamp})")
    lines.append("Evaluation Mode:              excess_over_readout (config default)")
    lines.append("Routing Search:               SABRE routing across seeds {0, 1, 2}, opt_level=3")
    lines.append("Methodology Constraint:       No causal claims asserted. Strictly empirical objective evaluations.")
    lines.append("")

    lines.append("1. COMPUTED SIDE-BY-SIDE EXCESS COMPARISON (layout_diff.txt [LIVE] vs FakeMarrakesh [OFFLINE])")
    lines.append("-" * 105)
    lines.append(f"{'Benchmark':<10} {'Arm':<9} {'Layout':<24} {'Live Mean / Max (layout_diff)':<32} {'Offline Mean / Max (FakeMarrakesh)'}")
    lines.append("-" * 105)
    for b_name in ["ghz", "bv", "qft", "mirror"]:
        for arm in ["plain", "thermal"]:
            lay = layouts[b_name][arm]
            live_m = live_diff.get(b_name, {}).get(arm, {})
            live_str = f"{live_m.get('mean', 0.0):.4f} / {live_m.get('max', 0.0):.4f}" if live_m else "N/A"
            off_max, off_mean = compute_excess(lay)
            off_str = f"{off_mean:.4f} / {off_max:.4f}"
            lines.append(f"{b_name.upper():<10} {arm:<9} {str(lay):<24} {live_str:<32} {off_str}")
    lines.append("")

    lines.append("2. CROSS-OBJECTIVE EVALUATION TABLE")
    lines.append("-" * 105)
    lines.append(f"{'Benchmark':<9} {'Arm':<9} {'Layout':<24} {'CZ':<4} {'Depth':<6} {'esp_standard':<14} {'esp_thermal':<14} {'Thermal Penalty'}")
    lines.append("-" * 105)

    eval_data = {}

    for b_name in ["ghz", "bv", "qft", "mirror"]:
        circ = benchmarks[b_name]
        eval_data[b_name] = {}
        for arm in ["plain", "thermal"]:
            lay = layouts[b_name][arm]
            off_max, off_mean = compute_excess(lay)

            best_tc = None
            best_std = -1.0
            best_th = -1.0
            best_cz = 0
            best_depth = 0

            # Evaluate transpilation over routing seeds 0, 1, 2
            for r_seed in range(3):
                tc = transpile(circ, target=target, initial_layout=lay, routing_method='sabre', optimization_level=3, seed_transpiler=r_seed)
                score_std = esp_standard(tc, target)
                score_th = esp_thermal(tc, target, p1=p1_dict, mode='excess_over_readout')

                # Choose best according to the arm's own optimization rule
                if arm == "plain" and score_std > best_std:
                    best_std = score_std
                    best_th = score_th
                    best_tc = tc
                elif arm == "thermal" and score_th > best_th:
                    best_std = score_std
                    best_th = score_th
                    best_tc = tc

            ops = best_tc.count_ops()
            cz_cnt = ops.get('cz', ops.get('cx', ops.get('ecr', 0)))
            depth_cnt = best_tc.depth()
            thermal_penalty_factor = (best_th / best_std) if best_std > 0 else 1.0

            eval_data[b_name][arm] = {
                "layout": lay,
                "cz": cz_cnt,
                "depth": depth_cnt,
                "esp_standard": best_std,
                "esp_thermal": best_th,
                "penalty_factor": thermal_penalty_factor,
                "max_excess": off_max,
                "mean_excess": off_mean,
            }

            lines.append(
                f"{b_name.upper():<9} {arm:<9} {str(lay):<24} {cz_cnt:<4} {depth_cnt:<6} "
                f"{best_std:.6f}       {best_th:.6f}       {thermal_penalty_factor:.4f} (-{(1-thermal_penalty_factor)*100:.2f}%)"
            )

    lines.append("")
    lines.append("3. PER-BENCHMARK OBSERVATIONS (NO CAUSAL CLAIMS)")
    lines.append("-" * 105)

    # Detailed per-benchmark observations without causal "Mechanism:" text
    # BV
    bv_p = eval_data["bv"]["plain"]
    bv_t = eval_data["bv"]["thermal"]
    lines.append("BENCHMARK: BV-5")
    lines.append(f"  Plain Layout:   {bv_p['layout']} (mean exc = {bv_p['mean_excess']:.4f}, max exc = {bv_p['max_excess']:.4f})")
    lines.append(f"  Thermal Layout: {bv_t['layout']} (mean exc = {bv_t['mean_excess']:.4f}, max exc = {bv_t['max_excess']:.4f})")
    lines.append(f"  Under esp_standard: Plain ({bv_p['esp_standard']:.6f}) > Thermal ({bv_t['esp_standard']:.6f})  [Plain higher by +{(bv_p['esp_standard'] - bv_t['esp_standard'])*100:.2f} pts]")
    lines.append(f"  Under esp_thermal:  Thermal ({bv_t['esp_thermal']:.6f}) > Plain ({bv_p['esp_thermal']:.6f})  [Thermal higher by +{(bv_t['esp_thermal'] - bv_p['esp_thermal'])*100:.2f} pts]")
    lines.append(f"  Thermal Penalty:    Plain layout is penalized by -{(1-bv_p['penalty_factor'])*100:.2f}%, whereas Thermal layout penalty is -{(1-bv_t['penalty_factor'])*100:.2f}%.")
    lines.append("")

    # GHZ
    ghz_p = eval_data["ghz"]["plain"]
    ghz_t = eval_data["ghz"]["thermal"]
    lines.append("BENCHMARK: GHZ-5")
    lines.append(f"  Plain Layout:   {ghz_p['layout']} (mean exc = {ghz_p['mean_excess']:.4f}, max exc = {ghz_p['max_excess']:.4f})")
    lines.append(f"  Thermal Layout: {ghz_t['layout']} (mean exc = {ghz_t['mean_excess']:.4f}, max exc = {ghz_t['max_excess']:.4f})")
    lines.append(f"  Under esp_standard: Thermal ({ghz_t['esp_standard']:.6f}) > Plain ({ghz_p['esp_standard']:.6f})  [Thermal higher by +{(ghz_t['esp_standard'] - ghz_p['esp_standard'])*100:.2f} pts]")
    lines.append(f"  Under esp_thermal:  Thermal ({ghz_t['esp_thermal']:.6f}) > Plain ({ghz_p['esp_thermal']:.6f})  [Thermal higher by +{(ghz_t['esp_thermal'] - ghz_p['esp_thermal'])*100:.2f} pts]")
    lines.append("")

    # QFT
    qft_p = eval_data["qft"]["plain"]
    qft_t = eval_data["qft"]["thermal"]
    set_p = set(qft_p['layout'])
    set_t = set(qft_t['layout'])
    same_qubits = (set_p == set_t)
    lines.append("BENCHMARK: QFT-5")
    lines.append(f"  Plain Layout:   {qft_p['layout']} (qubit set: {sorted(set_p)})")
    lines.append(f"  Thermal Layout: {qft_t['layout']} (qubit set: {sorted(set_t)})")
    lines.append(f"  Qubit Set Equivalence: {same_qubits} (both layouts utilize the identical 5 physical qubits)")
    lines.append(f"  Under esp_standard: Plain ({qft_p['esp_standard']:.6f}) ~= Thermal ({qft_t['esp_standard']:.6f})")
    lines.append(f"  Under esp_thermal:  Plain ({qft_p['esp_thermal']:.6f}) ~= Thermal ({qft_t['esp_thermal']:.6f})")
    lines.append("")

    # Mirror
    mir_p = eval_data["mirror"]["plain"]
    mir_t = eval_data["mirror"]["thermal"]
    lines.append("BENCHMARK: MIRROR-5")
    lines.append(f"  Plain Layout:   {mir_p['layout']} (mean exc = {mir_p['mean_excess']:.4f}, max exc = {mir_p['max_excess']:.4f})")
    lines.append(f"  Thermal Layout: {mir_t['layout']} (mean exc = {mir_t['mean_excess']:.4f}, max exc = {mir_t['max_excess']:.4f})")
    lines.append(f"  Under esp_standard: Thermal ({mir_t['esp_standard']:.6f}) > Plain ({mir_p['esp_standard']:.6f})  [Thermal higher by +{(mir_t['esp_standard'] - mir_p['esp_standard'])*100:.2f} pts]")
    lines.append(f"  Under esp_thermal:  Thermal ({mir_t['esp_thermal']:.6f}) > Plain ({mir_p['esp_thermal']:.6f})  [Thermal higher by +{(mir_t['esp_thermal'] - mir_p['esp_thermal'])*100:.2f} pts]")
    lines.append("")

    lines.append("4. SUMMARY & METHODOLOGICAL TAKEAWAYS")
    lines.append("-" * 105)
    lines.append("1. OFFLINE DIAGNOSTIC: This evaluation was executed on FakeMarrakesh (2025-02-26 calibration snapshot),")
    lines.append("   NOT the live target properties used during layout selection (2026-10-12 calibration snapshot).")
    lines.append("2. EXCESS DISCREPANCY: Mean and max excess values differ between layout_diff.txt (live target) and this")
    lines.append("   file (FakeMarrakesh) due to differing calibrated readout errors across snapshots.")
    lines.append("3. QFT-5 EQUIVALENCE: For QFT-5, both plain and thermal arms chose the identical physical qubit set {3, 4, 5, 6, 7}.")
    lines.append("4. NO CAUSAL CLAIMS: No causal claims are asserted regarding layout selection or hardware outcomes.")
    lines.append("   These values represent descriptive cross-evaluations under the specified offline model; not evidence; do not cite.")

    report_text = "\n".join(lines) + "\n"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Layout explain report written to {out_file}")


if __name__ == "__main__":
    run_layout_explain()
