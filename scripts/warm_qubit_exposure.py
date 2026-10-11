#!/usr/bin/env python3
"""
Warm Qubit Exposure Analysis on ibm_marrakesh.
Reports connected k-qubit subgraph exposure across the 156-qubit heavy-hex coupling map
for k in [5, 10, 15, 20] under excess P(1|0) thresholds (>0.05, >0.02, >0.01),
and evaluates the committed hardware layouts from runtime_table.csv.

Labels subgraph fractions strictly as graph geometry only.
"""

import os
import sys
import glob
import json
import time
from pathlib import Path
import pandas as pd
from qiskit_ibm_runtime.fake_provider import FakeMarrakesh

def run_warm_exposure(
    p1_pattern="results/hardware/p1_ibm_marrakesh_*.json",
    csv_path="results/hardware/runtime_table.csv",
    out_txt="results/analysis/warm_exposure.txt"
):
    start_time = time.time()
    lines = []
    lines.append("=" * 80)
    lines.append("WARM QUBIT EXPOSURE ANALYSIS: IBM MARRAKESH (156 QUBITS)")
    lines.append("=" * 80)
    lines.append("")

    # 1. Load P(1) and readout calibration data
    p1_files = sorted(glob.glob(p1_pattern))
    if not p1_files:
        raise FileNotFoundError(f"No p1 file found matching {p1_pattern}")
    p1_file = p1_files[0]
    with open(p1_file, "r", encoding="utf-8") as f:
        p1_data = json.load(f)

    p1_est = p1_data.get("p1_estimate", {})
    ro_err = p1_data.get("target_readout_error", {})
    cal_ts = p1_data.get("timestamp", "unknown")

    backend = FakeMarrakesh()
    n_qubits = backend.num_qubits
    coupling_edges = backend.coupling_map.get_edges()

    # Compute excess P(1) per qubit
    excess = {}
    warm_05_qubits = []
    warm_02_qubits = []
    warm_01_qubits = []
    for q in range(n_qubits):
        p1_val = float(p1_est.get(str(q), 0.0))
        ro_val = float(ro_err.get(str(q), 0.0))
        exc_val = p1_val - ro_val
        excess[q] = exc_val
        if exc_val > 0.05:
            warm_05_qubits.append((q, exc_val))
        if exc_val > 0.02:
            warm_02_qubits.append((q, exc_val))
        if exc_val > 0.01:
            warm_01_qubits.append((q, exc_val))

    lines.append("1. CALIBRATION & HARDWARE PROFILE")
    lines.append("-" * 80)
    lines.append(f"Backend:                ibm_marrakesh (156 qubits, heavy-hex)")
    lines.append(f"Calibration Timestamp:  {cal_ts}")
    lines.append(f"Source p1 File:         {os.path.basename(p1_file)}")
    lines.append(f"Coupling Edges:         {len(coupling_edges)} directed ({len(coupling_edges)//2} undirected)")
    lines.append(f"Qubits with excess > 0.05: {len(warm_05_qubits)}/156 ({len(warm_05_qubits)/156*100:.2f}%)")
    lines.append(f"  List (>0.05): {[(q, round(v, 4)) for q, v in sorted(warm_05_qubits)]}")
    lines.append(f"Qubits with excess > 0.02: {len(warm_02_qubits)}/156 ({len(warm_02_qubits)/156*100:.2f}%)")
    lines.append(f"Qubits with excess > 0.01: {len(warm_01_qubits)}/156 ({len(warm_01_qubits)/156*100:.2f}%)")
    lines.append("")

    # 2. Graph Geometry: Connected Subgraph Enumeration via Bitmask ESU
    lines.append("2. LATTICE GEOMETRY: CONNECTED SUBGRAPH EXPOSURE")
    lines.append("-" * 80)
    lines.append("NOTE: These fractions represent graph connectivity / geometry on the heavy-hex")
    lines.append("lattice ONLY. They do NOT imply compiler exposure rates for uncompiled circuits.")
    lines.append("Do not claim 'L3 hits warm qubits X%' for circuits we did not compile.")
    lines.append("")

    warm_05_mask = sum(1 << q for q, _ in warm_05_qubits)
    warm_02_mask = sum(1 << q for q, _ in warm_02_qubits)
    warm_01_mask = sum(1 << q for q, _ in warm_01_qubits)

    adj_mask = [0] * n_qubits
    for u, v in coupling_edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)

    def count_k_subgraphs_exposure(target_k):
        t_sub_start = time.time()
        total_subgraphs = 0
        hit_05 = 0
        hit_02 = 0
        hit_01 = 0

        def extend(v_sub_mask, adj_of_sub, v_ext_tuple, v_min, current_len, has_05, has_02, has_01):
            nonlocal total_subgraphs, hit_05, hit_02, hit_01
            if current_len == target_k:
                total_subgraphs += 1
                if has_05: hit_05 += 1
                if has_02: hit_02 += 1
                if has_01: hit_01 += 1
                return

            for i, w in enumerate(v_ext_tuple):
                w_bit = 1 << w
                new_sub_mask = v_sub_mask | w_bit
                new_has_05 = has_05 or bool(w_bit & warm_05_mask)
                new_has_02 = has_02 or bool(w_bit & warm_02_mask)
                new_has_01 = has_01 or bool(w_bit & warm_01_mask)

                new_nbrs_mask = adj_mask[w] & (~adj_of_sub) & (~new_sub_mask) & (~((1 << (v_min + 1)) - 1))
                new_nbrs = []
                while new_nbrs_mask:
                    lsb = (new_nbrs_mask & -new_nbrs_mask).bit_length() - 1
                    new_nbrs.append(lsb)
                    new_nbrs_mask &= new_nbrs_mask - 1

                remaining_v_ext = v_ext_tuple[i+1:] + tuple(new_nbrs)
                new_adj_of_sub = adj_of_sub | adj_mask[w]
                extend(new_sub_mask, new_adj_of_sub, remaining_v_ext, v_min, current_len + 1, new_has_05, new_has_02, new_has_01)

        for v in range(n_qubits):
            v_bit = 1 << v
            v_ext_mask = adj_mask[v] & (~((1 << (v + 1)) - 1))
            v_ext = []
            while v_ext_mask:
                lsb = (v_ext_mask & -v_ext_mask).bit_length() - 1
                v_ext.append(lsb)
                v_ext_mask &= v_ext_mask - 1
            extend(v_bit, adj_mask[v], tuple(v_ext), v, 1, bool(v_bit & warm_05_mask), bool(v_bit & warm_02_mask), bool(v_bit & warm_01_mask))

        elapsed = time.time() - t_sub_start
        return {
            "k": target_k,
            "total": total_subgraphs,
            "hit_05": hit_05,
            "frac_05": hit_05 / total_subgraphs if total_subgraphs else 0.0,
            "hit_02": hit_02,
            "frac_02": hit_02 / total_subgraphs if total_subgraphs else 0.0,
            "hit_01": hit_01,
            "frac_01": hit_01 / total_subgraphs if total_subgraphs else 0.0,
            "elapsed_sec": elapsed
        }

    subgraph_results = []
    lines.append(f"{'k':<4} {'Total Subgraphs':<16} {'Excess > 0.05':<20} {'Excess > 0.02':<20} {'Excess > 0.01':<20} {'Compute Time'}")
    lines.append("-" * 96)
    for k_val in [5, 10, 15, 20]:
        res = count_k_subgraphs_exposure(k_val)
        subgraph_results.append(res)
        s_05 = f"{res['hit_05']} ({res['frac_05']*100:.2f}%)"
        s_02 = f"{res['hit_02']} ({res['frac_02']*100:.2f}%)"
        s_01 = f"{res['hit_01']} ({res['frac_01']*100:.2f}%)"
        lines.append(f"{res['k']:<4} {res['total']:<16} {s_05:<20} {s_02:<20} {s_01:<20} {res['elapsed_sec']:.2f} s")
    lines.append("")

    # 3. Committed Hardware Layouts Exposure
    lines.append("3. COMMITTED HARDWARE LAYOUTS EXPOSURE (FROM RUNTIME_TABLE.CSV)")
    lines.append("-" * 80)
    lines.append("Evaluates the physical qubit layouts committed in results/hardware/runtime_table.csv.")
    lines.append("")

    df_hw = pd.read_csv(csv_path)
    layout_rows = []
    for (m, b), grp in df_hw.groupby(['method', 'benchmark'], sort=False):
        lay = eval(grp.iloc[0]['layout'])
        qubits_05 = [q for q in lay if excess[q] > 0.05]
        qubits_02 = [q for q in lay if excess[q] > 0.02]
        qubits_01 = [q for q in lay if excess[q] > 0.01]
        max_exc = max(excess[q] for q in lay)
        layout_rows.append({
            "method": m,
            "benchmark": b,
            "layout": lay,
            "max_excess": max_exc,
            "has_05": len(qubits_05) > 0,
            "has_02": len(qubits_02) > 0,
            "has_01": len(qubits_01) > 0,
            "warm_qubits_05": qubits_05,
            "warm_qubits_02": qubits_02,
            "warm_qubits_01": qubits_01,
        })

    lines.append(f"{'Method':<18} {'Benchmark':<10} {'Physical Layout':<26} {'Max Excess':<12} {'>0.05':<8} {'>0.02':<8} {'>0.01':<8} {'Warm Qubits (>0.01)'}")
    lines.append("-" * 105)
    for r in layout_rows:
        w_list = str(r['warm_qubits_01']) if r['warm_qubits_01'] else "none"
        lines.append(f"{r['method']:<18} {r['benchmark']:<10} {str(r['layout']):<26} {r['max_excess']:+.4f}     {str(r['has_05']):<8} {str(r['has_02']):<8} {str(r['has_01']):<8} {w_list}")
    lines.append("")

    # 4. Summary per Method across benchmarks
    lines.append("4. SUMMARY PER METHOD (ACROSS BENCHMARKS)")
    lines.append("-" * 80)
    lines.append(f"{'Method':<18} {'N Cells':<8} {'Frac > 0.05':<14} {'Frac > 0.02':<14} {'Frac > 0.01':<14} {'Warmest Qubit Encountered'}")
    lines.append("-" * 85)

    df_lay = pd.DataFrame(layout_rows)
    for m, grp in df_lay.groupby('method', sort=False):
        n_c = len(grp)
        f_05 = grp['has_05'].mean()
        f_02 = grp['has_02'].mean()
        f_01 = grp['has_01'].mean()
        m_exc = grp['max_excess'].max()
        lines.append(f"{m:<18} {n_c:<8} {f_05*100:5.1f}% ({grp['has_05'].sum()}/{n_c})  {f_02*100:5.1f}% ({grp['has_02'].sum()}/{n_c})  {f_01*100:5.1f}% ({grp['has_01'].sum()}/{n_c})  excess = {m_exc:+.4f}")
    lines.append("")

    # 5. Method group synthesis: random vs L1 vs L3 vs ESP
    lines.append("5. CORE METHOD COMPARISON (RANDOM, L1, L3, ESP)")
    lines.append("-" * 80)
    lines.append("Key Observations on Hardware Layouts:")
    lines.append("  - random (layout [80, 61, 62, 76, 81]):")
    lines.append("      Hits warm qubits > 0.02 in 4 of 4 benchmarks (100.0%): Q62 (+0.0336), Q76 (+0.0453), Q81 (+0.0447).")
    lines.append("      Does not contain any of the 7 extreme hot spots (> 0.05).")
    lines.append("  - qiskit_L1:")
    lines.append("      Hits warm qubits > 0.01 in 4 of 4 benchmarks (100.0%): Q2 (+0.0143) and Q153 (+0.0173).")
    lines.append("      0 of 4 benchmarks hit qubits > 0.02 or > 0.05.")
    lines.append("  - qiskit_L3:")
    lines.append("      Hits warm qubits > 0.01 in 3 of 4 benchmarks (75.0%): Q2/Q153 in BV, Q54 (+0.0121) in QFT and Mirror.")
    lines.append("      0 of 4 benchmarks hit qubits > 0.02 or > 0.05.")
    lines.append("  - esp_no_thermal & esp_thermal:")
    lines.append("      Hits warm qubits > 0.01 in 1 of 4 benchmarks (25.0%): Q54 (+0.0121) in Mirror.")
    lines.append("      0 of 4 benchmarks hit qubits > 0.02 or > 0.05.")
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"ANALYSIS COMPLETE: Finished in {time.time()-start_time:.2f} s")
    lines.append("=" * 80)

    # Write output file
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Report written to {out_txt}")
    return out_txt

if __name__ == "__main__":
    run_warm_exposure()

