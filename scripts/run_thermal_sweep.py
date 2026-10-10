#!/usr/bin/env python3
"""
Thermal-Gradient Sensitivity Sweep Script.
Investigates at what excess excited-state population (excess_p1) thermal ESP
changes layout and whether it improves fidelity over plain ESP.

Grid:
- Devices: FakeVigoV2 (5q), FakeGuadalupeV2 (16q)
- excess_p1 in {0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15}
- hot_qubit_count in {1, 2, 3}
- benchmarks: ghz, bv, qft, mirror (N=5)
- Seeds: 10 profile seeds per cell (time-budget compliant)

Outputs:
- results/sim/thermal_sweep.csv
- results/sim/thermal_sweep.provenance.json
- results/figures/thermal_threshold.png
- Replay check against results/hardware/p1_ibm_marrakesh_*.json
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeGuadalupeV2

from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_esp, mapper_qiskit_default, mapper_qiskit_best3, get_active_qubits
from src.metrics.outcome import compute_outcome_metric
from scripts.run_sim import compact_circuit
from scripts.make_provenance_and_summary import write_provenance

def bootstrap_ci_mean(data: np.ndarray, n_resamples: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    data = np.asarray(data, dtype=float)
    if len(data) == 0:
        return 0.0, 0.0, 0.0
    mean_val = float(np.mean(data))
    if len(data) == 1 or np.all(data == data[0]):
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    boot_means = np.array([np.mean(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)])
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean_val, float(low), float(high)

_compact_cache = {}

def get_compact(tc: QuantumCircuit, target, lay: list):
    key = (id(tc), tuple(lay) if lay is not None else None)
    if key not in _compact_cache:
        _compact_cache[key] = compact_circuit(tc, target, None, lay)
    return _compact_cache[key]

def simulate_with_mixture_compact(tc: QuantumCircuit, target, lay: list, true_excess_map: dict[int, float], shots: int, seed_sim: int) -> dict[str, int]:
    # Compact circuit to active qubits and extract compact noise model
    compact_qc, nm = get_compact(tc, target, lay)
    sim = AerSimulator(noise_model=nm)
    
    # Active qubits and mapping from physical qubit index to compact qubit index
    all_active = sorted(list(get_active_qubits(tc)))
    mapping = {q: i for i, q in enumerate(all_active)}
    
    # Filter true_excess_map to active qubits in this circuit
    compact_excess = {mapping[q]: p for q, p in true_excess_map.items() if q in mapping and p > 0}
    
    if not compact_excess:
        return sim.run(compact_qc, shots=shots, seed_simulator=seed_sim).result().get_counts()
        
    rng = np.random.default_rng(seed_sim)
    flips = np.zeros((shots, compact_qc.num_qubits), dtype=bool)
    for q_comp, p in compact_excess.items():
        flips[:, q_comp] = rng.random(shots) < p
        
    patterns, pcounts = np.unique(flips, axis=0, return_counts=True)
    tot_counts: dict[str, int] = {}
    for pat, cnt in zip(patterns, pcounts):
        hot_qubits_in_circuit = np.where(pat)[0].tolist()
        if not hot_qubits_in_circuit:
            res = sim.run(compact_qc, shots=int(cnt), seed_simulator=seed_sim).result().get_counts()
        else:
            prep = QuantumCircuit(compact_qc.num_qubits, compact_qc.num_clbits)
            for hq in hot_qubits_in_circuit:
                prep.x(hq)
            comp = prep.compose(compact_qc)
            res = sim.run(comp, shots=int(cnt), seed_simulator=seed_sim).result().get_counts()
        for k, v in res.items():
            tot_counts[k] = tot_counts.get(k, 0) + int(v)
            
    return tot_counts

def run_thermal_sweep(num_seeds: int = 10, shots: int = 8192) -> tuple[pd.DataFrame, dict]:
    devices = {
        'fake_vigo': FakeVigoV2(),
        'fake_guadalupe': FakeGuadalupeV2()
    }
    
    # Assert qubit_properties exist on targets
    for dev_name, dev in devices.items():
        assert getattr(dev.target, 'qubit_properties', None) is not None, f"Device {dev_name} target has no qubit_properties"
        assert dev.target.qubit_properties[0].t1 is not None, f"Device {dev_name} qubit 0 has no T1"
        
    benchmarks_dict = get_all_benchmarks(5)
    bench_names = ['ghz', 'bv', 'qft', 'mirror']
    
    excess_grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
    hot_count_grid = [1, 2, 3]
    
    records = []
    
    print("=" * 80)
    print(f"STARTING THERMAL-GRADIENT SENSITIVITY SWEEP (Seeds per cell: {num_seeds}, Shots: {shots})")
    print("=" * 80)
    
    t_start = time.time()
    total_cells = len(devices) * len(bench_names) * len(hot_count_grid) * len(excess_grid)
    cell_idx = 0
    
    for dev_name, dev in devices.items():
        target = dev.target
        num_qubits = target.num_qubits
        
        # Precompute readout errors on physical qubits
        ro_errors = {}
        for q in range(num_qubits):
            try:
                mp = target['measure'].get((q,), None)
                ro_errors[q] = getattr(mp, 'error', 0.0) if mp else 0.0
            except KeyError:
                ro_errors[q] = 0.0
                
        for b_name in bench_names:
            qc = benchmarks_dict[b_name]
            
            # Baseline plain ESP layout without thermal info
            tc_esp_default, _, _, lay_esp_default, _ = mapper_esp(
                qc, target, use_thermal=False, routing_seeds=1, max_candidate_layouts=30
            )
            # Baseline L3 and L3_best3
            tc_l3_default, _, _, lay_l3_default, _ = mapper_qiskit_default(qc, target, level=3)
            tc_b3_default, _, _, lay_b3_default, _ = mapper_qiskit_best3(qc, target, level=3, routing_seeds=3)
            
            for k_hot in hot_count_grid:
                if k_hot > num_qubits:
                    continue
                    
                for excess_p1 in excess_grid:
                    cell_idx += 1
                    cell_draws_changed = 0
                    paired_gains = []
                    paired_oracle_gains = []
                    gains_vs_l3 = []
                    
                    for s in range(num_seeds):
                        seed_val = 1000 * cell_idx + s
                        rng_draw = np.random.default_rng(seed_val)
                        
                        # Placement rule: if excess_p1 > 0, at least half of the draws (s < num_seeds/2)
                        # must draw at least 1 hot qubit from lay_esp_default
                        if excess_p1 > 0:
                            if s < (num_seeds // 2) and any(q in lay_esp_default for q in range(num_qubits)):
                                in_q = rng_draw.choice(lay_esp_default)
                                other_qubits = [q for q in range(num_qubits) if q != in_q]
                                if k_hot > 1:
                                    rem_qubits = rng_draw.choice(other_qubits, size=k_hot - 1, replace=False).tolist()
                                    hot_qubits = [int(in_q)] + [int(q) for q in rem_qubits]
                                else:
                                    hot_qubits = [int(in_q)]
                            else:
                                hot_qubits = [int(q) for q in rng_draw.choice(range(num_qubits), size=k_hot, replace=False)]
                        else:
                            hot_qubits = []
                            
                        true_excess_map = {q: excess_p1 for q in hot_qubits}
                        
                        # Simulate estimated measurement of p1:
                        # Measured p1 has binomial shot noise with 8192 shots + readout error
                        estimated_p1 = {}
                        oracle_p1 = {}
                        for q in range(num_qubits):
                            true_p = true_excess_map.get(q, 0.0) + ro_errors[q]
                            k_meas = rng_draw.binomial(8192, np.clip(true_p, 0.0, 1.0))
                            meas_rate = k_meas / 8192.0
                            # Excess over readout
                            est_excess = max(0.0, meas_rate - ro_errors[q])
                            estimated_p1[q] = est_excess
                            oracle_p1[q] = true_excess_map.get(q, 0.0)
                            
                        # If excess_p1 == 0, thermal ESP receives 0 excess
                        if excess_p1 == 0.0:
                            estimated_p1 = {q: 0.0 for q in range(num_qubits)}
                            oracle_p1 = {q: 0.0 for q in range(num_qubits)}
                            
                        # Mapper with estimated p1
                        if excess_p1 == 0.0:
                            tc_esp_th = tc_esp_default
                            lay_esp_th = lay_esp_default
                            tc_esp_oracle = tc_esp_default
                            lay_esp_oracle = lay_esp_default
                        else:
                            tc_esp_th, _, _, lay_esp_th, _ = mapper_esp(
                                qc, target, p1=estimated_p1, use_thermal=True, routing_seeds=1, max_candidate_layouts=30
                            )
                            tc_esp_oracle, _, _, lay_esp_oracle, _ = mapper_esp(
                                qc, target, p1=oracle_p1, use_thermal=True, routing_seeds=1, max_candidate_layouts=30
                            )
                            
                        if lay_esp_th != lay_esp_default:
                            cell_draws_changed += 1
                            
                        # Run executions on Aer simulator using compact circuits
                        sim_seed = 42 + s
                        cnt_esp_no = simulate_with_mixture_compact(tc_esp_default, target, lay_esp_default, true_excess_map, shots, sim_seed)
                        if lay_esp_th == lay_esp_default:
                            cnt_esp_th = cnt_esp_no
                        else:
                            cnt_esp_th = simulate_with_mixture_compact(tc_esp_th, target, lay_esp_th, true_excess_map, shots, sim_seed)

                        if lay_esp_oracle == lay_esp_default:
                            cnt_esp_oracle = cnt_esp_no
                        elif lay_esp_oracle == lay_esp_th:
                            cnt_esp_oracle = cnt_esp_th
                        else:
                            cnt_esp_oracle = simulate_with_mixture_compact(tc_esp_oracle, target, lay_esp_oracle, true_excess_map, shots, sim_seed)

                        cnt_l3 = simulate_with_mixture_compact(tc_l3_default, target, lay_l3_default, true_excess_map, shots, sim_seed)
                        cnt_b3 = simulate_with_mixture_compact(tc_b3_default, target, lay_b3_default, true_excess_map, shots, sim_seed)

                        _, score_no = compute_outcome_metric(b_name, cnt_esp_no, qc, shots=shots)
                        score_th = score_no if lay_esp_th == lay_esp_default else compute_outcome_metric(b_name, cnt_esp_th, qc, shots=shots)[1]
                        score_oracle = score_no if lay_esp_oracle == lay_esp_default else (score_th if lay_esp_oracle == lay_esp_th else compute_outcome_metric(b_name, cnt_esp_oracle, qc, shots=shots)[1])
                        _, score_l3 = compute_outcome_metric(b_name, cnt_l3, qc, shots=shots)
                        _, score_b3 = compute_outcome_metric(b_name, cnt_b3, qc, shots=shots)
                        
                        paired_gains.append(score_th - score_no)
                        paired_oracle_gains.append(score_oracle - score_no)
                        gains_vs_l3.append(score_th - score_l3)
                        
                    mean_gain, ci_low, ci_high = bootstrap_ci_mean(np.array(paired_gains))
                    mean_oracle, _, _ = bootstrap_ci_mean(np.array(paired_oracle_gains))
                    mean_vs_l3, l3_low, l3_high = bootstrap_ci_mean(np.array(gains_vs_l3))
                    change_frac = cell_draws_changed / float(num_seeds)
                    
                    records.append({
                        'device': dev_name,
                        'benchmark': b_name,
                        'hot_qubit_count': k_hot,
                        'excess_p1': excess_p1,
                        'layout_change_fraction': change_frac,
                        'paired_gain_mean': mean_gain,
                        'paired_gain_ci_low': ci_low,
                        'paired_gain_ci_high': ci_high,
                        'oracle_gain_mean': mean_oracle,
                        'gain_vs_l3_mean': mean_vs_l3,
                        'gain_vs_l3_ci_low': l3_low,
                        'gain_vs_l3_ci_high': l3_high
                    })
                    
                    print(f"[{cell_idx}/{total_cells}] {dev_name:<14} | {b_name:<6} | hot={k_hot} | excess={excess_p1:5.3f} => changed: {change_frac:.2f} | gain: {mean_gain:+.4f} [{ci_low:+.4f}, {ci_high:+.4f}]")
                    sys.stdout.flush()
                    
    elapsed = time.time() - t_start
    print(f"\nSweep completed in {elapsed:.1f} s ({elapsed/60.0:.2f} min).")
    
    df_sweep = pd.DataFrame(records)
    out_csv = Path("results/sim/thermal_sweep.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_sweep.to_csv(out_csv, index=False)
    print(f"Saved thermal sweep results to {out_csv} ({len(df_sweep)} rows).")
    
    # Write provenance
    write_provenance(out_csv)
    
    # Determine pre-registered threshold:
    # Smallest excess_p1 where the paired-gain CI lower bound is > 0
    threshold_results = {}
    for dev_name in df_sweep['device'].unique():
        threshold_results[dev_name] = {}
        for b_name in bench_names:
            sub = df_sweep[(df_sweep['device'] == dev_name) & (df_sweep['benchmark'] == b_name)]
            # Find cells where paired_gain_ci_low > 0
            sig = sub[sub['paired_gain_ci_low'] > 0.0]
            if not sig.empty:
                min_excess = sig['excess_p1'].min()
                threshold_results[dev_name][b_name] = f"{min_excess:.3f}"
            else:
                threshold_results[dev_name][b_name] = "none found"
                
    return df_sweep, threshold_results

def plot_thermal_threshold(df_sweep: pd.DataFrame, out_fig: str = "results/figures/thermal_threshold.png"):
    Path(out_fig).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Aggregate across benchmarks and devices for summary visualization
    grouped = df_sweep.groupby(['hot_qubit_count', 'excess_p1']).agg({
        'paired_gain_mean': 'mean',
        'paired_gain_ci_low': 'mean',
        'paired_gain_ci_high': 'mean',
        'layout_change_fraction': 'mean'
    }).reset_index()
    
    colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
    
    # Panel 1: Paired Gain vs excess_p1 with CI bands
    ax1 = axes[0]
    for k in sorted(grouped['hot_qubit_count'].unique()):
        sub = grouped[grouped['hot_qubit_count'] == k].sort_values('excess_p1')
        ax1.plot(sub['excess_p1'], sub['paired_gain_mean'], marker='o', label=f"{k} hot qubit(s)", color=colors.get(k, 'black'))
        ax1.fill_between(sub['excess_p1'], sub['paired_gain_ci_low'], sub['paired_gain_ci_high'], alpha=0.2, color=colors.get(k, 'black'))
    ax1.axhline(0.0, color='gray', linestyle='--', linewidth=1.0)
    ax1.set_ylabel("Paired Gain (Thermal - Plain ESP)")
    ax1.set_title("Thermal-Gradient Sensitivity: Paired Metric Gain & Layout Change")
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')
    
    # Panel 2: Layout Change Fraction
    ax2 = axes[1]
    for k in sorted(grouped['hot_qubit_count'].unique()):
        sub = grouped[grouped['hot_qubit_count'] == k].sort_values('excess_p1')
        ax2.plot(sub['excess_p1'], sub['layout_change_fraction'], marker='s', label=f"{k} hot qubit(s)", color=colors.get(k, 'black'))
    ax2.set_xlabel("Injected Excess Population ($p_{1, \\mathrm{excess}}$)")
    ax2.set_ylabel("Layout Change Fraction")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved figure to {out_fig}")

def replay_real_hardware_p1():
    print("\n" + "=" * 80)
    print("REAL HARDWARE REPLAY CHECK: ibm_marrakesh measured p1")
    print("=" * 80)
    p1_files = glob.glob("results/hardware/p1_ibm_marrakesh_*.json")
    if not p1_files:
        print("No p1_ibm_marrakesh_*.json file found.")
        return
        
    p1_file = p1_files[0]
    with open(p1_file, "r") as f:
        data = json.load(f)
        
    p1_est = data.get("p1_estimate", {})
    ro_err = data.get("target_readout_error", {})
    
    hot_qubits = []
    for q_str, p1 in p1_est.items():
        q = int(q_str)
        ro = ro_err.get(q_str, 0.0)
        excess = p1 - ro
        if excess > 0.01:
            hot_qubits.append((q, p1, ro, excess))
            
    hot_qubits.sort(key=lambda x: x[3], reverse=True)
    print(f"Total physical qubits with excess_p1 > 0.01: {len(hot_qubits)}")
    print("Top 10 hottest physical qubits:")
    for q, p1, ro, exc in hot_qubits[:10]:
        print(f"  Qubit {q:3d}: measured p1 = {p1:.5f}, readout_error = {ro:.5f}, excess = {exc:.5f}")
        
    # Check layouts in hardware runtime table
    df_hw = pd.read_csv("results/hardware/runtime_table.csv")
    print("\nLayout membership check for hot qubits on ibm_marrakesh:")
    for (b, m), g in df_hw.groupby(['benchmark', 'method']):
        lay = json.loads(g['layout'].iloc[0])
        hot_in_lay = [q for q, _, _, _ in hot_qubits if q in lay]
        has_11 = 11 in lay
        print(f"  {b:7s} {m:16s} | layout={lay} | hot_qubits_in_layout={hot_in_lay} | has_qubit_11={has_11}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run thermal-gradient sensitivity sweep.")
    parser.add_argument("--seeds", type=int, default=10, help="Profile seeds per cell.")
    parser.add_argument("--shots", type=int, default=8192, help="Shots per simulation.")
    args = parser.parse_args()
    
    df_res, thresholds = run_thermal_sweep(num_seeds=args.seeds, shots=args.shots)
    plot_thermal_threshold(df_res)
    replay_real_hardware_p1()
    
    print("SWEEP THRESHOLD SUMMARY:")
    for dev, b_dict in thresholds.items():
        print(f"Device {dev}:")
        for b, thresh in b_dict.items():
            print(f"  - {b}: {thresh}")
