#!/usr/bin/env python3
"""
Populate absolute mean scores (qiskit_L3, esp_no_thermal, esp_thermal) for all cells
in results/sim/thermal_sweep.csv using the identical deterministic seed protocol.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeGuadalupeV2
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_qiskit_default, mapper_esp
from src.metrics.outcome import compute_outcome_metric
from scripts.run_thermal_sweep import simulate_with_mixture_compact
from scripts.make_provenance_and_summary import write_provenance

def main():
    devices = {
        'fake_vigo': FakeVigoV2(),
        'fake_guadalupe': FakeGuadalupeV2()
    }
    benchmarks_dict = get_all_benchmarks(5)
    bench_names = ['ghz', 'bv', 'qft', 'mirror']
    excess_grid = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15]
    hot_count_grid = [1, 2, 3]
    num_seeds = 10
    shots = 8192

    df_existing = pd.read_csv("results/sim/thermal_sweep.csv")
    
    means_l3 = []
    means_no = []
    means_th = []
    
    cell_idx = 0
    t0 = time.time()
    
    for dev_name, dev in devices.items():
        target = dev.target
        num_qubits = target.num_qubits
        
        for b_name in bench_names:
            qc = benchmarks_dict[b_name]
            tc_esp_default, _, _, lay_esp_default, _ = mapper_esp(
                qc, target, use_thermal=False, routing_seeds=1, max_candidate_layouts=30
            )
            tc_l3_default, _, _, lay_l3_default, _ = mapper_qiskit_default(qc, target, level=3)
            
            for k_hot in hot_count_grid:
                if k_hot > num_qubits:
                    continue
                for excess_p1 in excess_grid:
                    cell_idx += 1
                    scores_l3 = []
                    scores_no = []
                    
                    for s in range(num_seeds):
                        seed_val = 1000 * cell_idx + s
                        rng_draw = np.random.default_rng(seed_val)
                        
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
                        sim_seed = 42 + s
                        
                        cnt_esp_no = simulate_with_mixture_compact(tc_esp_default, target, lay_esp_default, true_excess_map, shots, sim_seed)
                        cnt_l3 = simulate_with_mixture_compact(tc_l3_default, target, lay_l3_default, true_excess_map, shots, sim_seed)
                        
                        _, score_no = compute_outcome_metric(b_name, cnt_esp_no, qc, shots=shots)
                        _, score_l3 = compute_outcome_metric(b_name, cnt_l3, qc, shots=shots)
                        
                        scores_no.append(score_no)
                        scores_l3.append(score_l3)
                        
                    mean_l3 = float(np.mean(scores_l3))
                    mean_no = float(np.mean(scores_no))
                    
                    # From exact paired gain in existing table:
                    paired_gain = df_existing.loc[cell_idx - 1, 'paired_gain_mean']
                    mean_th = mean_no + paired_gain
                    
                    means_l3.append(mean_l3)
                    means_no.append(mean_no)
                    means_th.append(mean_th)
                    
                    if cell_idx % 20 == 0 or cell_idx == len(df_existing):
                        print(f"[{cell_idx}/{len(df_existing)}] {dev_name:<14} {b_name:<6} k={k_hot} excess={excess_p1:.3f} | L3={mean_l3:.4f} ESP_no={mean_no:.4f} ESP_th={mean_th:.4f} (elapsed: {time.time()-t0:.1f}s)")
                        sys.stdout.flush()

    df_existing['mean_score_l3'] = means_l3
    df_existing['mean_score_esp_no'] = means_no
    df_existing['mean_score_esp_th'] = means_th
    
    out_csv = Path("results/sim/thermal_sweep.csv")
    df_existing.to_csv(out_csv, index=False)
    print(f"\nUpdated {out_csv} with mean_score_l3, mean_score_esp_no, mean_score_esp_th.")
    write_provenance(out_csv)
    print("Updated provenance sidecar.")

if __name__ == "__main__":
    main()

