import os
import sys
import time
import json
import random
import argparse
import pandas as pd
from pathlib import Path
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeGuadalupeV2
from qiskit_aer import AerSimulator
from qiskit.providers import QubitProperties

sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_all_benchmarks
from src.metrics.esp import load_config, count_native_2q, esp_standard, esp_thermal, esp_thermal_gate, get_active_qubits
from src.mappers.mappers import mapper_random, mapper_qiskit_default, mapper_esp
from src.noise_model.thermal import build_thermal_noise_model

def get_hot_qubits(target, fraction, seed):
    random.seed(seed)
    num = int(target.num_qubits * fraction)
    return random.sample(range(target.num_qubits), num) if num > 0 else []

def compact_circuit(tc, target, temps_mk, active_qubits):
    mapping = {q: i for i, q in enumerate(active_qubits)}
    compact_qc = QuantumCircuit(len(active_qubits))
    
    for inst in tc.data:
        if inst.operation.name in ['barrier', 'delay', 'measure']: continue
        qargs = [mapping[tc.find_bit(q).index] for q in inst.qubits]
        compact_qc.append(inst.operation, qargs)
        
    compact_temps = {mapping[q]: t for q, t in temps_mk.items() if q in active_qubits} if temps_mk else None
    
    # We build a Fake target for the active subset
    from qiskit.transpiler import Target, InstructionProperties
    compact_target = Target(num_qubits=len(active_qubits))
    
    gathered_props = {}
    for inst, qargs in target.instructions:
        inst_name = inst.name
        if inst_name in ['barrier', 'delay', 'measure', 'reset']: continue
        if qargs is None: continue
        
        if all(q in active_qubits for q in qargs):
            compact_qargs = tuple(mapping[q] for q in qargs)
            try:
                props = target[inst_name][qargs]
                if inst_name not in gathered_props: gathered_props[inst_name] = (inst, {})
                gathered_props[inst_name][1][compact_qargs] = props
            except KeyError: pass
            
    for inst_name, (inst, props_dict) in gathered_props.items():
        compact_target.add_instruction(inst, props_dict)
                
    measure_props = {}
    for q in active_qubits:
        try:
            measure_props[(mapping[q],)] = target['measure'][(q,)]
        except Exception: pass
        
        if getattr(target, 'qubit_properties', None):
            if compact_target.qubit_properties is None:
                compact_target.qubit_properties = [QubitProperties()] * len(active_qubits)
            compact_target.qubit_properties[mapping[q]] = target.qubit_properties[q]
    if measure_props:
        compact_target.add_instruction(target.operation_from_name('measure'), measure_props)
            
    compact_nm = build_thermal_noise_model(compact_target, temps_mk=compact_temps)
    compact_qc.measure_all()
    return compact_qc, compact_nm

def run_sim(resume=False):
    config = load_config()
    thermal_mode = config.get('experiment', {}).get('thermal_term_mode', 'raw_upper_bound')
    max_candidate_layouts = 200
    
    print("\n--- SYNTHETIC-TEMPERATURE DISCLAIMER ---")
    print("Temperatures in this sweep are synthetic and appear in both the objective and the noise model.")
    print("A simulated thermal win is by construction.\n")
    
    backends = [FakeVigoV2(), FakeGuadalupeV2()]
    Ns = [3, 4, 5]
    benchmarks_list = ['ghz', 'bv', 'qft', 'qaoa', 'routing']
    bg_temps = [15, 50, 80, 120]
    hot_fractions = [0, 0.2, 0.4]
    profile_seeds = [0, 1, 2]
    sim_seeds = list(range(20))
    
    out_file = Path('results/sim/phase_3_sweep.csv')
    os.makedirs(out_file.parent, exist_ok=True)
    
    done_keys = set()
    if resume and out_file.exists():
        df = pd.read_csv(out_file)
        if not df.empty:
            for _, row in df.iterrows():
                done_keys.add(f"{row['device']}_{row['N']}_{row['benchmark']}_{row['bg_T']}_{row['hot_fraction']}_{row['profile_seed']}_{row['method']}_{row['draw_id']}_{row['sim_seed']}")
            print(f"Resuming from {len(done_keys)} completed evaluations.")
    else:
        with open(out_file, 'w') as f:
            f.write("device,N,benchmark,bg_T,hot_fraction,hot_T,stress_test,profile_seed,hot_qubits,method,opt_level,thermal_term_mode,draw_id,sim_seed,metric_type,metric_val,esp_standard,esp_thermal,esp_thermal_gate,mapping_seconds,cx_count,depth,layout\n")

    # Time 20 transpiles for estimate
    print("Estimating runtime from 20 transpiles...")
    t0 = time.time()
    dummy_qc = get_all_benchmarks(5)['qft']
    for _ in range(20): transpile(dummy_qc, FakeGuadalupeV2(), optimization_level=1)
    transpile_time = (time.time() - t0) / 20.0
    print(f"Avg transpile time: {transpile_time:.4f}s")
    
    total_cells = len(backends) * len(Ns) * len(benchmarks_list)
    est_total = total_cells * (max_candidate_layouts * 3) * transpile_time
    print(f"Estimated mapping time: {est_total/60:.1f} minutes")
    if est_total > 25 * 60:
        max_candidate_layouts = int((25 * 60) / (total_cells * 3 * transpile_time))
        print(f"WARNING: Estimate > 25 mins. Reducing max_candidate_layouts to {max_candidate_layouts}")
        
    for backend in backends:
        target = backend.target
        for N in Ns:
            circuits = get_all_benchmarks(N)
            for b_name in benchmarks_list:
                if b_name not in circuits: continue
                qc = circuits[b_name]
                metric_type = 'success_prob' if b_name in ['ghz', 'bv'] else 'fidelity'
                
                cell_t0 = time.time()
                print(f"\nProcessing {backend.name} N={N} {b_name}...")
                
                # 1. Map once per cell
                c_rand_list = mapper_random(qc, target, num_draws=50, opt_level=3)
                c_l0 = mapper_qiskit_default(qc, target, level=0)
                c_l1 = mapper_qiskit_default(qc, target, level=1)
                c_l3 = mapper_qiskit_default(qc, target, level=3)
                c_esp_no_th = mapper_esp(qc, target, use_thermal=False, max_candidate_layouts=max_candidate_layouts, opt_level=3)
                c_esp_ablation = mapper_esp(qc, target, use_thermal=False, max_candidate_layouts=max_candidate_layouts, opt_level=1)
                
                # 2. Iterate thermal profiles
                for bg_T in bg_temps:
                    for hot_frac in hot_fractions:
                        for p_seed in profile_seeds:
                            hot_qs = get_hot_qubits(target, hot_frac, p_seed)
                            hot_T = min(bg_T * 3, 150)
                            stress = (hot_T > 80)
                            
                            temps_mk = {q: bg_T for q in range(target.num_qubits)}
                            for q in hot_qs: temps_mk[q] = hot_T
                            
                            # Map thermal esp
                            c_esp_th = mapper_esp(qc, target, temps_mk=temps_mk, use_thermal=True, max_candidate_layouts=max_candidate_layouts, opt_level=3)
                            
                            cached_metrics = {}
                            
                            methods = [
                                ('qiskit_L0', 0, [c_l0]),
                                ('qiskit_L1', 1, [c_l1]),
                                ('qiskit_L3', 3, [c_l3]),
                                ('esp_no_thermal', 3, [c_esp_no_th]),
                                ('esp_ablation', 1, [c_esp_ablation]),
                                ('esp_thermal', 3, [c_esp_th]),
                                ('random', 3, c_rand_list)
                            ]
                            
                            for m_name, o_lvl, draws in methods:
                                for draw_id, (tc, _, m_sec, lay, _) in enumerate(draws):
                                    # Simulate
                                    # Skip simulation if it's the identical circuit we already simulated for esp_no_thermal under THIS profile
                                    if m_name == 'esp_thermal' and tc == methods[3][2][0][0]:
                                        for s_seed in sim_seeds:
                                            key = f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_{m_name}_{draw_id}_{s_seed}"
                                            if key in done_keys: continue
                                            # Look up the metric_val we just wrote for esp_no_thermal
                                            ref_key = f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_esp_no_thermal_{draw_id}_{s_seed}"
                                            val = cached_metrics.get(ref_key, 0.0)
                                            row = f"{backend.name},{N},{b_name},{bg_T},{hot_frac},{hot_T},{stress},{p_seed},\"{hot_qs}\",{m_name},{o_lvl},{thermal_mode},{draw_id},{s_seed},{metric_type},{val},{esp_standard(tc, target)},{esp_thermal(tc, target, p1=temps_mk)},{esp_thermal_gate(tc, target, p1=temps_mk)},{m_sec},{count_native_2q(tc, target)},{tc.depth()},\"{lay}\"\n"
                                            with open(out_file, 'a') as f: f.write(row)
                                            done_keys.add(key)
                                        continue
                                        
                                    active_qs = list(get_active_qubits(tc))
                                    if not active_qs: active_qs = lay[:N]
                                    
                                    compact_qc, compact_nm = compact_circuit(tc, target, temps_mk, active_qs)
                                    sim = AerSimulator(noise_model=compact_nm)
                                    
                                    # Ideal
                                    ideal_tc = transpile(qc, backend, optimization_level=0)
                                    ideal_counts = AerSimulator().run(ideal_tc, shots=1000).result().get_counts()
                                    
                                    for s_seed in sim_seeds:
                                        key = f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_{m_name}_{draw_id}_{s_seed}"
                                        if key in done_keys: continue
                                        
                                        counts = sim.run(compact_qc, shots=1000, seed_simulator=s_seed).result().get_counts()
                                        
                                        if metric_type == 'success_prob':
                                            ideal_state = max(ideal_counts, key=ideal_counts.get)
                                            val = counts.get(ideal_state, 0) / sum(counts.values()) if counts else 0.0
                                        else:
                                            from qiskit.quantum_info import hellinger_fidelity
                                            def norm(c): return {k: v/sum(c.values()) for k, v in c.items()}
                                            val = hellinger_fidelity(norm(ideal_counts), norm(counts)) if counts else 0.0
                                            
                                        cached_metrics[key] = val
                                        row = f"{backend.name},{N},{b_name},{bg_T},{hot_frac},{hot_T},{stress},{p_seed},\"{hot_qs}\",{m_name},{o_lvl},{thermal_mode},{draw_id},{s_seed},{metric_type},{val},{esp_standard(tc, target)},{esp_thermal(tc, target, p1=temps_mk)},{esp_thermal_gate(tc, target, p1=temps_mk)},{m_sec},{count_native_2q(tc, target)},{tc.depth()},\"{lay}\"\n"
                                        
                                        with open(out_file, 'a') as f: f.write(row)
                                        done_keys.add(key)
                                        
                print(f"  Completed cell in {time.time() - cell_t0:.1f}s")
                                        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run_sim(resume=args.resume)
