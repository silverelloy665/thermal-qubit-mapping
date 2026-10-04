import argparse
import time
import os
import random
import math
import numpy as np
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import GenericBackendV2

from src.benchmarks.circuits import get_all_benchmarks
from src.noise_model.thermal import build_thermal_noise_model
from src.metrics.esp import esp_standard, esp_thermal, esp_thermal_gate, count_native_2q
from src.mappers.mappers import (
    mapper_random, mapper_qiskit_default, mapper_esp, get_active_qubits, mapper_qiskit_best3
)

# Guadalupe fake
class FakeGuadalupeV2(GenericBackendV2):
    def __init__(self):
        super().__init__(
            num_qubits=16,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
            coupling_map=[
                [0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 5], [5, 3],
                [1, 4], [4, 1], [5, 8], [8, 5], [4, 7], [7, 4], [6, 7], [7, 6],
                [8, 9], [9, 8], [8, 11], [11, 8], [7, 10], [10, 7], [10, 12], [12, 10],
                [11, 14], [14, 11], [12, 13], [13, 12], [13, 14], [14, 13], [14, 15], [15, 14]
            ],
            seed=42
        )
        self.name = 'fake_guadalupe'

# Vigo fake
class FakeVigoV2(GenericBackendV2):
    def __init__(self):
        super().__init__(
            num_qubits=5,
            basis_gates=['cx', 'id', 'rz', 'sx', 'x'],
            coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1], [3, 4], [4, 3]],
            seed=42
        )
        self.name = 'fake_vigo'

def compact_circuit(tc, target, temps_mk, active_qubits):
    mapping = {q: i for i, q in enumerate(sorted(active_qubits))}
    compact_qc = QuantumCircuit(len(mapping), len(tc.clbits))
    mapping_c = {c: compact_qc.clbits[i] for i, c in enumerate(tc.clbits)}
    for inst in tc.data:
        new_qargs = [compact_qc.qubits[mapping[tc.find_bit(q).index]] for q in inst.qubits]
        new_cargs = [mapping_c[c] for c in inst.clbits]
        compact_qc.append(inst.operation, new_qargs, new_cargs)
    if temps_mk is None: temps_mk = {q: 15 for q in range(target.num_qubits)}
    compact_temps = {mapping[q]: temps_mk[q] for q in active_qubits}
    
    from qiskit.transpiler import Target
    compact_target = Target(num_qubits=len(mapping))
    for inst_name, inst_props in target.items():
        if inst_name in ['barrier', 'delay']: continue
        new_props = {}
        for qargs, props in inst_props.items():
            if all(q in mapping for q in qargs):
                new_qargs = tuple(mapping[q] for q in qargs)
                new_props[new_qargs] = props
        if new_props:
            compact_target.add_instruction(target.operation_from_name(inst_name), new_props)
                
    nm = build_thermal_noise_model(compact_target, temps_mk=compact_temps)
    return compact_qc, nm

def get_hot_qubits(target, fraction, seed):
    if fraction == 0: return []
    random.seed(seed)
    num_hot = max(1, int(target.num_qubits * fraction))
    return random.sample(range(target.num_qubits), num_hot)

def print_t3(b_name, methods, tc_list, backend, target):
    print(f"\n--- T3 DATA FOR VIGO N=4 {b_name.upper()} ---")
    for m_name, _, draws in methods:
        if m_name not in ['qiskit_L3', 'qiskit_L3_best3', 'esp_no_thermal', 'random']: continue
        if m_name == 'random':
            # mean properties
            d_list = []
            f_list = []
            nq_list = []
            esp_list = []
            for draw_id, (tc, _, m_sec, lay, _) in enumerate(draws):
                d_list.append(tc.depth())
                nq_list.append(count_native_2q(tc, target))
                esp_list.append(esp_standard(tc, target))
                # compute fidelity properly? We don't have fidelity here directly without sim, but wait, I can compute it later?
                # Actually, wait, T3 asks for mean fidelity. That comes from the simulation loop!
            print(f"random mean: depth={np.mean(d_list):.2f}, 2q={np.mean(nq_list):.2f}, esp={np.mean(esp_list):.4f}")
            continue
            
        tc = draws[0][0]
        lay = draws[0][3]
        depth = tc.depth()
        n2q = count_native_2q(tc, target)
        esp = esp_standard(tc, target)
        
        # scheduled duration approximation (sum of longest path? Just use duration of 2q gates)
        dur = 0
        from qiskit.transpiler.passes import ALAPScheduleAnalysis
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.instruction_durations import InstructionDurations
        durations = InstructionDurations.from_backend(backend)
        try:
            pm = PassManager([ALAPScheduleAnalysis(durations)])
            tc_sched = pm.run(tc)
            dur = tc_sched.duration
        except:
            dur = 0
            
        # per edge errors
        edges = set()
        for inst in tc.data:
            if len(inst.qubits) == 2:
                q0 = tc.find_bit(inst.qubits[0]).index
                q1 = tc.find_bit(inst.qubits[1]).index
                edges.add((q0, q1))
        
        errs = []
        for e in edges:
            err = getattr(target['cx'].get(e, None), 'error', 0.0)
            errs.append(f"{e}:{err:.4f}")
            
        print(f"{m_name}: layout={lay}, 2q={n2q}, depth={depth}, dur={dur}, errs=[{', '.join(errs)}], esp={esp:.4f}")

def run_sweep():
    # Setup configs
    configs = [
        {
            'device': FakeVigoV2(),
            'Ns': [3, 4, 5],
            'benchmarks': ['ghz', 'bv', 'qft', 'qaoa', 'routing', 'routing_old'],
            'bg_temps': [15],
            'hot_fractions': [0.0, 0.2],
            'profile_seeds': [0],
            'sim_seeds': list(range(20)),
            'max_candidate_layouts': 100
        },
        {
            'device': FakeGuadalupeV2(),
            'Ns': [5],
            'benchmarks': ['ghz', 'qft', 'routing'],
            'bg_temps': [15, 50, 80, 120],
            'hot_fractions': [0.2, 0.4],
            'profile_seeds': [0, 1, 2],
            'sim_seeds': list(range(20)),
            'max_candidate_layouts': 50
        }
    ]
    
    out_file = Path('results/sim/phase_b_sweep.csv')
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not out_file.exists():
        with open(out_file, 'w') as f:
            f.write("device,N,benchmark,bg_T,hot_frac,hot_T,stress,profile_seed,hot_qs,method,opt_level,thermal_mode,draw_id,sim_seed,metric_type,metric_val,esp_standard,esp_thermal,esp_thermal_gate,mapping_seconds,native_2q,depth,layout\n")
            
    print("=== EFFECTIVE CONFIG ===")
    for cfg in configs:
        print(f"Device: {cfg['device'].name} | Ns: {cfg['Ns']} | Benchmarks: {cfg['benchmarks']}")
        print(f"  Bg Temps: {cfg['bg_temps']} | Hot Fracs: {cfg['hot_fractions']} | Profile Seeds: {cfg['profile_seeds']}")
        print(f"  Sim Seeds: {len(cfg['sim_seeds'])} | Max Candidate Layouts: {cfg['max_candidate_layouts']}")
    print("========================\n")
            
    # Estimate time
    print("Estimating runtime from 20 transpiles...")
    vigo = FakeVigoV2()
    t0 = time.time()
    for _ in range(20):
        qc = get_all_benchmarks(5)['qft']
        transpile(qc, target=vigo.target, optimization_level=3)
    t_map = (time.time() - t0) / 20.0
    
    t0 = time.time()
    for _ in range(20):
        qc = get_all_benchmarks(5)['qft']
        tc = transpile(qc, target=vigo.target, optimization_level=3)
        sim = AerSimulator()
        sim.run(tc, shots=8192).result()
    t_sim = (time.time() - t0) / 20.0
    
    # Calculate total time
    total_cells = 0
    total_sims = 0
    for cfg in configs:
        b_count = len(cfg['benchmarks'])
        n_count = len(cfg['Ns'])
        prof_count = len(cfg['bg_temps']) * len(cfg['hot_fractions']) * len(cfg['profile_seeds'])
        cells = b_count * n_count * prof_count
        total_cells += cells
        # methods = L0, L1, L3, L3_best3, esp_no, esp_ab, esp_th, random(50) -> approx 57 layouts
        layouts_per_cell = 57
        total_sims += cells * layouts_per_cell * len(cfg['sim_seeds'])
        
    est_mins = (total_cells * t_map * 50 + total_sims * t_sim) / 60.0
    print(f"Total Estimated Time: {est_mins:.1f} minutes")
    
    if est_mins > 20:
        print("Estimate exceeds 20 minutes! Reducing max_candidate_layouts dynamically.")
        for c in configs: c['max_candidate_layouts'] = 20

    cached_metrics = {}
    ideal_probs_cache = {}
    
    done_keys = set()
    if out_file.exists():
        import pandas as pd
        df = pd.read_csv(out_file)
        for _, row in df.iterrows():
            key = f"{row['device']}_{row['N']}_{row['benchmark']}_{row['bg_T']}_{row['hot_frac']}_{row['profile_seed']}_{row['method']}_{row['draw_id']}_{row['sim_seed']}"
            done_keys.add(key)
            if row['method'] == 'esp_no_thermal':
                cached_metrics[key] = row['metric_val']
                
    for cfg in configs:
        backend = cfg['device']
        target = backend.target
        for N in cfg['Ns']:
            circuits = get_all_benchmarks(N)
            for b_name in cfg['benchmarks']:
                if b_name not in circuits: continue
                qc = circuits[b_name]
                metric_type = 'success_prob' if b_name in ['ghz', 'bv'] else 'fidelity'
                
                print(f"\nProcessing {backend.name} N={N} {b_name}...")
                
                # Maps
                num_rand_draws = 50
                c_rand_list = mapper_random(qc, target, num_draws=num_rand_draws, opt_level=3)
                c_l0 = mapper_qiskit_default(qc, target, level=0)
                c_l1 = mapper_qiskit_default(qc, target, level=1)
                c_l3 = mapper_qiskit_default(qc, target, level=3)
                c_l3_best3 = mapper_qiskit_best3(qc, target, level=3, routing_seeds=3)
                
                c_esp_no_th = mapper_esp(qc, target, use_thermal=False, max_candidate_layouts=cfg['max_candidate_layouts'], opt_level=3)
                c_esp_ablation = mapper_esp(qc, target, use_thermal=False, max_candidate_layouts=cfg['max_candidate_layouts'], opt_level=1)
                
                methods_base = [
                    ('qiskit_L0', 0, [c_l0]),
                    ('qiskit_L1', 1, [c_l1]),
                    ('qiskit_L3', 3, [c_l3]),
                    ('qiskit_L3_best3', 3, [c_l3_best3]),
                    ('esp_no_thermal', 3, [c_esp_no_th]),
                    ('esp_ablation', 1, [c_esp_ablation]),
                    ('random', 3, c_rand_list)
                ]
                
                if backend.name == 'fake_vigo' and N == 4 and b_name == 'qft':
                    print_t3(b_name, methods_base, [], backend, target)
                
                t3_acc = {'qiskit_L3': [], 'qiskit_L3_best3': [], 'esp_no_thermal': [], 'random': []}
                
                for bg_T in cfg['bg_temps']:
                    for hot_frac in cfg['hot_fractions']:
                        for p_seed in cfg['profile_seeds']:
                            cell_t0 = time.time()
                            hot_qs = get_hot_qubits(target, hot_frac, p_seed)
                            hot_T = min(bg_T * 3, 150)
                            stress = (hot_T > 80)
                            
                            temps_mk = {q: (hot_T if q in hot_qs else bg_T) for q in range(target.num_qubits)}
                            thermal_mode = 'raw_upper_bound'
                            
                            c_esp_th = mapper_esp(qc, target, temps_mk=temps_mk, use_thermal=True, max_candidate_layouts=cfg['max_candidate_layouts'], opt_level=3)
                            
                            methods = methods_base + [('esp_thermal', 3, [c_esp_th])]
                            
                            for m_name, o_lvl, draws in methods:
                                for draw_id, (tc, _, m_sec, lay, _) in enumerate(draws):
                                    # Fix 1: Explicit layout check and KeyError raise
                                    if m_name == 'esp_thermal' and lay == methods_base[4][2][0][3]:
                                        for s_seed in cfg['sim_seeds']:
                                            key = f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_{m_name}_{draw_id}_{s_seed}"
                                            if key in done_keys: continue
                                            ref_key = f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_esp_no_thermal_{draw_id}_{s_seed}"
                                            val = cached_metrics[ref_key]
                                            row = f"{backend.name},{N},{b_name},{bg_T},{hot_frac},{hot_T},{stress},{p_seed},\"{hot_qs}\",{m_name},{o_lvl},{thermal_mode},{draw_id},{s_seed},{metric_type},{val},{esp_standard(tc, target)},{esp_thermal(tc, target, p1=temps_mk)},{esp_thermal_gate(tc, target, p1=temps_mk)},{m_sec},{count_native_2q(tc, target)},{tc.depth()},\"{lay}\"\n"
                                            with open(out_file, 'a') as f: f.write(row)
                                            done_keys.add(key)
                                        continue
                                        
                                    active_qs = list(get_active_qubits(tc))
                                    if not active_qs: active_qs = lay[:N]
                                    compact_qc, compact_nm = compact_circuit(tc, target, temps_mk, active_qs)
                                    sim = AerSimulator(noise_model=compact_nm)
                                    
                                    if (b_name, N) not in ideal_probs_cache:
                                        from qiskit.quantum_info import Statevector
                                        qc_no_meas = qc.remove_final_measurements(inplace=False)
                                        sv = Statevector(qc_no_meas)
                                        probs = sv.probabilities_dict()
                                        ideal_probs_cache[(b_name, N)] = {k: v for k, v in probs.items() if v > 1e-10}
                                        
                                    ideal_probs = ideal_probs_cache[(b_name, N)]
                                    
                                    missing_seeds = [s for s in cfg['sim_seeds'] if f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_{m_name}_{draw_id}_{s}" not in done_keys]
                                    if not missing_seeds: continue
                                    
                                    # Batch execute
                                    qcs_to_run = [compact_qc] * len(missing_seeds)
                                    res = sim.run(qcs_to_run, shots=8192, seed_simulator=missing_seeds[0]).result()
                                    counts_list = res.get_counts()
                                    if len(missing_seeds) == 1: counts_list = [counts_list]
                                    
                                    for s_seed, counts in zip(missing_seeds, counts_list):
                                        key = f"{backend.name}_{N}_{b_name}_{bg_T}_{hot_frac}_{p_seed}_{m_name}_{draw_id}_{s_seed}"
                                        val = 0.0
                                        if metric_type == 'success_prob':
                                            target_str = "1" * qc.num_clbits
                                            if b_name == 'bv': target_str = "1" + "0"*(N-3) + "1" if N >= 3 else "1"
                                            val = counts.get(target_str, 0) / 8192.0
                                        else:
                                            from qiskit.quantum_info import hellinger_fidelity
                                            norm_c = {k: v/8192.0 for k, v in counts.items()}
                                            val = hellinger_fidelity(ideal_probs, norm_c)
                                            
                                        if m_name == 'esp_no_thermal':
                                            cached_metrics[key] = val
                                            
                                        if backend.name == 'fake_vigo' and N == 4 and b_name == 'qft' and m_name in t3_acc:
                                            t3_acc[m_name].append(val)
                                            
                                        row = f"{backend.name},{N},{b_name},{bg_T},{hot_frac},{hot_T},{stress},{p_seed},\"{hot_qs}\",{m_name},{o_lvl},{thermal_mode},{draw_id},{s_seed},{metric_type},{val},{esp_standard(tc, target)},{esp_thermal(tc, target, p1=temps_mk)},{esp_thermal_gate(tc, target, p1=temps_mk)},{m_sec},{count_native_2q(tc, target)},{tc.depth()},\"{lay}\"\n"
                                        with open(out_file, 'a') as f: f.write(row)
                                        done_keys.add(key)
                                        
                            print(f"  {bg_T}mK {hot_frac}hot prof={p_seed} -> {time.time() - cell_t0:.1f}s")
                            
                if backend.name == 'fake_vigo' and N == 4 and b_name == 'qft':
                    print("\n--- T3 MEAN FIDELITIES ---")
                    for m in ['qiskit_L3', 'qiskit_L3_best3', 'esp_no_thermal', 'random']:
                        mean_f = np.mean(t3_acc[m]) if t3_acc[m] else 0.0
                        print(f"{m} mean fidelity: {mean_f:.4f}")

if __name__ == '__main__':
    run_sweep()
