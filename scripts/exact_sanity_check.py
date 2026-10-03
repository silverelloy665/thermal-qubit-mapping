import sys
from pathlib import Path
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit.quantum_info import Statevector, hellinger_fidelity

sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_random, mapper_qiskit_default, mapper_esp
from scripts.run_sim import compact_circuit
from src.metrics.esp import get_active_qubits

def get_ideal_probs(qc):
    qc_no_meas = qc.remove_final_measurements(inplace=False)
    meas_qargs = [None] * qc.num_clbits
    for inst in qc.data:
        if inst.operation.name == 'measure':
            for q, c in zip(inst.qubits, inst.clbits):
                meas_qargs[qc.find_bit(c).index] = qc.find_bit(q).index
    sv = Statevector(qc_no_meas)
    probs = sv.probabilities_dict(qargs=meas_qargs)
    return {k: v for k, v in probs.items() if v > 1e-10}

def total_variation_distance(p, q):
    keys = set(p.keys()) | set(q.keys())
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)

if __name__ == '__main__':
    backend = FakeVigoV2()
    target = backend.target
    
    print(f"{'Benchmark':<10} | {'Method':<20} | {'N':<3} | {'Min Fid (1 - H)':<15} | {'Max TVD':<15}")
    print("-" * 75)
    
    for N in [3, 4, 5]:
        circs = get_all_benchmarks(N)
        for b_name, qc in circs.items():
            ideal_probs = get_ideal_probs(qc)
            metric_type = 'success_prob' if b_name in ['ghz', 'bv'] else 'fidelity'
            
            c_l0 = mapper_qiskit_default(qc, target, level=0)
            c_l1 = mapper_qiskit_default(qc, target, level=1)
            c_l3 = mapper_qiskit_default(qc, target, level=3)
            c_esp_no_th = mapper_esp(qc, target, use_thermal=False, max_candidate_layouts=20, opt_level=3)
            c_esp_th = mapper_esp(qc, target, temps_mk={q: 15 for q in range(target.num_qubits)}, use_thermal=True, max_candidate_layouts=20, opt_level=3)
            c_esp_abl = mapper_esp(qc, target, use_thermal=False, max_candidate_layouts=20, opt_level=1)
            c_rand_list = mapper_random(qc, target, num_draws=20, opt_level=1)
            
            methods = [
                ('qiskit_L0', [c_l0]),
                ('qiskit_L1', [c_l1]),
                ('qiskit_L3', [c_l3]),
                ('esp_no_thermal', [c_esp_no_th]),
                ('esp_thermal', [c_esp_th]),
                ('esp_ablation', [c_esp_abl]),
                ('random', c_rand_list)
            ]
            
            for m_name, draws in methods:
                fidelities = []
                tvds = []
                for tc, _, _, lay, _ in draws:
                    active_qs = list(get_active_qubits(tc))
                    if not active_qs: active_qs = lay[:N]
                    compact_qc, _ = compact_circuit(tc, target, None, active_qs)
                    
                    actual_probs = get_ideal_probs(compact_qc)
                    
                    if metric_type == 'success_prob':
                        # Sum over support
                        val = sum(actual_probs.get(k, 0) for k in ideal_probs.keys())
                        fidelities.append(val)
                        # For TVD, just compare exactly to ideal
                        tvd = total_variation_distance(ideal_probs, actual_probs)
                        tvds.append(tvd)
                    else:
                        fid = hellinger_fidelity(ideal_probs, actual_probs)
                        tvd = total_variation_distance(ideal_probs, actual_probs)
                        fidelities.append(fid)
                        tvds.append(tvd)
                
                min_fid_diff = 1.0 - min(fidelities)
                max_tvd = max(tvds)
                
                assert min_fid_diff < 1e-9, f"{b_name} {m_name} failed fid diff {min_fid_diff} >= 1e-9"
                assert max_tvd < 1e-9, f"{b_name} {m_name} failed TVD {max_tvd} >= 1e-9"
                
                print(f"{b_name:<10} | {m_name:<20} | {N:<3} | {min_fid_diff:<15.1e} | {max_tvd:<15.1e}")
