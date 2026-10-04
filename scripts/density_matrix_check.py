import sys
from pathlib import Path
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit_aer import AerSimulator
from qiskit.quantum_info import hellinger_fidelity

sys.path.append(str(Path(__file__).parent.parent))
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_qiskit_default
from scripts.run_sim import compact_circuit
from scripts.exact_sanity_check import get_ideal_probs
from src.metrics.esp import get_active_qubits
from src.noise_model.thermal import build_thermal_noise_model

def norm(c): return {k: v/sum(c.values()) for k, v in c.items()}

if __name__ == '__main__':
    backend = FakeVigoV2()
    target = backend.target
    qc = get_all_benchmarks(3)['qft']
    ideal_probs = get_ideal_probs(qc)
    
    tc, _, _, lay, _ = mapper_qiskit_default(qc, target, level=1)
    temps = {q: 15 for q in range(target.num_qubits)}
    temps[lay[1]] = 100
    
    active_qs = list(get_active_qubits(tc))
    compact_qc, compact_nm = compact_circuit(tc, target, temps, active_qs)
    
    # Add save_probabilities
    compact_qc.save_probabilities_dict()
    
    sim = AerSimulator(noise_model=compact_nm, method='density_matrix')
    result = sim.run(compact_qc, shots=8192).result()
    
    # density matrix exact probabilities
    exact_noisy_probs = result.data(0)['probabilities']
    n_clbits = compact_qc.num_clbits
    
    formatted_exact = {}
    for k, v in exact_noisy_probs.items():
        if isinstance(k, str) and k.startswith('0x'):
            bit_str = bin(int(k, 16))[2:].zfill(n_clbits)
        else:
            bit_str = bin(k)[2:].zfill(n_clbits)
        if v > 1e-6:
            formatted_exact[bit_str] = v
            
    exact_noisy_probs = formatted_exact
    
    # sampling
    counts = result.get_counts()
    sampled_probs = norm(counts)
    
    print("Exact Noisy Hellinger vs Ideal:", hellinger_fidelity(ideal_probs, exact_noisy_probs))
    print("Sampled Noisy Hellinger vs Ideal:", hellinger_fidelity(ideal_probs, sampled_probs))
    diff = abs(hellinger_fidelity(ideal_probs, exact_noisy_probs) - hellinger_fidelity(ideal_probs, sampled_probs))
    print("Difference:", diff)
