import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from qiskit_ibm_runtime.fake_provider import FakeVigoV2, FakeGuadalupeV2
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, DensityMatrix, hellinger_fidelity
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.noise_model.profile import load_config, get_thermal_profile
from src.noise_model.thermal import build_thermal_noise_model
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_random, mapper_qiskit_default, mapper_esp
from src.metrics.esp import compute_esp

def compute_fidelity(ideal_counts, noisy_counts):
    # normalize
    def norm(c):
        t = sum(c.values())
        return {k: v/t for k, v in c.items()}
    return hellinger_fidelity(norm(ideal_counts), norm(noisy_counts))

def run_phase_a():
    config = load_config()
    shots = config['experiment']['shots']
    
    # 5 qubit backend
    backend = FakeVigoV2()
    target = backend.target
    temps_mk = get_thermal_profile(target.num_qubits, config)
    
    noise_model = build_thermal_noise_model(target, temps_mk)
    sim_ideal = AerSimulator()
    sim_noisy = AerSimulator(noise_model=noise_model)
    
    benchmarks = get_all_benchmarks(target.num_qubits)
    
    results = []
    
    for b_name, b_circ in benchmarks.items():
        print(f"Running benchmark {b_name}...")
        # Ideal execution
        ideal_tc = mapper_qiskit_default(b_circ, target, level=0)
        ideal_counts = sim_ideal.run(ideal_tc, shots=shots).result().get_counts()
        
        # Mappers
        mappers_to_run = {
            'random': lambda c: mapper_random(c, target, num_draws=1, seed=42)[0],
            'qiskit_L0': lambda c: mapper_qiskit_default(c, target, level=0),
            'qiskit_L1': lambda c: mapper_qiskit_default(c, target, level=1),
            'qiskit_L3': lambda c: mapper_qiskit_default(c, target, level=3),
            'esp_no_thermal': lambda c: mapper_esp(c, target, temps_mk, use_thermal=False, exhaustive=True)[0],
            'esp_thermal': lambda c: mapper_esp(c, target, temps_mk, use_thermal=True, exhaustive=True)[0]
        }
        
        for m_name, m_func in mappers_to_run.items():
            mapped_circ = m_func(b_circ)
            
            # compute esp
            esp_std, esp_th = compute_esp(mapped_circ, target, temps_mk)
            
            # simulate
            noisy_counts = sim_noisy.run(mapped_circ, shots=shots).result().get_counts()
            fid = compute_fidelity(ideal_counts, noisy_counts)
            
            cx_count = mapped_circ.count_ops().get('cx', 0)
            
            results.append({
                'benchmark': b_name,
                'method': m_name,
                'fidelity': fid,
                'esp': esp_th,
                'esp_std': esp_std,
                'cx_count': cx_count,
                'depth': mapped_circ.depth()
            })
            
    df = pd.DataFrame(results)
    os.makedirs('results/sim', exist_ok=True)
    df.to_csv('results/sim/phase_a_results.csv', index=False)
    print("Phase A simulation complete. Results saved to results/sim/phase_a_results.csv")
    print(df.groupby(['benchmark', 'method'])[['fidelity', 'esp', 'cx_count']].mean())

if __name__ == "__main__":
    run_phase_a()
