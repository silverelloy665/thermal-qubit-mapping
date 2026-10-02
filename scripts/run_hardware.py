import os
import sys
import time
import pandas as pd
from pathlib import Path
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_random, mapper_qiskit_default, mapper_esp

def run_hardware(dry_run=True):
    config = load_config()
    budget = config['experiment']['budget_cap_qpu_seconds']
    backend_name = config['backends']['real_5q']
    shots = config['experiment']['shots']

    print(f"Connecting to IBM Quantum Runtime (backend: {backend_name})...")
    token = os.getenv("QISKIT_IBM_TOKEN")
    if token:
        service = QiskitRuntimeService(channel="ibm_quantum", token=token)
    else:
        try:
            service = QiskitRuntimeService(channel="ibm_quantum")
        except Exception as e:
            print(f"CRITICAL ERROR: No IBM credentials found. Configure QISKIT_IBM_TOKEN or save an account.")
            sys.exit(1)

    backend = service.backend(backend_name)
    target = backend.target

    benchmarks = get_all_benchmarks(min(5, target.num_qubits))

    jobs_to_run = []
    
    for b_name, b_circ in benchmarks.items():
        print(f"Mapping {b_name}...")
        tc_rand = mapper_random(b_circ, target, num_draws=1, seed=42)[0]
        tc_l1 = mapper_qiskit_default(b_circ, target, level=1)
        tc_l3 = mapper_qiskit_default(b_circ, target, level=3)
        tc_esp, _ = mapper_esp(b_circ, target, temps_mk=None, use_thermal=False, exhaustive=False)

        pm = generate_preset_pass_manager(target=target, optimization_level=0)
        jobs_to_run.extend([
            (b_name, 'random', pm.run(tc_rand)),
            (b_name, 'qiskit_L1', pm.run(tc_l1)),
            (b_name, 'qiskit_L3', pm.run(tc_l3)),
            (b_name, 'esp', pm.run(tc_esp))
        ])

    estimated_seconds_per_circuit = 3 # Approx 3s for 8k shots on small depth
    total_estimate = len(jobs_to_run) * estimated_seconds_per_circuit
    
    print(f"\n--- QPU Budget Estimate ---")
    print(f"Total circuits: {len(jobs_to_run)}")
    print(f"Estimated QPU seconds: {total_estimate}")
    print(f"Budget Cap: {budget} seconds")

    if total_estimate > budget:
        print("ERROR: Estimated time exceeds QPU budget! Aborting.")
        sys.exit(1)

    if dry_run:
        print("Dry run complete. No jobs submitted. Run with 'python scripts/run_hardware.py execute' to submit.")
        return

    print("\nSubmitting jobs...")
    sampler = Sampler(mode=backend)
    
    results = []
    total_qpu_used = 0.0
    
    for b_name, m_name, circ in jobs_to_run:
        print(f"Submitting {b_name} - {m_name}")
        t_submit = time.time()
        # Ensure twirling / dynamical decoupling are off for clean comparison
        # (SamplerV2 defaults are fine, but can be explicitly set in options)
        job = sampler.run([circ], shots=shots)
        
        # We wait for it here to get precise queue/exec times in order. (Could be async but simpler this way)
        res = job.result()
        t_done = time.time()
        
        usage = job.usage()
        qpu_time = usage.quantum_seconds if hasattr(usage, 'quantum_seconds') else 0.0
        total_qpu_used += qpu_time
        
        results.append({
            'benchmark': b_name,
            'method': m_name,
            'job_id': job.job_id(),
            'qpu_time': qpu_time,
            'wall_time': t_done - t_submit,
            'cx_count': circ.count_ops().get('cx', 0),
            'depth': circ.depth()
        })
        
    df = pd.DataFrame(results)
    os.makedirs('results/hardware', exist_ok=True)
    df.to_csv('results/hardware/runtime_table.csv', index=False)
    print(f"Hardware run complete. Cumulative QPU seconds used: {total_qpu_used}")
    print("Results saved to results/hardware/runtime_table.csv")

if __name__ == "__main__":
    dry_run = len(sys.argv) < 2 or sys.argv[1] != 'execute'
    run_hardware(dry_run=dry_run)
