import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_random, mapper_qiskit_default, mapper_esp
from src.runner_ibm import get_ibm_service, get_backend, get_calibration_timestamp, check_pilot_and_approval, extract_qpu_seconds

def validate_isa(circuit, target):
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if not target.instruction_supported(op.name, qargs):
             raise ValueError(f"Instruction {op.name} on {qargs} not supported by target.")

def run_hardware(args):
    config = load_config()
    shots = config['experiment']['shots']
    
    if args.local:
        from qiskit.providers.fake_provider import GenericBackendV2
        backend = GenericBackendV2(num_qubits=127)
        backend_name = backend.name
        timestamp = "local_fake_timestamp"
    else:
        service = get_ibm_service()
        backend = get_backend(service, config)
        backend_name = backend.name
        timestamp = get_calibration_timestamp(backend)

    target = backend.target
    benchmarks = get_all_benchmarks(min(5, target.num_qubits))
    
    jobs_to_run = []
    
    for b_name, b_circ in benchmarks.items():
        tc_rand, _, _, lay_rand, _ = mapper_random(b_circ, target, num_draws=1, seed=42)[0]
        tc_l1 = mapper_qiskit_default(b_circ, target, level=1)
        tc_esp, _, _, lay_esp, _ = mapper_esp(b_circ, target, temps_mk=None, use_thermal=False, exhaustive=False)
        
        pm = generate_preset_pass_manager(target=target, optimization_level=0)
        isa_rand = pm.run(tc_rand)
        isa_l1 = pm.run(tc_l1)
        isa_esp = pm.run(tc_esp)
        
        validate_isa(isa_rand, target)
        validate_isa(isa_l1, target)
        validate_isa(isa_esp, target)
        
        jobs_to_run.extend([
            (b_name, 'random', isa_rand),
            (b_name, 'qiskit_L1', isa_l1),
            (b_name, 'esp', isa_esp)
        ])
        
    if args.pilot:
        b_name, m_name, circ = jobs_to_run[0]
        print(f"Running PILOT: 1 circuit ({b_name} - {m_name}) with {shots} shots...")
        
        sampler = Sampler(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        
        job = sampler.run([circ], shots=shots)
        print(f"Job ID: {job.job_id()}")
        job.result()
        usage = extract_qpu_seconds(job, 3.0)
        
        scale_data = {
            "timestamp": time.time(),
            "qpu_seconds_per_shot": usage / shots,
            "pilot_usage": usage,
            "pilot_shots": shots
        }
        os.makedirs("results/hardware", exist_ok=True)
        with open("results/hardware/qpu_scale.json", "w") as f:
            json.dump(scale_data, f, indent=2)
        print(f"Pilot scale saved. Estimated seconds per shot: {scale_data['qpu_seconds_per_shot']}")
        return
        
    scale_file = "results/hardware/qpu_scale.json"
    est_per_shot = 3.0 / 8192
    if os.path.exists(scale_file):
        with open(scale_file, "r") as f:
            est_per_shot = json.load(f).get("qpu_seconds_per_shot", est_per_shot)
            
    total_est = len(jobs_to_run) * shots * est_per_shot
    print(f"Total Circuits: {len(jobs_to_run)}")
    print(f"Estimated QPU Seconds: {total_est}")
    
    if args.execute:
        if not args.local:
            check_pilot_and_approval(total_est, args.approve_seconds)
            
        print("Executing jobs...")
        sampler = Sampler(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        
        results = []
        for b_name, m_name, circ in jobs_to_run:
            job = sampler.run([circ], shots=shots)
            res = job.result()
            usage = extract_qpu_seconds(job, shots * est_per_shot) if not args.local else 0.0
            
            results.append({
                "benchmark": b_name,
                "method": m_name,
                "job_id": job.job_id() if not args.local else "",
                "qpu_time": usage
            })
            
        with open("results/hardware/runtime_table.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Execution complete.")
    else:
        print("[DRY RUN] Use --execute --approve-seconds N to submit.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true')
    parser.add_argument('--pilot', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--approve-seconds', type=int)
    args = parser.parse_args()
    run_hardware(args)
