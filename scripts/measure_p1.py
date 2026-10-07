import os
import sys
import json
import time
import math
import argparse
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from src.runner_ibm import get_ibm_service, get_backend, get_calibration_timestamp, check_pilot_and_approval, get_scale_estimate, extract_qpu_seconds

def validate_isa(circuit, target):
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if op.name not in target or qargs not in target[op.name]:
            if op.name != 'barrier' and op.name != 'measure':
                if not target.instruction_supported(op.name, qargs):
                     raise ValueError(f"Instruction {op.name} on {qargs} not supported by target.")

def run_measure_p1(args):
    config = load_config()
    region = config.get('experiment', {}).get('qubit_region', [0, 1, 2, 3])
    shots = 4096
    
    if args.local:
        from qiskit_ibm_runtime.fake_provider import FakeVigoV2
        backend = FakeVigoV2()
        timestamp = get_calibration_timestamp(backend)
    else:
        service = get_ibm_service()
        backend = get_backend(service, config)
        timestamp = get_calibration_timestamp(backend)
        
    target = backend.target
    
    # Check if region is within target
    region = [q for q in region if q < target.num_qubits]
    if not region:
        print("No valid qubits in region.")
        return

    # PUB 1: |0> state
    qc0 = QuantumCircuit(target.num_qubits, len(region))
    for i, q in enumerate(region):
        qc0.measure(q, i)
        
    # PUB 2: |1> state (X gate)
    qc1 = QuantumCircuit(target.num_qubits, len(region))
    for i, q in enumerate(region):
        qc1.x(q)
        qc1.measure(q, i)
        
    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    isa0 = pm.run(qc0)
    isa1 = pm.run(qc1)
    
    validate_isa(isa0, target)
    validate_isa(isa1, target)
    
    scale = get_scale_estimate()
    est_qpu = scale * shots * 2  # scale is per shot
    
    if args.execute or args.local:
        if not args.local:
            check_pilot_and_approval(est_qpu, args.approve_seconds)
            
        print(f"Submitting job to {backend.name}...")
        sampler = Sampler(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        
        job = sampler.run([isa0, isa1], shots=shots)
        print(f"Job ID: {job.job_id() if not args.local else ''}")
        result = job.result()
        qpu_time = extract_qpu_seconds(job, est_qpu) if not args.local else 0.0
        print(f"QPU time used: {qpu_time}")
        
        pub0_res = result[0].data
        pub1_res = result[1].data
        
        counts0 = pub0_res.c.get_counts() if hasattr(pub0_res, 'c') else pub0_res.meas.get_counts()
        counts1 = pub1_res.c.get_counts() if hasattr(pub1_res, 'c') else pub1_res.meas.get_counts()
        
        # Marginalize counts for each qubit
        p1_est = {}
        r_asym = {}
        t_err = {}
        
        for i, q in enumerate(region):
            # Qiskit counts are little-endian (rightmost bit is index 0)
            # Find prob of 1 in qc0 (P(1|0)) and prob of 0 in qc1 (P(0|1))
            c0_1 = sum(v for k, v in counts0.items() if k[-(i+1)] == '1') / shots
            c1_0 = sum(v for k, v in counts1.items() if k[-(i+1)] == '0') / shots
            
            p1_est[str(q)] = c1_0
            r_asym[str(q)] = c1_0 - c0_1
            
            err = 0.0
            if 'measure' in target and (q,) in target['measure']:
                prop = target['measure'][(q,)]
                if prop and getattr(prop, 'error', None):
                    err = prop.error
            t_err[str(q)] = err
            
        data = {
             "job_id": job.job_id() if not args.local else "",
             "timestamp": timestamp,
             "p1_estimate": p1_est,
             "readout_asymmetry": r_asym,
             "target_readout_error": t_err
        }
        
        os.makedirs("results/hardware", exist_ok=True)
        safe_ts = timestamp.replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")
        out_file = f"results/hardware/p1_{backend.name}_{safe_ts}.json"
        with open(out_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {out_file}")
    else:
        print(f"[DRY RUN] Would submit 2 PUBs to {backend.name} for {shots} shots each.")
        print(f"Estimated QPU seconds: {est_qpu}")
        if not args.local:
            print("Run with --execute and --approve-seconds N to submit.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help="Run on local fake backend")
    parser.add_argument('--execute', action='store_true', help="Execute on hardware")
    parser.add_argument('--approve-seconds', type=int, help="Approve QPU budget")
    args = parser.parse_args()
    run_measure_p1(args)
