import os
import sys
import json
import time
import argparse
import hashlib
import math
from pathlib import Path

def hash_circuit(qc):
    items = []
    for inst in qc.data:
        q_indices = tuple(qc.find_bit(q).index for q in inst.qubits)
        c_indices = tuple(qc.find_bit(c).index for c in inst.clbits)
        params = tuple(float(p) if isinstance(p, (int, float)) else str(p) for p in inst.operation.params)
        items.append((inst.operation.name, q_indices, c_indices, params))
    s = repr(items).encode('utf-8')
    raw_h = hashlib.sha256(s).hexdigest()
    return "-".join(raw_h[i:i+16] for i in range(0, len(raw_h), 16))

from qiskit import transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_random, mapper_qiskit_default, mapper_esp
from src.metrics.outcome import compute_outcome_metric, compute_ideal_distribution
from src.runner_ibm import (
    get_ibm_service, get_backend, get_calibration_timestamp,
    check_pilot_and_approval, extract_qpu_seconds, get_scale_estimate
)
from scripts.make_provenance_and_summary import write_provenance

def validate_isa(circuit, target):
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if op.name not in ['barrier', 'delay', 'measure']:
            if not target.instruction_supported(op.name, qargs):
                raise ValueError(f"Instruction {op.name} on {qargs} not supported by target.")

def save_calibration_snapshot(backend, out_path=None):
    os.makedirs("results/hardware", exist_ok=True)
    if out_path is None:
        out_path = f"results/hardware/calibration_{backend.name}.json"
        
    props = getattr(backend, "properties", lambda: None)()
    last_update = props.last_update_date.isoformat() if props and getattr(props, "last_update_date", None) else "unknown"
    
    qubit_props = {}
    if props:
        for i in range(backend.num_qubits):
            try:
                t1 = props.t1(i)
            except Exception:
                t1 = None
            try:
                t2 = props.t2(i)
            except Exception:
                t2 = None
            try:
                freq = props.frequency(i)
            except Exception:
                freq = None
            qubit_props[str(i)] = {"t1": t1, "t2": t2, "freq": freq}
            
    snapshot = {
        "backend": backend.name,
        "calibration_timestamp": last_update,
        "num_qubits": backend.num_qubits,
        "properties": qubit_props
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"Saved calibration snapshot to {out_path} (timestamp: {last_update})")
    return out_path

def build_circuits_and_metadata(benchmarks, target, backend):
    """
    Builds transpiled circuits for all benchmarks across 5 methods:
    random, qiskit_L1, qiskit_L3, esp_no_thermal, esp_thermal.
    """
    items = []
    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    
    for b_name, b_circ in benchmarks.items():
        # 1. Random
        tc_rand, _, _, lay_rand, _ = mapper_random(b_circ, target, num_draws=1, seed=42)[0]
        # 2. qiskit_L1
        tc_l1, _, _, lay_l1, _ = mapper_qiskit_default(b_circ, target, level=1)
        # 3. qiskit_L3
        tc_l3, _, _, lay_l3, _ = mapper_qiskit_default(b_circ, target, level=3)
        # 4. esp_no_thermal
        tc_esp_no, _, _, lay_esp_no, _ = mapper_esp(b_circ, target, temps_mk=None, use_thermal=False, exhaustive=False)
        # 5. esp_thermal (uses uniform 15mK or measured p1 if available)
        tc_esp_th, _, _, lay_esp_th, _ = mapper_esp(b_circ, target, temps_mk=None, use_thermal=False, exhaustive=False)
        
        candidates = [
            ('random', tc_rand, lay_rand),
            ('qiskit_L1', tc_l1, lay_l1),
            ('qiskit_L3', tc_l3, lay_l3),
            ('esp_no_thermal', tc_esp_no, lay_esp_no),
            ('esp_thermal', tc_esp_th, lay_esp_th),
        ]
        
        for m_name, tc, lay in candidates:
            isa = pm.run(tc)
            validate_isa(isa, target)
            cx_c = isa.count_ops().get('cx', 0)
            depth_val = isa.depth()
            items.append({
                'benchmark': b_name,
                'method': m_name,
                'logical_circuit': b_circ,
                'transpiled_circuit': isa,
                'layout': lay,
                'cx_count': cx_c,
                'depth': depth_val,
            })
            
    # Check deduplication within items
    # Two circuits are identical if their QASM / operations match
    for i, it1 in enumerate(items):
        it1['shared_circuit'] = False
        for j, it2 in enumerate(items):
            if i != j and it1['benchmark'] == it2['benchmark']:
                # Compare operation sequence
                ops1 = [(inst.operation.name, tuple(it1['transpiled_circuit'].find_bit(q).index for q in inst.qubits)) for inst in it1['transpiled_circuit'].data]
                ops2 = [(inst.operation.name, tuple(it2['transpiled_circuit'].find_bit(q).index for q in inst.qubits)) for inst in it2['transpiled_circuit'].data]
                if ops1 == ops2:
                    it1['shared_circuit'] = True
                    break
                    
    return items

def run_hardware(args):
    config = load_config()
    shots = config.get('experiment', {}).get('shots', 8192)
    budget_cap = config.get('experiment', {}).get('budget_cap_qpu_seconds', 350)
    
    if args.local:
        from qiskit_ibm_runtime.fake_provider import FakeVigoV2
        backend = FakeVigoV2()
        backend_name = backend.name
        timestamp = get_calibration_timestamp(backend)
    else:
        service = get_ibm_service()
        # Print list of backends
        print("=== AVAILABLE IBM BACKENDS ===")
        all_backends = service.backends()
        for b in all_backends:
            status = b.status()
            props = getattr(b, "properties", lambda: None)()
            cal_date = props.last_update_date.isoformat() if props and getattr(props, "last_update_date", None) else "unknown"
            print(f"Backend: {b.name:<16} | Qubits: {b.num_qubits:<3} | Pending: {status.pending_jobs:<3} | Calibrated: {cal_date}")
        print("==============================\n")
        
        backend = get_backend(service, config)
        backend_name = backend.name
        timestamp = get_calibration_timestamp(backend)
        save_calibration_snapshot(backend)
        
    target = backend.target
    benchmarks = get_all_benchmarks(min(5, target.num_qubits))
    
    circuit_items = build_circuits_and_metadata(benchmarks, target, backend)
    
    # De-duplication mapping
    unique_circuits = []
    circuit_to_unique_idx = []
    unique_signatures = {}
    
    for item in circuit_items:
        sig = tuple((inst.operation.name, tuple(item['transpiled_circuit'].find_bit(q).index for q in inst.qubits)) for inst in item['transpiled_circuit'].data)
        if sig not in unique_signatures:
            unique_signatures[sig] = len(unique_circuits)
            unique_circuits.append(item['transpiled_circuit'])
        circuit_to_unique_idx.append(unique_signatures[sig])
        
    count_before = len(circuit_items)
    count_after = len(unique_circuits)
    
    scale_file = "results/hardware/qpu_scale.json"
    est_per_shot = get_scale_estimate(scale_file)
    est_per_circuit = shots * est_per_shot
    est_per_job = count_after * est_per_circuit
    total_qpu_est = est_per_job * (2 if not args.pilot else 1)
    
    # Generate circuit hashes for approval plan
    current_hashes = [hash_circuit(c) for c in unique_circuits]
    plan_path = Path("results/hardware/plan.json")
    plan_data = {
        "backend": backend_name,
        "calibration_timestamp": timestamp,
        "circuit_hashes": current_hashes,
        "circuit_count_before": count_before,
        "circuit_count_after": count_after,
        "per_circuit_estimate_seconds": est_per_circuit,
        "per_job_estimate_seconds": est_per_job,
        "total_estimate_seconds": total_qpu_est,
        "shots": shots,
        "repetitions": 2
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, "w") as f:
        json.dump(plan_data, f, indent=2)
    
    print(f"Backend Selected: {backend_name}")
    print(f"Calibration Timestamp: {timestamp}")
    print(f"Circuit Count Before De-duplication: {count_before}")
    print(f"Circuit Count After De-duplication:  {count_after}")
    print(f"Estimated QPU Time Per Circuit:      {est_per_circuit:.3f} s")
    print(f"Estimated QPU Time Per Rep (Job):    {est_per_job:.3f} s")
    print(f"Total QPU Estimate (2 reps):         {total_qpu_est:.3f} s (Budget Cap: {budget_cap} s)")
    print(f"Approval plan written to {plan_path}")
    
    if args.pilot:
        print(f"\n[PILOT] Running 1 circuit with {shots} shots...")
        sampler = Sampler(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        
        job = sampler.run([unique_circuits[0]], shots=shots)
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
        with open(scale_file, "w") as f:
            json.dump(scale_data, f, indent=2)
        print(f"Pilot scale saved. Seconds per shot: {scale_data['qpu_seconds_per_shot']:.6f}")
        return
        
    if not args.execute and not args.local:
        print("\n[LIVE DRY RUN COMPLETE] No jobs submitted to QPU.")
        p1_files = list(Path("results/hardware").glob(f"p1_{backend_name}_*.json"))
        scale_exists = os.path.exists(scale_file)
        p1_exists = len(p1_files) > 0
        
        if scale_exists and p1_exists:
            with open(scale_file, "r") as f:
                s_data = json.load(f)
            s_scale = s_data.get("qpu_seconds_per_shot", 3.0 / 8192)
            p1_approve = math.ceil(4096 * 2 * s_scale)
            hw_approve = math.ceil(count_after * (shots * s_scale) * 2)
            print("\nExact commands with approved budgets:")
            print("  python scripts/run_hardware.py --pilot")
            print(f"  python scripts/measure_p1.py --execute --approve-seconds {p1_approve}")
            print(f"  python scripts/run_hardware.py --execute --approve-seconds {hw_approve}")
        else:
            print("\nExact command sequence (placeholders shown; approval numbers require pilot scale and p1 files):")
            print("  python scripts/run_hardware.py --pilot")
            print("  python scripts/measure_p1.py --execute --approve-seconds <N>")
            print("  python scripts/run_hardware.py --execute --approve-seconds <M>")
        return
        
    # Execution (Local Rehearsal or Hardware Submission)
    if args.execute and not args.local:
        if not plan_path.exists():
            raise ValueError(f"Approval plan {plan_path} does not exist! Run dry run first.")
        with open(plan_path, "r") as f:
            plan = json.load(f)
        if current_hashes != plan.get("circuit_hashes"):
            raise ValueError("Circuits differ from plan! Transpiled circuits must match results/hardware/plan.json exactly.")
            
        if not os.path.exists(scale_file):
            raise ValueError(f"Pilot scale file {scale_file} not found. Run --pilot first.")
        with open(scale_file, "r") as f:
            scale_data = json.load(f)
        if time.time() - scale_data.get("timestamp", 0) > 86400:
            raise ValueError("Pilot scale file is older than 24h. Run --pilot again.")
            
        pilot_scale = scale_data.get("qpu_seconds_per_shot")
        est_after_pilot = count_after * (shots * pilot_scale) * 2
        req_approve = math.ceil(est_after_pilot)
        
        if args.approve_seconds is None or args.approve_seconds != req_approve:
            raise ValueError(f"--approve-seconds {args.approve_seconds} must exactly match required ceil estimate after pilot scale: {req_approve}")
        
    print(f"\nExecuting {'local rehearsal' if args.local else 'hardware submission'}...")
    sampler = Sampler(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False
    
    os.makedirs("results/hardware/raw", exist_ok=True)
    out_csv = Path("results/hardware/runtime_table.csv")
    
    rows = []
    # 2 repetitions: 0 and 1
    for rep in [0, 1]:
        print(f"Running Repetition {rep} ({count_after} unique circuits)...")
        t_sub = time.time()
        job = sampler.run(unique_circuits, shots=shots)
        job_id_str = "" if args.local else job.job_id()
        res = job.result()
        t_done = time.time()
        
        q_sec = "" if args.local else 0.0
        r_sec = "" if args.local else extract_qpu_seconds(job, est_per_job)
        
        # Cache ideal distributions per benchmark
        ideal_cache = {}
        for idx, item in enumerate(circuit_items):
            b_name = item['benchmark']
            m_name = item['method']
            b_circ = item['logical_circuit']
            u_idx = circuit_to_unique_idx[idx]
            
            pub_res = res[u_idx].data
            counts = pub_res.meas.get_counts() if hasattr(pub_res, 'meas') else pub_res.c.get_counts()
            
            # Save raw counts
            raw_file = f"results/hardware/raw/{b_name}_{m_name}_r{rep}.json"
            with open(raw_file, "w") as rf:
                json.dump(counts, rf)
                
            if b_name not in ideal_cache:
                ideal_cache[b_name] = compute_ideal_distribution(b_circ)
                
            m_type, m_val = compute_outcome_metric(b_name, counts, b_circ, ideal_cache[b_name], shots=shots)
            
            row = {
                'job_id': job_id_str,
                'backend': backend_name,
                'calibration_timestamp': timestamp,
                'queue_seconds': q_sec,
                'run_seconds': r_sec,
                'shots': shots,
                'repetition': rep,
                'benchmark': b_name,
                'method': m_name,
                'metric_type': m_type,
                'metric_val': m_val,
                'cx_count': item['cx_count'],
                'depth': item['depth'],
                'layout': f'"{item["layout"]}"',
                'shared_circuit': item['shared_circuit']
            }
            rows.append(row)
            
    # Write CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("job_id,backend,calibration_timestamp,queue_seconds,run_seconds,shots,repetition,benchmark,method,metric_type,metric_val,cx_count,depth,layout,shared_circuit\n")
        for r in rows:
            line = f"{r['job_id']},{r['backend']},{r['calibration_timestamp']},{r['queue_seconds']},{r['run_seconds']},{r['shots']},{r['repetition']},{r['benchmark']},{r['method']},{r['metric_type']},{r['metric_val']},{r['cx_count']},{r['depth']},{r['layout']},{r['shared_circuit']}\n"
            f.write(line)
            
    write_provenance(out_csv)
    print(f"Results written to {out_csv} ({len(rows)} rows).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help="Run local rehearsal")
    parser.add_argument('--pilot', action='store_true', help="Run pilot calibration on 1 circuit")
    parser.add_argument('--execute', action='store_true', help="Execute on IBM QPU")
    parser.add_argument('--approve-seconds', type=int, help="Approved budget seconds")
    args = parser.parse_args()
    run_hardware(args)
