#!/usr/bin/env python3
"""
scripts/measure_p1_v2.py

Enhanced excited-state population characterization protocol (Checkpoint U).
Features:
- Simultaneous 156-qubit measurement (Block A, 2 reps)
- Graph-isolated measurement (Block B, distance >= 3 between measured qubits, 2 reps)
- Single-qubit isolated measurement for key qubits (Block C, qubits 125 and 22)
- Recorded P(1|0) and P(0|1) for every measurement
- Strict QPU safety gating (pilot, approve-seconds, reserve and cap checks)
- JSON output with Wilson 95% CIs and backend calibration cross-referencing.
"""

import os
import sys
import json
import time
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path

import rustworkx as rx
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from src.runner_ibm import (
    get_ibm_service, get_backend, get_calibration_timestamp,
    check_pilot_and_approval, get_scale_estimate, extract_qpu_seconds
)

def wilson_score_interval(k: int, n: int, z: float = 1.959963984540054):
    """Computes Wilson 95% confidence interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + (z ** 2) / n
    center = (p + (z ** 2) / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1.0 - p) / n + (z ** 2) / (4 * (n ** 2)))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)

def get_p1_qubit_sets(p1_dict, ro_dict, target=None):
    """
    Computes HOT, FAIL, and CONTROL qubit sets:
    - HOT: top 10 by old excess among ro <= 0.20
    - FAIL: [94, 130, 82, 102, 113]
    - CONTROL: 8 qubits with excess <= 0 and ro <= 0.02, spread over the chip
    """
    num_q = target.num_qubits if target else 156
    
    # 1. HOT qubits
    eligible_hot = []
    for q_str, p1_val in p1_dict.items():
        q = int(q_str)
        if q >= num_q:
            continue
        ro_val = ro_dict.get(q_str, 0.0)
        if ro_val <= 0.20:
            exc = max(0.0, p1_val - ro_val)
            eligible_hot.append((exc, q))
    eligible_hot.sort(reverse=True)
    hot_qubits = [q for _, q in eligible_hot[:10]]
    
    # 2. FAIL qubits
    raw_fail = [94, 130, 82, 102, 113]
    fail_qubits = [q for q in raw_fail if q < num_q]
    
    # 3. CONTROL qubits: excess <= 0 and ro <= 0.02
    control_candidates = []
    for q_str, p1_val in p1_dict.items():
        q = int(q_str)
        if q >= num_q or q in hot_qubits or q in fail_qubits:
            continue
        ro_val = ro_dict.get(q_str, 0.0)
        if (p1_val - ro_val) <= 0.0 and ro_val <= 0.02:
            control_candidates.append(q)
    
    control_candidates.sort()
    num_controls = min(8, len(control_candidates))
    if num_controls > 0:
        step = max(1, len(control_candidates) // num_controls)
        control_qubits = [control_candidates[i * step] for i in range(num_controls)]
    else:
        control_qubits = []

    return {
        "HOT": hot_qubits,
        "FAIL": fail_qubits,
        "CONTROL": control_qubits,
    }

def partition_isolated_batches(qubits, target, min_distance=3):
    """
    Partitions the given qubits into batches such that for any two qubits
    in the same batch, their shortest graph distance on target is >= min_distance.
    """
    if not qubits:
        return []
    
    cm = target.build_coupling_map()
    g = rx.PyGraph()
    g.add_nodes_from(range(target.num_qubits))
    for edge in cm.get_edges():
        if edge[0] < edge[1]:
            g.add_edge(edge[0], edge[1], 1)
    
    dist_matrix = rx.distance_matrix(g)
    
    batches = []
    for q in qubits:
        placed = False
        for b in batches:
            if all(dist_matrix[q][other] >= min_distance for other in b):
                b.append(q)
                placed = True
                break
        if not placed:
            batches.append([q])
            
    # Verify distance constraint on all batches
    for b_idx, b in enumerate(batches):
        for i in range(len(b)):
            for j in range(i + 1, len(b)):
                u, v = b[i], b[j]
                d = dist_matrix[u][v]
                if d < min_distance:
                    raise ValueError(f"Batch {b_idx} violation: distance({u}, {v}) = {d} < {min_distance}")
    
    return batches

def generate_isolated_batches(target, min_distance=3, base_p1_path="results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json"):
    """Generates isolated qubit batches for target using committed base p1 data."""
    p = Path(base_p1_path)
    if not p.exists():
        return []
    with open(p, "r", encoding="utf-8") as f:
        base_p1_data = json.load(f)
    p1_dict = base_p1_data.get("p1_estimate", {})
    ro_dict = base_p1_data.get("target_readout_error", {})
    qubit_sets = get_p1_qubit_sets(p1_dict, ro_dict, target=target)
    isolated_pool = list(dict.fromkeys(qubit_sets["HOT"] + qubit_sets["FAIL"] + qubit_sets["CONTROL"]))
    return partition_isolated_batches(isolated_pool, target, min_distance=min_distance)

def check_budget_refusal(total_est_seconds: float, config: dict):
    """Checks budget refusal gates against extra_run_cap_s and reserve_s."""
    extra_run_cap_s = float(config.get("extra_run_cap_s", 110.0))
    allowance_remaining = float(config.get("allowance_remaining_s", 279.0))
    reserve_s = float(config.get("reserve_s", 150.0))
    if total_est_seconds > extra_run_cap_s:
        return False, f"Refusal: Total estimated time ({total_est_seconds:.2f} s) exceeds extra_run_cap_s ({extra_run_cap_s:.2f} s)."
    if (allowance_remaining - total_est_seconds) < reserve_s:
        return False, f"Refusal: Insufficient allowance. Remaining ({allowance_remaining:.1f} s) - estimated spend ({total_est_seconds:.1f} s) < reserve ({reserve_s:.1f} s)."
    return True, "PASS"

def build_measurement_circuits(target, qubit_sets, isolated_batches):
    """
    Builds measurement circuit descriptors for blocks A, B, and C.
    Each item is a dict with metadata and QuantumCircuit object.
    """
    num_q = target.num_qubits
    circuits = []
    
    # Block A: All qubits simultaneously, 2 repetitions
    all_qubits = list(range(num_q))
    for rep in range(2):
        # A prep |0>
        qc0 = QuantumCircuit(num_q, num_q)
        for q in all_qubits:
            qc0.measure(q, q)
        circuits.append({
            "block": "A",
            "rep": rep,
            "prep": 0,
            "qubits": all_qubits,
            "circuit": qc0,
            "name": f"A_rep{rep}_prep0"
        })
        
        # A prep |1>
        qc1 = QuantumCircuit(num_q, num_q)
        for q in all_qubits:
            qc1.x(q)
            qc1.measure(q, q)
        circuits.append({
            "block": "A",
            "rep": rep,
            "prep": 1,
            "qubits": all_qubits,
            "circuit": qc1,
            "name": f"A_rep{rep}_prep1"
        })

    # Block B: Isolated batches (distance >= 3), 2 repetitions
    for b_idx, batch_qubits in enumerate(isolated_batches):
        for rep in range(2):
            # B prep |0>
            qc0 = QuantumCircuit(num_q, len(batch_qubits))
            for c_idx, q in enumerate(batch_qubits):
                qc0.measure(q, c_idx)
            circuits.append({
                "block": "B",
                "batch_id": b_idx,
                "rep": rep,
                "prep": 0,
                "qubits": batch_qubits,
                "circuit": qc0,
                "name": f"B_b{b_idx}_rep{rep}_prep0"
            })
            
            # B prep |1>
            qc1 = QuantumCircuit(num_q, len(batch_qubits))
            for c_idx, q in enumerate(batch_qubits):
                qc1.x(q)
                qc1.measure(q, c_idx)
            circuits.append({
                "block": "B",
                "batch_id": b_idx,
                "rep": rep,
                "prep": 1,
                "qubits": batch_qubits,
                "circuit": qc1,
                "name": f"B_b{b_idx}_rep{rep}_prep1"
            })

    # Block C: Single-qubit isolated measurement for qubit 125 and qubit 22
    single_qubits = [q for q in [125, 22] if q < num_q]
    for q in single_qubits:
        # C prep |0>
        qc0 = QuantumCircuit(num_q, 1)
        qc0.measure(q, 0)
        circuits.append({
            "block": "C",
            "qubit_id": q,
            "rep": 0,
            "prep": 0,
            "qubits": [q],
            "circuit": qc0,
            "name": f"C_q{q}_prep0"
        })
        
        # C prep |1>
        qc1 = QuantumCircuit(num_q, 1)
        qc1.x(q)
        qc1.measure(q, 0)
        circuits.append({
            "block": "C",
            "qubit_id": q,
            "rep": 0,
            "prep": 1,
            "qubits": [q],
            "circuit": qc1,
            "name": f"C_q{q}_prep1"
        })

    return circuits

def validate_isa(circuit, target):
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if op.name not in ['barrier', 'delay', 'measure']:
            if not target.instruction_supported(op.name, qargs):
                raise ValueError(f"Instruction {op.name} on {qargs} not supported by target.")

def run_measure_p1_v2(args):
    config = load_config()
    shots = 4096

    # 1. Backend resolution & pinning
    target_backend_name = config.get("extra_backend", "ibm_marrakesh")
    if args.local:
        if args.backend_override == "fake_marrakesh":
            from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
            backend = FakeMarrakesh()
        else:
            from qiskit_ibm_runtime.fake_provider import FakeVigoV2
            backend = FakeVigoV2()
        cal_timestamp = get_calibration_timestamp(backend)
    else:
        service = get_ibm_service()
        live_backends = [b.name for b in service.backends()]
        if target_backend_name not in live_backends:
            raise ValueError(f"Refusal: Pinned backend '{target_backend_name}' not in live backends list: {live_backends}")
        backend = service.backend(target_backend_name)
        cal_timestamp = get_calibration_timestamp(backend)

    target = backend.target

    # 2. Load latest committed p1 file
    p1_file = Path("results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json")
    if not p1_file.exists():
        raise FileNotFoundError(f"Base p1 calibration file not found: {p1_file}")
    with open(p1_file, "r", encoding="utf-8") as f:
        base_p1_data = json.load(f)

    p1_dict = base_p1_data.get("p1_estimate", {})
    ro_dict = base_p1_data.get("target_readout_error", {})

    # Compute qubit sets
    qubit_sets = get_p1_qubit_sets(p1_dict, ro_dict, target=target)
    
    # Assert expected hot qubits are present if running on Marrakesh scale
    if target.num_qubits >= 156:
        expected_hot = [125, 22, 26, 11, 145, 23, 76, 30]
        for eq in expected_hot:
            if eq not in qubit_sets["HOT"]:
                raise AssertionError(f"Expected hot qubit {eq} missing from HOT set: {qubit_sets['HOT']}")

    # Form isolated batches (distance >= 3)
    isolated_pool = list(dict.fromkeys(qubit_sets["HOT"] + qubit_sets["FAIL"] + qubit_sets["CONTROL"]))
    isolated_batches = partition_isolated_batches(isolated_pool, target, min_distance=3)

    # Build circuits
    circuit_descs = build_measurement_circuits(target, qubit_sets, isolated_batches)
    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    
    isa_circuits = []
    for item in circuit_descs:
        isa_qc = pm.run(item["circuit"])
        validate_isa(isa_qc, target)
        item["isa_circuit"] = isa_qc
        isa_circuits.append(isa_qc)

    # 3. Timing & Budget Accounting
    scale = get_scale_estimate()
    sec_per_pub = scale * shots
    total_est_seconds = len(circuit_descs) * sec_per_pub

    allowance_remaining = float(config.get("allowance_remaining_s", 279))
    reserve_s = float(config.get("reserve_s", 150))
    extra_run_cap_s = float(config.get("extra_run_cap_s", 110))

    print("=" * 80)
    print("ENHANCED P1 CHARACTERIZATION PROTOCOL (measure_p1_v2)")
    print("=" * 80)
    print(f"Backend:                  {backend.name} ({target.num_qubits} qubits)")
    print(f"Calibration Timestamp:    {cal_timestamp}")
    print(f"Shots per PUB:            {shots}")
    print(f"Total Circuits (PUBs):    {len(circuit_descs)}")
    print(f"  - Block A (all qubits): 4 circuits (2 reps x 2 preps)")
    print(f"  - Block B (isolated):   {len(isolated_batches) * 4} circuits ({len(isolated_batches)} batches x 2 reps x 2 preps)")
    print(f"  - Block C (single):     {len([q for q in [125, 22] if q < target.num_qubits]) * 2} circuits")
    print(f"Estimated QPU time / PUB: {sec_per_pub:.2f} s")
    print(f"Total Estimated QPU Time: {total_est_seconds:.2f} s")
    print(f"Config Allowance:         {allowance_remaining:.1f} s")
    print(f"Config Reserve:           {reserve_s:.1f} s")
    print(f"Config Run Cap:           {extra_run_cap_s:.1f} s")
    print("-" * 80)

    # Budget Refusal Gates
    if total_est_seconds > extra_run_cap_s:
        raise ValueError(
            f"Refusal: Total estimated time ({total_est_seconds:.2f} s) exceeds extra_run_cap_s ({extra_run_cap_s:.2f} s)."
        )

    if (allowance_remaining - total_est_seconds) < reserve_s:
        raise ValueError(
            f"Refusal: Insufficient allowance. Remaining ({allowance_remaining:.1f} s) - "
            f"estimated spend ({total_est_seconds:.1f} s) < reserve ({reserve_s:.1f} s)."
        )

    print("Budget Gate: PASS (Estimate within cap and preserves reserve allowance).")

    if not args.execute and not args.local:
        print("\n[DRY RUN ONLY] No jobs submitted to hardware.")
        print(f"To execute on live QPU, run with: --execute --approve-seconds {math.ceil(total_est_seconds)}")
        return

    # Execution Mode
    if not args.local:
        check_pilot_and_approval(total_est_seconds, args.approve_seconds)

    print(f"\nSubmitting {len(isa_circuits)} measurement PUBs to {backend.name}...")
    
    if args.local:
        # Mock results for local execution / tests
        job_id = "local_test_p1v2"
        qpu_time = 0.0
        # Generate deterministic synthetic counts
        results_data = []
        for desc in circuit_descs:
            prep = desc["prep"]
            qubits = desc["qubits"]
            # Target ideal bitstring: all 0 if prep=0 else all 1
            if prep == 0:
                counts = {"0" * len(qubits): int(shots * 0.98), "1" * len(qubits): int(shots * 0.02)}
            else:
                counts = {"1" * len(qubits): int(shots * 0.97), "0" * len(qubits): int(shots * 0.03)}
            results_data.append(counts)
    else:
        sampler = Sampler(mode=backend)
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        job = sampler.run(isa_circuits, shots=shots)
        job_id = job.job_id()
        print(f"Job ID: {job_id}")
        job_res = job.result()
        qpu_time = extract_qpu_seconds(job, total_est_seconds)
        results_data = []
        for i in range(len(circuit_descs)):
            pub_data = job_res[i].data
            c_dict = pub_data.c.get_counts() if hasattr(pub_data, 'c') else pub_data.meas.get_counts()
            results_data.append(c_dict)

    # 4. Extract per-qubit, per-block metrics
    exec_timestamp = datetime.now(timezone.utc).isoformat()
    
    # Check calibrated backend properties
    cal_p10 = {}
    cal_p01 = {}
    cal_ro = {}
    has_cal_asym = True
    
    props = getattr(backend, "properties", lambda: None)()
    for q in range(target.num_qubits):
        p10 = None
        p01 = None
        ro = 0.0
        if props:
            p10 = props.prob_meas1_prep0(q) if hasattr(props, 'prob_meas1_prep0') else None
            p01 = props.prob_meas0_prep1(q) if hasattr(props, 'prob_meas0_prep1') else None
            ro = props.readout_error(q) if hasattr(props, 'readout_error') else 0.0
        else:
            m_prop = target['measure'].get((q,), None)
            ro = getattr(m_prop, 'error', 0.0) if m_prop else 0.0

        if p10 is None:
            has_cal_asym = False

        cal_p10[str(q)] = p10
        cal_p01[str(q)] = p01
        cal_ro[str(q)] = ro

    # Structure data payload
    output_payload = {
        "job_id": job_id,
        "backend": backend.name,
        "execution_timestamp": exec_timestamp,
        "calibration_timestamp": cal_timestamp,
        "shots_per_pub": shots,
        "qpu_seconds_estimated": total_est_seconds,
        "qpu_seconds_measured": qpu_time,
        "calibrated_asymmetry_exposed": has_cal_asym,
        "target_calibrated_ro": cal_ro,
        "target_calibrated_prob_meas1_prep0": cal_p10,
        "target_calibrated_prob_meas0_prep1": cal_p01,
        "qubit_sets": qubit_sets,
        "isolated_batches": isolated_batches,
        "measurements": {}
    }

    # Populate per-block measurement results
    # Aggregate counts by block and prep
    for item_idx, desc in enumerate(circuit_descs):
        counts = results_data[item_idx]
        block = desc["block"]
        prep = desc["prep"]
        rep = desc["rep"]
        qubits = desc["qubits"]
        m_name = desc["name"]

        for bit_pos, q in enumerate(qubits):
            # Qiskit little-endian index
            idx = -(bit_pos + 1)
            count_0 = sum(v for k, v in counts.items() if k[idx] == '0')
            count_1 = sum(v for k, v in counts.items() if k[idx] == '1')
            
            p_1 = count_1 / shots
            p_0 = count_0 / shots
            
            ci_low, ci_high = wilson_score_interval(count_1 if prep == 0 else count_0, shots)

            q_key = str(q)
            if q_key not in output_payload["measurements"]:
                output_payload["measurements"][q_key] = {}
            if block not in output_payload["measurements"][q_key]:
                output_payload["measurements"][q_key][block] = {}
            
            sub_key = f"rep{rep}" if block != "B" else f"batch{desc['batch_id']}_rep{rep}"
            if sub_key not in output_payload["measurements"][q_key][block]:
                output_payload["measurements"][q_key][block][sub_key] = {}

            if prep == 0:
                output_payload["measurements"][q_key][block][sub_key]["P10"] = p_1
                output_payload["measurements"][q_key][block][sub_key]["P10_count"] = count_1
                output_payload["measurements"][q_key][block][sub_key]["P10_wilson_ci"] = [ci_low, ci_high]
            else:
                output_payload["measurements"][q_key][block][sub_key]["P01"] = p_0
                output_payload["measurements"][q_key][block][sub_key]["P01_count"] = count_0
                output_payload["measurements"][q_key][block][sub_key]["P01_wilson_ci"] = [ci_low, ci_high]

    # Save to results/hardware_v2/
    out_dir = Path("results/hardware_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = exec_timestamp.replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")
    out_file = out_dir / f"p1v2_{backend.name}_{safe_ts}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] p1v2 data saved to {out_file}")

    return str(out_file)

def main():
    parser = argparse.ArgumentParser(description="Measure QPU P1 (v2) Protocol")
    parser.add_argument("--local", action="store_true", help="Run with local fake backend")
    parser.add_argument("--backend-override", type=str, default="fake_vigo", help="fake_vigo or fake_marrakesh for local runs")
    parser.add_argument("--execute", action="store_true", help="Execute on live hardware")
    parser.add_argument("--approve-seconds", type=int, default=None, help="Approve QPU execution budget")
    args = parser.parse_args()

    run_measure_p1_v2(args)

if __name__ == "__main__":
    main()

