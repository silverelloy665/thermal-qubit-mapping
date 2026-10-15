#!/usr/bin/env python3
"""
scripts/measure_p1_v2.py

Enhanced excited-state population characterization protocol (Checkpoint U & Z).
Features:
- Parameterized backend: ibm_marrakesh or ibm_fez (Checkpoint Z2)
- Multi-stage execution: full (24 PUBs), survey (4 PUBs), confirm (16 PUBs) (Checkpoint Z3)
- Recorded P(1|0) and P(0|1) for every measurement
- Strict QPU safety gating (pilot per backend, approve-seconds, reserve and cap checks)
- JSON output with Wilson 95% CIs and backend calibration cross-referencing.
"""

import os
import sys
import glob
import json
import time
import math
import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml
import rustworkx as rx
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from scripts.budget_table import compute_z4_cumulative_budget, format_z4_budget_table
from src.runner_ibm import (
    get_ibm_service, get_backend, get_calibration_timestamp,
    check_pilot_and_approval, get_scale_estimate, extract_qpu_seconds
)

def parse_iso_ts(ts_str):
    try:
        return datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

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
    Computes HOT, FAIL, and CONTROL qubit sets for primary full characterization:
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

def get_survey_qubit_sets(survey_data, target=None):
    """
    Extracts Candidate HOT and CONTROL qubits from survey data (Checkpoint Z3):
    1. Candidate Hot: Top 10 qubits ranked by survey excess satisfying:
       (a) Calibrated readout error ro <= 0.20
       (b) Survey excess > 0.05 in BOTH rep0 AND rep1.
    2. CONTROL: Exactly 8 qubits spread topologically across the lattice with:
       (a) Survey excess <= 0.0
       (b) Calibrated readout error ro <= 0.02.
    """
    num_q = target.num_qubits if target else 156
    cal_ro = survey_data.get("target_calibrated_ro", {})
    cal_p10 = survey_data.get("target_calibrated_prob_meas1_prep0", {})
    measurements = survey_data.get("measurements", {})

    hot_candidates = []
    control_candidates = []

    for q_str, m_blocks in measurements.items():
        q = int(q_str)
        if q >= num_q:
            continue
        block_a = m_blocks.get("A", {})
        r0 = block_a.get("rep0", {})
        r1 = block_a.get("rep1", {})
        p10_0 = r0.get("P10", 0.0)
        p10_1 = r1.get("P10", 0.0)

        ro_val = cal_ro.get(q_str, 0.0)
        base_p10 = cal_p10.get(q_str) if cal_p10.get(q_str) is not None else ro_val

        exc_0 = p10_0 - base_p10
        exc_1 = p10_1 - base_p10
        exc_avg = (exc_0 + exc_1) / 2.0

        if ro_val <= 0.20 and exc_0 > 0.05 and exc_1 > 0.05:
            hot_candidates.append((exc_avg, q))
        elif exc_avg <= 0.0 and ro_val <= 0.02:
            control_candidates.append(q)

    hot_candidates.sort(reverse=True)
    hot_qubits = [q for _, q in hot_candidates[:10]]

    control_candidates.sort()
    num_controls = min(8, len(control_candidates))
    if num_controls > 0:
        step = max(1, len(control_candidates) // num_controls)
        control_qubits = [control_candidates[i * step] for i in range(num_controls)]
    else:
        # Fallback controls if none strictly <= 0
        all_sorted = sorted(
            [int(q_str) for q_str in measurements.keys() if int(q_str) not in hot_qubits],
            key=lambda q: cal_ro.get(str(q), 0.0)
        )
        control_qubits = all_sorted[:8]

    return {
        "HOT": hot_qubits,
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

def build_measurement_circuits(target, qubit_sets, isolated_batches, stage="full"):
    """
    Builds measurement circuit descriptors for the requested stage:
    - full: Block A (all qubits, 4 PUBs), Block B (isolated, distance >= 3), Block C (single key qubits)
    - survey: Block A only (all qubits simultaneously, 2 reps x 2 preps = 4 PUBs)
    - confirm: Block B only on target set (isolated batches, 2 reps x 2 preps)
    """
    num_q = target.num_qubits
    circuits = []
    
    # 1. Block A: All qubits simultaneously, 2 repetitions
    if stage in ["full", "survey"]:
        all_qubits = list(range(num_q))
        for rep in range(2):
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

    # 2. Block B: Isolated batches (distance >= 3), 2 repetitions
    if stage in ["full", "confirm"]:
        for b_idx, batch_qubits in enumerate(isolated_batches):
            for rep in range(2):
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

    # 3. Block C: Single-qubit isolated measurement for qubit 125 and qubit 22 (full stage only)
    if stage == "full":
        single_qubits = [q for q in [125, 22] if q < num_q]
        for q in single_qubits:
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
    # Config loading & isolation
    cfg_override = getattr(args, "config", None)
    if cfg_override:
        with open(cfg_override, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        config = load_config()

    shots = 4096
    stage = getattr(args, "stage", "full")

    # 1. Backend resolution & validation (Checkpoint Z2)
    backend_cli = getattr(args, "backend", None)
    target_backend_name = backend_cli if backend_cli else config.get("extra_backend", "ibm_marrakesh")
    allowed_backends = ["ibm_marrakesh", "ibm_fez", "fake_marrakesh", "fake_fez"]
    if target_backend_name not in allowed_backends:
        raise ValueError(
            f"Refusal: Backend '{target_backend_name}' is not supported. Supported backends are ibm_marrakesh, ibm_fez."
        )

    # Fez protocol rule: Fez requires two-stage (survey / confirm) under addendum
    if target_backend_name in ["ibm_fez", "fake_fez"] and stage == "full":
        raise ValueError(
            "Refusal: ibm_fez requires two-stage characterization (--stage survey followed by --stage confirm) under PREREGISTRATION_v2_4_addendum_fez."
        )

    target_mode = getattr(args, "target", "fake")

    # X3: measure_p1_v2 without --target live must never touch a live backend.
    if getattr(args, "execute", False) and target_mode != "live" and not getattr(args, "local", False):
        raise ValueError("Refusal: --execute requires --target live for QPU execution. To run an offline rehearsal, pass --local explicitly.")

    if getattr(args, "local", False):
        print("  [OFFLINE LOCAL REHEARSAL ONLY - NOT SUBMITTED TO QPU HARDWARE]")

    if args.local or target_mode != "live":
        if target_backend_name in ["ibm_fez", "fake_fez"] or getattr(args, "backend_override", "") == "fake_fez":
            from qiskit_ibm_runtime.fake_provider import FakeFez
            backend = FakeFez()
        elif getattr(args, "backend_override", "") == "fake_vigo":
            from qiskit_ibm_runtime.fake_provider import FakeVigoV2
            backend = FakeVigoV2()
        else:
            from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
            backend = FakeMarrakesh()
        cal_timestamp = get_calibration_timestamp(backend)
    else:
        service = get_ibm_service()
        live_backends = [b.name for b in service.backends()]
        if target_backend_name not in live_backends:
            raise ValueError(f"Refusal: Pinned backend '{target_backend_name}' not in live backends list: {live_backends}")
        backend = service.backend(target_backend_name)
        cal_timestamp = get_calibration_timestamp(backend)

    target = backend.target

    # 2. Stage Handling & Circuit Construction
    qubit_sets = {}
    isolated_batches = []
    survey_data = None
    survey_path = None

    if stage == "full":
        p1_file = Path("results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json")
        if not p1_file.exists():
            raise FileNotFoundError(f"Base p1 calibration file not found: {p1_file}")
        with open(p1_file, "r", encoding="utf-8") as f:
            base_p1_data = json.load(f)
        p1_dict = base_p1_data.get("p1_estimate", {})
        ro_dict = base_p1_data.get("target_readout_error", {})
        qubit_sets = get_p1_qubit_sets(p1_dict, ro_dict, target=target)
        if target.num_qubits >= 156:
            expected_hot = [125, 22, 26, 11, 145, 23, 76, 30]
            for eq in expected_hot:
                if eq not in qubit_sets["HOT"]:
                    raise AssertionError(f"Expected hot qubit {eq} missing from HOT set: {qubit_sets['HOT']}")
        isolated_pool = list(dict.fromkeys(qubit_sets["HOT"] + qubit_sets["FAIL"] + qubit_sets["CONTROL"]))
        isolated_batches = partition_isolated_batches(isolated_pool, target, min_distance=3)

    elif stage == "survey":
        # Survey measures all qubits simultaneously; no batches needed
        pass

    elif stage == "confirm":
        # Checkpoint Z3: Confirm stage requires valid survey file
        from_survey_arg = getattr(args, "from_survey", None)
        if from_survey_arg:
            survey_path = Path(from_survey_arg)
        else:
            cand_files = sorted(glob.glob(f"results/hardware_v2/p1v2_survey_{backend.name}_*.json"))
            if not cand_files:
                cand_files = sorted(glob.glob(f"results/hardware_v2/p1v2_survey_{target_backend_name}_*.json"))
            survey_path = Path(cand_files[-1]) if cand_files else None

        if not survey_path or not survey_path.exists():
            raise ValueError("Refusal: Survey file is missing for --stage confirm. Run --stage survey first.")

        with open(survey_path, "r", encoding="utf-8") as f:
            survey_data = json.load(f)

        # Refusal gate: Backend match
        s_backend = survey_data.get("backend", "")
        if s_backend != backend.name and s_backend != target_backend_name:
            raise ValueError(f"Refusal: Survey backend '{s_backend}' does not match target backend '{target_backend_name}'.")

        # Refusal gate: Calibration match
        s_cal_ts = survey_data.get("calibration_timestamp")
        if s_cal_ts != cal_timestamp:
            raise ValueError(f"Refusal: Survey calibration timestamp '{s_cal_ts}' does not match live target calibration timestamp '{cal_timestamp}'. Re-run --stage survey on fresh calibration.")

        # Refusal gate: Age <= 6 hours
        s_exec_ts_str = survey_data.get("execution_timestamp", "")
        s_exec_ts = parse_iso_ts(s_exec_ts_str).timestamp() if s_exec_ts_str else float(survey_data.get("timestamp", 0.0))
        age_hours = abs(time.time() - s_exec_ts) / 3600.0
        if age_hours > 6.0:
            raise ValueError(f"Refusal: Survey file '{survey_path}' is older than 6 hours ({age_hours:.2f} h > 6.0 h). Re-run --stage survey.")

        qubit_sets = get_survey_qubit_sets(survey_data, target=target)
        isolated_pool = list(dict.fromkeys(qubit_sets["HOT"] + qubit_sets["CONTROL"]))
        isolated_batches = partition_isolated_batches(isolated_pool, target, min_distance=3)

    # Build circuits
    circuit_descs = build_measurement_circuits(target, qubit_sets, isolated_batches, stage=stage)
    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    isa_circuits = []
    for item in circuit_descs:
        isa_qc = pm.run(item["circuit"])
        validate_isa(isa_qc, target)
        item["isa_circuit"] = isa_qc
        isa_circuits.append(isa_qc)

    # 3. Pilot file path and scale resolution per backend (Checkpoint Z2)
    if target_backend_name in ["ibm_fez", "fake_fez"]:
        pilot_scale_path = Path("results/hardware_extra/pilot_scale_ibm_fez.json")
    else:
        pilot_scale_path = Path("results/hardware_extra/pilot_scale.json")

    valid_live_pilot = False
    if pilot_scale_path.exists():
        try:
            with open(pilot_scale_path, "r", encoding="utf-8") as f:
                ps_data = json.load(f)
            j_id = str(ps_data.get("job_id", "")).strip()
            is_loc = ps_data.get("is_local", False) or ps_data.get("target") in ["fake", "local"]
            b_name = ps_data.get("backend", "")
            cal_ts = ps_data.get("calibration_timestamp")
            age_s = abs(time.time() - float(ps_data.get("timestamp", 0.0)))
            if (
                not j_id.startswith("pilot_local")
                and not is_loc
                and b_name == target_backend_name
                and cal_ts
                and age_s <= 86400
            ):
                valid_live_pilot = True
                scale = ps_data.get("qpu_seconds_per_shot", ps_data.get("seconds_per_shot", 3.0 / shots))
        except Exception:
            valid_live_pilot = False

    if not valid_live_pilot:
        scale = 3.0 / shots

    sec_per_pub = scale * shots
    total_est_seconds = len(circuit_descs) * sec_per_pub

    allowance_remaining = float(config.get("allowance_remaining_s", 279))
    reserve_s = float(config.get("reserve_s", 150))
    extra_run_cap_s = float(config.get("extra_run_cap_s", 110))

    # Print Z4 Cumulative Budget Priority Table at the top of dry run (Checkpoint Z4)
    z4_budget = compute_z4_cumulative_budget(config)
    print(format_z4_budget_table(z4_budget))
    print()

    print("=" * 80)
    print(f"ENHANCED P1 CHARACTERIZATION PROTOCOL (measure_p1_v2 --stage {stage})")
    print("=" * 80)
    print(f"Backend:                  {backend.name} ({target.num_qubits} qubits)")
    print(f"Calibration Timestamp:    {cal_timestamp}")
    print(f"Protocol Stage:           {stage}")
    print(f"Shots per PUB:            {shots}")
    print(f"Total Circuits (PUBs):    {len(circuit_descs)}")
    if stage == "full":
        print(f"  - Block A (all qubits): 4 circuits (2 reps x 2 preps)")
        print(f"  - Block B (isolated):   {len(isolated_batches) * 4} circuits ({len(isolated_batches)} batches x 2 reps x 2 preps)")
        print(f"  - Block C (single):     {len([q for q in [125, 22] if q < target.num_qubits]) * 2} circuits")
    elif stage == "survey":
        print(f"  - Block A (all qubits): 4 circuits (2 reps x 2 preps, simultaneous survey)")
    elif stage == "confirm":
        print(f"  - Survey Source:        {survey_path.as_posix() if survey_path else 'None'}")
        print(f"  - Candidate Hot:        {qubit_sets.get('HOT', [])}")
        print(f"  - Control Qubits:       {qubit_sets.get('CONTROL', [])}")
        print(f"  - Isolated Batches:     {len(isolated_batches)} batches (min distance >= 3)")
        print(f"  - Block B (isolated):   {len(circuit_descs)} circuits ({len(isolated_batches)} batches x 2 reps x 2 preps)")

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

    # 4. Pilot Mode Handling
    pilot_est_seconds = sec_per_pub
    if getattr(args, "pilot", False):
        print("\n" + "=" * 80)
        print(f"PILOT CALIBRATION MODE (measure_p1_v2 --backend {backend.name})")
        print("=" * 80)
        print(f"Preparing 1 measurement PUB to calibrate timing scale on {backend.name}...")
        print(f"Estimated pilot duration: {pilot_est_seconds:.2f} s")
        required_pilot_approve = math.ceil(pilot_est_seconds)

        if not args.execute and not args.local:
            print("\n[PILOT DRY RUN COMPLETE] No jobs submitted to hardware.")
            print(f"To execute pilot on QPU, run with: --pilot --execute --approve-seconds {required_pilot_approve}")
            return

        if args.approve_seconds is None or int(args.approve_seconds) != required_pilot_approve:
            raise ValueError(f"Refusal: Must provide --approve-seconds {required_pilot_approve} for pilot execution.")

        pilot_circuit = isa_circuits[0]
        if args.local or target_mode != "live":
            j_id = f"pilot_local_{int(time.time())}"
            usage = pilot_est_seconds
            is_local = True
            origin_target = "fake" if target_mode != "live" else "local"
        else:
            sampler = Sampler(mode=backend)
            sampler.options.dynamical_decoupling.enable = False
            sampler.options.twirling.enable_gates = False
            job = sampler.run([pilot_circuit], shots=shots)
            j_id = job.job_id()
            job.result()
            usage = extract_qpu_seconds(job, pilot_est_seconds)
            is_local = False
            origin_target = "live"

        pilot_scale_path.parent.mkdir(parents=True, exist_ok=True)
        pilot_data = {
            "timestamp": time.time(),
            "backend": backend.name,
            "calibration_timestamp": cal_timestamp,
            "job_id": j_id,
            "seconds_per_shot": usage / shots,
            "qpu_seconds_per_shot": usage / shots,
            "pilot_usage": usage,
            "pilot_shots": shots,
            "source": f"measure_p1_v2 --pilot --backend {backend.name}",
            "is_local": is_local,
            "target": origin_target,
        }
        with open(pilot_scale_path, "w", encoding="utf-8") as f:
            json.dump(pilot_data, f, indent=2)
        print(f"\n[SUCCESS] Pilot scale saved to {pilot_scale_path} (Job ID: {j_id}, Usage: {usage:.2f} s).")
        return

    if not args.execute and not args.local:
        print("\n[DRY RUN ONLY] No jobs submitted to hardware.")
        print(f"To execute on live QPU, run with: --execute --approve-seconds {math.ceil(total_est_seconds)}")
        return

    # 5. Execution Mode (Full measurement run)
    if getattr(args, "execute", False):
        if not pilot_scale_path.exists():
            raise ValueError(f"Refusal: Pilot file '{pilot_scale_path}' missing. Run --pilot first.")
        with open(pilot_scale_path, "r", encoding="utf-8") as f:
            pilot_data = json.load(f)

        # Cross-backend refusal check
        p_backend = pilot_data.get("backend", "")
        is_match = (p_backend == target_backend_name) or (p_backend == backend.name)
        if not is_match:
            raise ValueError(f"Refusal: Pilot file backend '{p_backend}' does not match config extra_backend '{target_backend_name}'.")

        # Live refusal gates
        if target_mode == "live" and not getattr(args, "local", False):
            job_id_str = str(pilot_data.get("job_id", "")).strip()
            if job_id_str.startswith("pilot_local"):
                raise ValueError(f"Refusal: Pilot file contains local job_id '{job_id_str}'. Real QPU pilot run required.")
            if pilot_data.get("is_local", False) or pilot_data.get("target") in ["fake", "local"] or "fake" in str(p_backend).lower():
                raise ValueError("Refusal: Pilot file was generated under --local or --target fake. Real QPU pilot run required.")
            if not pilot_data.get("calibration_timestamp") or not str(pilot_data.get("calibration_timestamp")).strip():
                raise ValueError("Refusal: Pilot file lacks calibration timestamp.")
            pilot_ts = float(pilot_data.get("timestamp", 0.0))
            if abs(time.time() - pilot_ts) > 86400:
                raise ValueError("Refusal: Pilot scale file is older than 24 hours. Re-run --pilot.")
            if pilot_data.get("calibration_timestamp") != cal_timestamp:
                raise ValueError("Refusal: Pilot calibration timestamp does not match target calibration timestamp. Run --pilot on fresh calibration.")
        else:
            pilot_ts = float(pilot_data.get("timestamp", 0.0))
            if abs(time.time() - pilot_ts) > 86400:
                raise ValueError("Refusal: Pilot scale file is older than 24 hours. Run --pilot again.")

        required_approve = math.ceil(total_est_seconds)
        if args.approve_seconds is None or int(args.approve_seconds) != required_approve:
            raise ValueError(f"Refusal: Must provide --approve-seconds {required_approve} matching total estimate.")

    print(f"\nSubmitting {len(isa_circuits)} measurement PUBs to {backend.name}...")
    
    if args.local:
        job_id = f"local_test_{stage}_{int(time.time())}"
        qpu_time = 0.0
        results_data = []
        for desc in circuit_descs:
            prep = desc["prep"]
            qubits = desc["qubits"]
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

    # 6. Extract per-qubit, per-block metrics
    exec_timestamp = datetime.now(timezone.utc).isoformat()
    
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

    output_payload = {
        "job_id": job_id,
        "backend": backend.name,
        "stage": stage,
        "execution_timestamp": exec_timestamp,
        "timestamp": time.time(),
        "calibration_timestamp": cal_timestamp,
        "shots_per_pub": shots,
        "qpu_seconds_estimated": total_est_seconds,
        "qpu_seconds_measured": qpu_time,
        "calibrated_asymmetry_exposed": has_cal_asym,
        "subtraction_method": "excess_over_calibrated_prob_meas1_prep0" if has_cal_asym else "excess_over_calibrated_ro",
        "target_calibrated_ro": cal_ro,
        "target_calibrated_prob_meas1_prep0": cal_p10,
        "target_calibrated_prob_meas0_prep1": cal_p01,
        "qubit_sets": qubit_sets,
        "isolated_batches": isolated_batches,
        "measurements": {}
    }

    if stage == "confirm":
        output_payload["survey_source"] = survey_path.as_posix() if survey_path else ""

    # Aggregate counts by block and prep
    for item_idx, desc in enumerate(circuit_descs):
        counts = results_data[item_idx]
        block = desc["block"]
        prep = desc["prep"]
        rep = desc["rep"]
        qubits = desc["qubits"]

        for bit_pos, q in enumerate(qubits):
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

    # Evaluate persistence if confirm stage
    if stage == "confirm" and survey_data:
        persistent_qubits = []
        cand_hot = qubit_sets.get("HOT", [])
        for q in cand_hot:
            q_str = str(q)
            ro_val = cal_ro.get(q_str, 0.0)
            base_p10 = cal_p10.get(q_str) if cal_p10.get(q_str) is not None else ro_val

            # Isolated confirm check
            b_blocks = output_payload["measurements"].get(q_str, {}).get("B", {})
            confirmed_isolated = False
            for b_sub, b_vals in b_blocks.items():
                p10_conf = b_vals.get("P10", 0.0)
                if (p10_conf - base_p10) > 0.05:
                    confirmed_isolated = True
                    break
            if confirmed_isolated and ro_val <= 0.20:
                persistent_qubits.append(q)

        output_payload["persistent_qubits"] = persistent_qubits
        output_payload["persistent_count"] = len(persistent_qubits)

    # Save artifact
    out_dir = Path(getattr(args, "out_dir", "results/hardware_v2"))
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_ts = exec_timestamp.replace(":", "").replace("-", "").replace("+", "_").replace(".", "_")

    if stage == "survey":
        out_file = out_dir / f"p1v2_survey_{backend.name}_{safe_ts}.json"
    elif stage == "confirm":
        out_file = out_dir / f"p1v2_confirm_{backend.name}_{safe_ts}.json"
    else:
        out_file = out_dir / f"p1v2_{backend.name}_{safe_ts}.json"
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\n[SUCCESS] Stage {stage} data saved to {out_file}")

    return str(out_file)

def main():
    parser = argparse.ArgumentParser(description="Measure QPU P1 (v2) Protocol")
    parser.add_argument("--target", choices=["fake", "live"], default="fake", help="Target mode: fake (offline rehearsal) or live (real QPU target)")
    parser.add_argument("--backend", choices=["ibm_marrakesh", "ibm_fez"], default=None, help="Target backend name (default: config extra_backend)")
    parser.add_argument("--stage", choices=["full", "survey", "confirm"], default="full", help="Protocol stage: full (24 PUBs), survey (4 PUBs), confirm (16 PUBs)")
    parser.add_argument("--from-survey", type=str, default=None, help="Survey JSON file for --stage confirm")
    parser.add_argument("--dry-run", action="store_true", help="Dry run estimation without hardware execution")
    parser.add_argument("--pilot", action="store_true", help="Run 1 PUB to calibrate timing scale")
    parser.add_argument("--execute", action="store_true", help="Execute on hardware")
    parser.add_argument("--approve-seconds", type=int, default=None, help="Approve QPU execution budget")
    parser.add_argument("--local", action="store_true", help="Run with local fake backend")
    parser.add_argument("--out-dir", type=str, default="results/hardware_v2", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--backend-override", type=str, default="fake_marrakesh", help="fake_vigo, fake_marrakesh, or fake_fez for local runs")
    args = parser.parse_args()

    run_measure_p1_v2(args)

if __name__ == "__main__":
    main()
