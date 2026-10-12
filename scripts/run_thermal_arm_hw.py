#!/usr/bin/env python3
"""
scripts/run_thermal_arm_hw.py

Pre-registered / gated thermal-arm hardware rerun on IBM Quantum (ibm_marrakesh).
Compares plain ESP vs thermal ESP on the live target.

Gates:
- Gate 0: Budget cap verification against config.yaml.
- Gate 1: Offline layout diff on live ibm_marrakesh target.
- Gate 2: Decision rule (skip if all layouts identical; submit only differing benchmarks).
- Gate 3: Approval flag verification (--approve-seconds).
- Execution: Single job on QPU (8192 shots, 2 repetitions).
"""

import os
import sys
import json
import time
import math
import hashlib
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

from src.noise_model.profile import load_config
from src.benchmarks.circuits import get_all_benchmarks
from src.mappers.mappers import mapper_esp
from src.metrics.outcome import compute_outcome_metric, compute_ideal_distribution
from src.runner_ibm import (
    get_ibm_service, get_calibration_timestamp, extract_qpu_seconds
)

def validate_isa(circuit, target):
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if op.name not in ['barrier', 'delay', 'measure']:
            if not target.instruction_supported(op.name, qargs):
                raise ValueError(f"Instruction {op.name} on {qargs} not supported by target.")

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def run_thermal_arm(approve_seconds=None):
    out_dir = Path("results/hardware_thermal_arm")
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path("results/analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # GATE 0: BUDGET VERIFICATION
    # ---------------------------------------------------------
    config = load_config()
    # Read config values and report file & lines:
    # config.yaml line 31: allowance_remaining_s: 279
    # config.yaml line 32: reserve_s: 150
    # config.yaml line 33: extra_run_cap_s: 110
    allowance_remaining_s = float(config.get("allowance_remaining_s", 279))
    reserve_s = float(config.get("reserve_s", 150))
    extra_run_cap_s = float(config.get("extra_run_cap_s", 110))

    spendable = allowance_remaining_s - reserve_s
    hard_cap = min(extra_run_cap_s, spendable)
    planning_sec_per_circuit = 3.0

    print("=" * 80)
    print("GATE 0: BUDGET VERIFICATION")
    print("=" * 80)
    print(f"Config path: config.yaml")
    print(f"  Line 31: allowance_remaining_s = {allowance_remaining_s:.1f} s")
    print(f"  Line 32: reserve_s             = {reserve_s:.1f} s")
    print(f"  Line 33: extra_run_cap_s       = {extra_run_cap_s:.1f} s")
    print(f"Spendable Budget (allowance - reserve): {spendable:.1f} s")
    print(f"Hard Cap for Experiment:               {hard_cap:.1f} s")
    print(f"Planning Rate:                          {planning_sec_per_circuit:.1f} s / circuit")
    print("Gate 0 Budget check: PASS")
    print("")

    # ---------------------------------------------------------
    # GATE 1: OFFLINE LAYOUT DIFF ON LIVE TARGET
    # ---------------------------------------------------------
    print("=" * 80)
    print("GATE 1: OFFLINE LAYOUT DIFF ON LIVE IBM_MARRAKESH TARGET")
    print("=" * 80)
    backend_name = config.get("extra_backend", "ibm_marrakesh")
    print(f"Connecting read-only to live backend: {backend_name}...")
    service = get_ibm_service()
    backend = service.backend(backend_name)
    target = backend.target
    cal_timestamp = get_calibration_timestamp(backend)
    print(f"Live Backend:               {backend_name}")
    print(f"Live Calibration Timestamp: {cal_timestamp}")

    # Load measured p1 data
    p1_path = Path("results/hardware/p1_ibm_marrakesh_20261009T001839_0530.json")
    if not p1_path.exists():
        raise FileNotFoundError(f"Measured p1 file {p1_path} not found.")
    with open(p1_path, "r", encoding="utf-8") as f:
        p1_data = json.load(f)
    p1_dict = {int(k): float(v) for k, v in p1_data["p1_estimate"].items()}
    p1_cal_time = p1_data.get("timestamp", "unknown")
    print(f"Measured p1 File:           {p1_path.name} (calibrated {p1_cal_time})")

    # Readout errors per qubit on target
    ro_errors = {}
    for q in range(target.num_qubits):
        try:
            m_props = target["measure"].get((q,), None)
            ro_errors[q] = getattr(m_props, "error", 0.0) if m_props else 0.0
        except KeyError:
            ro_errors[q] = 0.0

    def get_layout_excess(layout):
        excess_list = [max(0.0, p1_dict.get(q, 0.0) - ro_errors.get(q, 0.0)) for q in layout]
        return max(excess_list), sum(excess_list) / len(excess_list)

    benchmarks = get_all_benchmarks(5)
    bench_names = ['ghz', 'bv', 'qft', 'mirror']

    diff_results = {}
    print("\nRunning paired mapper_esp (plain vs thermal) on live target (seed=42)...")
    for b_name in bench_names:
        circ = benchmarks[b_name]
        tc_plain, esp_plain, _, lay_plain, _ = mapper_esp(
            circ, target, temps_mk=None, p1=None, use_thermal=False, seed=42, exhaustive=False
        )
        tc_th, esp_th, _, lay_th, _ = mapper_esp(
            circ, target, temps_mk=None, p1=p1_dict, use_thermal=True, seed=42, exhaustive=False
        )

        p_max_exc, p_mean_exc = get_layout_excess(lay_plain)
        t_max_exc, t_mean_exc = get_layout_excess(lay_th)
        is_diff = (lay_plain != lay_th)

        diff_results[b_name] = {
            "plain_circuit": tc_plain,
            "thermal_circuit": tc_th,
            "plain_layout": lay_plain,
            "thermal_layout": lay_th,
            "plain_esp": esp_plain,
            "thermal_esp": esp_th,
            "plain_max_excess": p_max_exc,
            "plain_mean_excess": p_mean_exc,
            "thermal_max_excess": t_max_exc,
            "thermal_mean_excess": t_mean_exc,
            "differs": is_diff,
        }
        print(f"  {b_name.upper():<6}: plain={lay_plain} (max exc={p_max_exc:.4f}, mean exc={p_mean_exc:.4f})")
        print(f"          thermal={lay_th} (max exc={t_max_exc:.4f}, mean exc={t_mean_exc:.4f}) -> Differs: {is_diff}")

    # Write results/hardware_thermal_arm/layout_diff.txt
    layout_diff_path = out_dir / "layout_diff.txt"
    diff_lines = []
    diff_lines.append("=" * 80)
    diff_lines.append("HARDWARE THERMAL-ARM OFFLINE LAYOUT DIFF REPORT")
    diff_lines.append("=" * 80)
    diff_lines.append(f"Backend:                      {backend_name}")
    diff_lines.append(f"Live Target Calibration Time: {cal_timestamp}")
    diff_lines.append(f"Source p1 Calibration File:   {p1_path.name} ({p1_cal_time})")
    diff_lines.append(f"Paired Evaluation:            Both arms compiled on SAME live target in same run")
    diff_lines.append(f"Transpiler Settings:          seed=42, exhaustive=False, opt_level=3")
    diff_lines.append("")
    diff_lines.append("GATE 0 BUDGET AUDIT:")
    diff_lines.append(f"  allowance_remaining_s:      {allowance_remaining_s:.1f} s (config.yaml line 31)")
    diff_lines.append(f"  reserve_s:                  {reserve_s:.1f} s (config.yaml line 32)")
    diff_lines.append(f"  extra_run_cap_s:            {extra_run_cap_s:.1f} s (config.yaml line 33)")
    diff_lines.append(f"  spendable:                  {spendable:.1f} s")
    diff_lines.append(f"  hard_cap:                   {hard_cap:.1f} s")
    diff_lines.append("")
    diff_lines.append("PER-BENCHMARK LAYOUT COMPARISON:")
    diff_lines.append("-" * 80)
    diff_lines.append(f"{'Benchmark':<10} {'Diff?':<8} {'Plain Layout':<25} {'Plain Exc (mean/max)':<22} {'Thermal Layout':<25} {'Thermal Exc (mean/max)'}")
    diff_lines.append("-" * 115)
    for b_name in bench_names:
        r = diff_results[b_name]
        diff_lines.append(
            f"{b_name.upper():<10} {str(r['differs']):<8} {str(r['plain_layout']):<25} "
            f"{r['plain_mean_excess']:.4f} / {r['plain_max_excess']:.4f}       "
            f"{str(r['thermal_layout']):<25} {r['thermal_mean_excess']:.4f} / {r['thermal_max_excess']:.4f}"
        )
    diff_lines.append("")

    # ---------------------------------------------------------
    # GATE 2: DECISION RULE
    # ---------------------------------------------------------
    differing_benchmarks = [b for b in bench_names if diff_results[b]["differs"]]
    num_diff = len(differing_benchmarks)

    diff_lines.append("GATE 2 DECISION RULE:")
    diff_lines.append("-" * 80)
    if num_diff == 0:
        diff_msg = f"thermal term did not change any layout on the live target at {cal_timestamp}; no rerun needed"
        diff_lines.append(diff_msg)
        with open(layout_diff_path, "w", encoding="utf-8") as f:
            f.write("\n".join(diff_lines) + "\n")
        print(f"\nGATE 2 DECISION: {diff_msg}")
        print(f"Report written to {layout_diff_path}")
        return

    # Differing benchmarks to submit
    arms_count = 2  # plain and thermal
    reps_count = 2  # 2 repetitions
    circuits_count = num_diff * arms_count * reps_count
    planned_seconds = circuits_count * planning_sec_per_circuit

    diff_lines.append(f"Differing Benchmarks ({num_diff}): {', '.join(b.upper() for b in differing_benchmarks)}")
    diff_lines.append(f"Circuits to execute:    {num_diff} benchmarks x {arms_count} arms x {reps_count} reps = {circuits_count} circuits")
    diff_lines.append(f"Planned QPU Seconds:    {circuits_count} x {planning_sec_per_circuit:.1f} s = {planned_seconds:.1f} s")
    diff_lines.append(f"Hard Cap:               {hard_cap:.1f} s")
    if planned_seconds > hard_cap:
        diff_lines.append(f"STATUS: ABORT (Planned seconds {planned_seconds:.1f} s exceeds hard cap {hard_cap:.1f} s)")
        with open(layout_diff_path, "w", encoding="utf-8") as f:
            f.write("\n".join(diff_lines) + "\n")
        raise RuntimeError(f"Planned seconds {planned_seconds:.1f} s exceeds hard cap {hard_cap:.1f} s!")
    diff_lines.append("STATUS: APPROVED FOR HARDWARE RUN (Planned seconds <= Hard Cap)")
    with open(layout_diff_path, "w", encoding="utf-8") as f:
        f.write("\n".join(diff_lines) + "\n")

    print("\n" + "=" * 80)
    print("GATE 2: DECISION RULE")
    print("=" * 80)
    print(f"Differing benchmarks: {', '.join(b.upper() for b in differing_benchmarks)}")
    print(f"Planned QPU Seconds:  {planned_seconds:.1f} s ({circuits_count} circuits @ {planning_sec_per_circuit:.1f} s/circuit)")
    print(f"Hard Cap:             {hard_cap:.1f} s")
    print("Gate 2 check: PASS (Proceeding to Gate 3)")
    print("")

    # ---------------------------------------------------------
    # GATE 3: APPROVAL FLAG
    # ---------------------------------------------------------
    print("=" * 80)
    print("GATE 3: APPROVAL FLAG CHECK")
    print("=" * 80)
    req_approve = math.ceil(planned_seconds)
    print(f"Required approval seconds: {req_approve} s")
    print(f"Provided --approve-seconds: {approve_seconds}")

    if approve_seconds is None:
        print("\n[DRY RUN COMPLETE] No jobs submitted to QPU.")
        print("To execute this gated experiment on ibm_marrakesh, rerun with:")
        print(f"  python scripts/run_thermal_arm_hw.py --approve-seconds {req_approve}")
        return

    if approve_seconds < req_approve:
        raise ValueError(
            f"--approve-seconds {approve_seconds} is less than required planned seconds {req_approve} s! Aborting."
        )

    print(f"Gate 3 Approval check: PASS (--approve-seconds {approve_seconds} >= {req_approve} s)")
    print("Proceeding to single QPU job submission on live ibm_marrakesh...")
    print("")

    # ---------------------------------------------------------
    # HARDWARE EXECUTION (SINGLE JOB, ONE SESSION)
    # ---------------------------------------------------------
    print("=" * 80)
    print("HARDWARE EXECUTION ON IBM_MARRAKESH")
    print("=" * 80)

    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build unique transpiled ISA circuits
    compiled_items = []
    for b_name in differing_benchmarks:
        b_circ = benchmarks[b_name]
        res = diff_results[b_name]

        isa_plain = pm.run(res["plain_circuit"])
        validate_isa(isa_plain, target)
        ops_p = isa_plain.count_ops()
        cx_p = ops_p.get('cz', ops_p.get('cx', ops_p.get('ecr', 0)))

        isa_th = pm.run(res["thermal_circuit"])
        validate_isa(isa_th, target)
        ops_t = isa_th.count_ops()
        cx_t = ops_t.get('cz', ops_t.get('cx', ops_t.get('ecr', 0)))

        compiled_items.append({
            "benchmark": b_name,
            "method": "plain",
            "logical_circuit": b_circ,
            "isa_circuit": isa_plain,
            "layout": res["plain_layout"],
            "cx_count": cx_p,
            "depth": isa_plain.depth(),
        })
        compiled_items.append({
            "benchmark": b_name,
            "method": "thermal",
            "logical_circuit": b_circ,
            "isa_circuit": isa_th,
            "layout": res["thermal_layout"],
            "cx_count": cx_t,
            "depth": isa_th.depth(),
        })

    # Prepare single-job circuit batch (2 repetitions)
    # circuits_batch will contain 8 circuits for rep 0 followed by 8 circuits for rep 1
    execution_items = []
    circuits_batch = []
    for rep in [0, 1]:
        for item in compiled_items:
            circuits_batch.append(item["isa_circuit"])
            execution_items.append({
                "repetition": rep,
                "benchmark": item["benchmark"],
                "method": item["method"],
                "logical_circuit": item["logical_circuit"],
                "layout": item["layout"],
                "cx_count": item["cx_count"],
                "depth": item["depth"],
            })

    print(f"Total PUBs in single job: {len(circuits_batch)} (8 circuits x 2 reps)")
    shots = int(config.get("shots", 8192))
    print(f"Shots per PUB:            {shots}")
    print(f"Submitting single job to {backend_name}...")

    submission_timestamp = cal_timestamp
    sampler = Sampler(mode=backend)
    sampler.options.dynamical_decoupling.enable = False
    sampler.options.twirling.enable_gates = False

    t0_sub = time.time()
    job = sampler.run(circuits_batch, shots=shots)
    job_id = job.job_id()
    print(f"Submitted Job ID: {job_id}")
    print("Waiting for job completion (no retries on failure)...")

    result = job.result()
    t_elapsed = time.time() - t0_sub
    print(f"Job completed in {t_elapsed:.1f} s wall time.")

    # Extract metrics
    run_sec = extract_qpu_seconds(job, planned_seconds)
    print(f"Job QPU run_seconds: {run_sec:.2f} s")
    print(f"Note: run_seconds is from job metrics and must be cross-checked against the IBM dashboard manually.")

    # Process and save raw counts
    ideal_cache = {}
    rows = []
    raw_files = []

    for idx, exec_item in enumerate(execution_items):
        b_name = exec_item["benchmark"]
        m_name = exec_item["method"]
        rep = exec_item["repetition"]
        b_circ = exec_item["logical_circuit"]

        pub_res = result[idx].data
        counts = pub_res.meas.get_counts() if hasattr(pub_res, 'meas') else pub_res.c.get_counts()

        raw_filename = f"{b_name}_{m_name}_r{rep}.json"
        raw_filepath = raw_dir / raw_filename
        with open(raw_filepath, "w", encoding="utf-8") as rf:
            json.dump(counts, rf)
        raw_files.append(raw_filepath)

        if b_name not in ideal_cache:
            ideal_cache[b_name] = compute_ideal_distribution(b_circ)

        m_type, m_val = compute_outcome_metric(b_name, counts, b_circ, ideal_cache[b_name], shots=shots)

        rows.append({
            "job_id": job_id,
            "backend": backend_name,
            "calibration_timestamp": submission_timestamp,
            "queue_seconds": 0.0,
            "run_seconds": run_sec,
            "shots": shots,
            "repetition": rep,
            "benchmark": b_name,
            "method": m_name,
            "metric_type": m_type,
            "metric_val": m_val,
            "cx_count": exec_item["cx_count"],
            "depth": exec_item["depth"],
            "layout": f'"{exec_item["layout"]}"',
        })

    # Write raw.sha256
    sha256_path = out_dir / "raw.sha256"
    with open(sha256_path, "w", encoding="utf-8") as sf:
        for rf in sorted(raw_files):
            digest = compute_sha256(rf)
            rel_name = f"raw/{rf.name}"
            sf.write(f"{digest}  {rel_name}\n")
    print(f"Raw counts SHA256 written to {sha256_path}")

    # Write thermal_arm_table.csv
    table_path = out_dir / "thermal_arm_table.csv"
    with open(table_path, "w", encoding="utf-8") as tf:
        tf.write("job_id,backend,calibration_timestamp,queue_seconds,run_seconds,shots,repetition,benchmark,method,metric_type,metric_val,cx_count,depth,layout\n")
        for r in rows:
            line = f"{r['job_id']},{r['backend']},{r['calibration_timestamp']},{r['queue_seconds']},{r['run_seconds']},{r['shots']},{r['repetition']},{r['benchmark']},{r['method']},{r['metric_type']},{r['metric_val']},{r['cx_count']},{r['depth']},{r['layout']}\n"
            tf.write(line)
    print(f"Results table written to {table_path} ({len(rows)} rows)")

    # ---------------------------------------------------------
    # COMPUTE SUMMARY AND FACTS
    # ---------------------------------------------------------
    # Compute paired summary (thermal minus plain per benchmark)
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("THERMAL-ARM HARDWARE RERUN RESULTS (EXPLORATORY)")
    summary_lines.append("=" * 80)
    summary_lines.append(f"Backend:                      {backend_name}")
    summary_lines.append(f"Live Calibration Timestamp:   {submission_timestamp}")
    summary_lines.append(f"Job ID:                       {job_id}")
    summary_lines.append(f"Shots:                        {shots} per circuit")
    summary_lines.append(f"Repetitions:                  {reps_count}")
    summary_lines.append(f"Circuits Executed:            {len(circuits_batch)} (single job)")
    summary_lines.append(f"Planned QPU Seconds:          {planned_seconds:.1f} s")
    summary_lines.append(f"Measured Job Run Seconds:     {run_sec:.2f} s")
    remaining_allowance = allowance_remaining_s - run_sec
    summary_lines.append(f"Allowance Remaining:          {remaining_allowance:.2f} s (allowance_remaining_s {allowance_remaining_s:.1f} s - measured {run_sec:.2f} s; dashboard must be checked manually)")
    summary_lines.append("")
    summary_lines.append("HONESTY CONSTRAINT:")
    summary_lines.append("  exploratory, one calibration day, 2 repetitions, 1 job. No p-values or CIs computed.")
    summary_lines.append("  Do not offer a causal story. If thermal does not beat plain, state plainly.")
    summary_lines.append("")
    summary_lines.append("PER-BENCHMARK PAIRED RESULTS:")
    summary_lines.append("-" * 80)
    summary_lines.append(f"{'Benchmark':<10} {'Metric':<18} {'Plain r0 / r1 (Mean)':<24} {'Thermal r0 / r1 (Mean)':<24} {'Delta (Th - Pl)'}")
    summary_lines.append("-" * 85)

    facts_lines = []
    facts_lines.append(f"backend: {backend_name}")
    facts_lines.append(f"calibration_timestamp: {submission_timestamp}")
    facts_lines.append(f"job_id: {job_id}")
    facts_lines.append(f"planned_seconds: {planned_seconds:.1f}")
    facts_lines.append(f"measured_run_seconds: {run_sec:.2f}")
    facts_lines.append(f"allowance_remaining_s: {remaining_allowance:.2f}")

    for b_name in differing_benchmarks:
        m_type = [r["metric_type"] for r in rows if r["benchmark"] == b_name][0]
        p_r0 = [r["metric_val"] for r in rows if r["benchmark"] == b_name and r["method"] == "plain" and r["repetition"] == 0][0]
        p_r1 = [r["metric_val"] for r in rows if r["benchmark"] == b_name and r["method"] == "plain" and r["repetition"] == 1][0]
        p_mean = (p_r0 + p_r1) / 2.0

        t_r0 = [r["metric_val"] for r in rows if r["benchmark"] == b_name and r["method"] == "thermal" and r["repetition"] == 0][0]
        t_r1 = [r["metric_val"] for r in rows if r["benchmark"] == b_name and r["method"] == "thermal" and r["repetition"] == 1][0]
        t_mean = (t_r0 + t_r1) / 2.0

        delta_r0 = (t_r0 - p_r0) * 100.0
        delta_r1 = (t_r1 - p_r1) * 100.0
        delta_mean = (t_mean - p_mean) * 100.0

        summary_lines.append(
            f"{b_name.upper():<10} {m_type:<18} "
            f"{p_r0*100:.2f}% / {p_r1*100:.2f}% ({p_mean*100:.2f}%)  "
            f"{t_r0*100:.2f}% / {t_r1*100:.2f}% ({t_mean*100:.2f}%)  "
            f"{delta_mean:+.2f} pts"
        )
        facts_lines.append(f"{b_name}_plain_mean: {p_mean*100:.2f}%")
        facts_lines.append(f"{b_name}_plain_r0: {p_r0*100:.2f}%")
        facts_lines.append(f"{b_name}_plain_r1: {p_r1*100:.2f}%")
        facts_lines.append(f"{b_name}_thermal_mean: {t_mean*100:.2f}%")
        facts_lines.append(f"{b_name}_thermal_r0: {t_r0*100:.2f}%")
        facts_lines.append(f"{b_name}_thermal_r1: {t_r1*100:.2f}%")
        facts_lines.append(f"{b_name}_delta_mean_pts: {delta_mean:+.2f}")
        facts_lines.append(f"{b_name}_delta_r0_pts: {delta_r0:+.2f}")
        facts_lines.append(f"{b_name}_delta_r1_pts: {delta_r1:+.2f}")

    summary_file = out_dir / "summary.txt"
    with open(summary_file, "w", encoding="utf-8") as sf:
        sf.write("\n".join(summary_lines) + "\n")

    facts_file = analysis_dir / "thermal_arm_rerun_facts.txt"
    with open(facts_file, "w", encoding="utf-8") as ff:
        ff.write("\n".join(facts_lines) + "\n")

    print("\n" + "\n".join(summary_lines))
    print(f"\nSummary written to {summary_file}")
    print(f"Facts written to {facts_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gated thermal-arm rerun on IBM Quantum.")
    parser.add_argument("--approve-seconds", type=int, default=None, help="Approved budget seconds")
    args = parser.parse_args()
    run_thermal_arm(approve_seconds=args.approve_seconds)

