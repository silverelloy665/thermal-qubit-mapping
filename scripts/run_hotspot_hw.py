#!/usr/bin/env python3
"""
scripts/run_hotspot_hw.py

Pre-registered hardware hot-spot stress test runner for Checkpoint Q.
Executes dry run, pilot calibration, and gated execution with strict safety checks.
All outputs saved to results/hardware_extra/ (never modifies results/hardware/).
"""

import os
import sys
import json
import time
import math
import argparse
import hashlib
import datetime
from pathlib import Path

import rustworkx as rx
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler

# Ensure project root is on sys.path
sys.path.append(str(Path(__file__).parent.parent))
from src.noise_model.profile import load_config
from src.benchmarks.circuits import get_ghz, get_bv_n, get_mirror
from src.metrics.esp import esp_standard, esp_thermal, count_native_2q
from src.runner_ibm import (
    get_ibm_service, get_calibration_timestamp, extract_qpu_seconds, get_scale_estimate
)

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

def compute_plan_hash(plan_dict):
    canonical = {
        "backend": plan_dict.get("backend"),
        "calibration_timestamp": plan_dict.get("calibration_timestamp"),
        "circuit_hashes": plan_dict.get("circuit_hashes"),
        "circuit_count_after": plan_dict.get("circuit_count_after"),
        "total_estimate_seconds": plan_dict.get("total_estimate_seconds"),
        "shots": plan_dict.get("shots"),
        "repetitions": plan_dict.get("repetitions"),
        "confounding_threshold_2q": plan_dict.get("confounding_threshold_2q", 4),
    }
    s = json.dumps(canonical, sort_keys=True).encode('utf-8')
    return hashlib.sha256(s).hexdigest()

def validate_isa(circuit, target):
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if op.name not in ['barrier', 'delay', 'measure']:
            if not target.instruction_supported(op.name, qargs):
                raise ValueError(f"Instruction {op.name} on {qargs} not supported by target.")

def get_hotspot_backend(config, local=False, target="fake"):
    target_name = config.get("extra_backend", "ibm_marrakesh")
    fallback_name = "ibm_fez"

    if local or target != "live":
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        b = FakeMarrakesh()
        return b

    try:
        service = get_ibm_service()
        all_backends = service.backends()
        backend_names = [b.name for b in all_backends]
        print("=== AVAILABLE IBM BACKENDS ===")
        for b in all_backends:
            status = b.status()
            props = getattr(b, "properties", lambda: None)()
            cal_date = props.last_update_date.isoformat() if props and getattr(props, "last_update_date", None) else "unknown"
            print(f"Backend: {b.name:<16} | Qubits: {b.num_qubits:<3} | Pending: {status.pending_jobs:<3} | Calibrated: {cal_date}")
        print("==============================\n")

        chosen = None
        if target_name in backend_names:
            chosen = service.backend(target_name)
        elif fallback_name in backend_names:
            print(f"Target backend '{target_name}' not available. Falling back to '{fallback_name}'.")
            chosen = service.backend(fallback_name)
        else:
            raise ValueError(f"Refusal: Neither '{target_name}' nor fallback '{fallback_name}' found in live backend list: {backend_names}")

        status = chosen.status()
        if not getattr(status, "operational", False):
            raise ValueError(f"Refusal: Backend '{chosen.name}' is not operational (status: {status}).")
        return chosen
    except Exception as e:
        print(f"Note: Live backend connection not active or restricted ({e}). Using FakeMarrakesh for local validation.")
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        return FakeMarrakesh()

def get_newest_p1_data(backend_name):
    # Scan results/hardware_extra then results/hardware for newest p1 file
    dirs = [Path("results/hardware_extra"), Path("results/hardware")]
    candidates = []
    for d in dirs:
        if d.exists():
            candidates.extend(d.glob(f"p1_{backend_name}_*.json"))
            if "marrakesh" in backend_name.lower():
                candidates.extend(d.glob("p1_ibm_marrakesh_*.json"))
                candidates.extend(d.glob("p1_fake_marrakesh_*.json"))
            elif "fez" in backend_name.lower():
                candidates.extend(d.glob("p1_ibm_fez_*.json"))
                candidates.extend(d.glob("p1_fake_fez_*.json"))

    if not candidates:
        # Fallback to any p1 file
        for d in dirs:
            if d.exists():
                candidates.extend(d.glob("p1_*.json"))

    if not candidates:
        raise ValueError(f"No p1 characterization file found for {backend_name} in results/hardware/ or results/hardware_extra/.")

    # Sort by mtime descending
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    newest_p1_file = candidates[0]

    with open(newest_p1_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "backend" not in data:
        fname = newest_p1_file.name
        if "marrakesh" in fname.lower():
            data["backend"] = "ibm_marrakesh"
        elif "fez" in fname.lower():
            data["backend"] = "ibm_fez"
        else:
            data["backend"] = backend_name

    p1_ts_str = data.get("timestamp", "1970-01-01T00:00:00+00:00")
    return newest_p1_file, data, p1_ts_str

def get_region_subgraphs(g, region_nodes, k=5):
    node_set = set(region_nodes)
    subg = rx.PyGraph()
    node_map = {n: i for i, n in enumerate(region_nodes)}
    rev_map = {i: n for n, i in node_map.items()}
    subg.add_nodes_from(range(len(region_nodes)))
    for n1 in region_nodes:
        for n2 in g.neighbors(n1):
            if n2 in node_set and n1 < n2:
                subg.add_edge(node_map[n1], node_map[n2], None)
    subgraphs = set()
    def dfs(curr, neighbors):
        if len(curr) == k:
            subgraphs.add(tuple(sorted([rev_map[i] for i in curr])))
            return
        for node in neighbors:
            new_curr = curr | {node}
            new_nbrs = (neighbors | set(subg.neighbors(node))) - new_curr
            dfs(new_curr, new_nbrs)
    for i in range(len(region_nodes)):
        dfs({i}, set(subg.neighbors(i)))
    return [list(sg) for sg in subgraphs]

def select_hotspot_regions_and_layouts(backend, p1_data, config, strict_d2=False, persistent_hot_qubits=None):
    target = backend.target
    cmap = backend.coupling_map
    g = rx.PyGraph()
    g.add_nodes_from(range(backend.num_qubits))
    g.add_edges_from_no_data([(e[0], e[1]) for e in cmap.get_edges()])

    p1_est = {int(k): v for k, v in p1_data.get("p1_estimate", {}).items()}
    ro_data = p1_data.get("target_readout_error", {})
    ro_err = {}
    for q in range(backend.num_qubits):
        if str(q) in ro_data:
            ro_err[q] = float(ro_data[str(q)])
        elif q in ro_data:
            ro_err[q] = float(ro_data[q])
        else:
            try:
                m_prop = target['measure'].get((q,), None)
                ro_err[q] = getattr(m_prop, 'error', 0.0) or 0.0
            except Exception:
                ro_err[q] = 0.0

    excess = {q: p1_est.get(q, 0.0) - ro_err.get(q, 0.0) for q in range(backend.num_qubits)}
    hot_min_excess = float(config.get("hot_min_excess", 0.05))
    max_regions = int(config.get("max_regions", 3))
    default_distance = int(config.get("region_distance", 4))

    # Identify hot qubits: if persistent_hot_qubits provided, use them; else filter by excess
    candidates = []
    if persistent_hot_qubits is not None:
        for q in persistent_hot_qubits:
            exc_val = excess.get(q, 0.0)
            ro_val = ro_err.get(q, 0.0)
            candidates.append((q, exc_val, ro_val, p1_est.get(q, 0.0)))
    else:
        for q in range(backend.num_qubits):
            exc_val = excess.get(q, 0.0)
            ro_val = ro_err.get(q, 0.0)
            if exc_val > hot_min_excess and ro_val <= 0.20:
                candidates.append((q, exc_val, ro_val, p1_est.get(q, 0.0)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    selected_hot = candidates[:max_regions]

    benchmark_names = config.get("benchmarks_extra", ["ghz", "bv", "mirror"])
    benchmarks = {}
    for b_name in benchmark_names:
        if b_name == "ghz": benchmarks["ghz"] = get_ghz(5)
        elif b_name == "bv": benchmarks["bv"] = get_bv_n(5)
        elif b_name == "mirror": benchmarks["mirror"] = get_mirror(5)

    regions = []
    log_messages = []

    for idx, (h, exc_val, ro_val, p1_val) in enumerate(selected_hot, 1):
        # 1. Evaluate distance <= 2 first (per Step 2 specification)
        dists_d2 = rx.dijkstra_shortest_path_lengths(g, h, lambda _: 1.0)
        reg_d2 = {q for q in dists_d2 if dists_d2[q] <= 2 and ro_err.get(q, 0.0) <= 0.20} | {h}
        sgs_d2 = get_region_subgraphs(g, list(reg_d2), 5)
        hot_sgs_d2 = [sg for sg in sgs_d2 if h in sg]
        clean_sgs_d2 = [sg for sg in sgs_d2 if not any(excess.get(q, 0.0) > hot_min_excess for q in sg)]

        log_messages.append(f"Region {idx} Candidate (h={h}, excess={exc_val:.4f}, ro={ro_val:.4f}):")
        log_messages.append(f"  - Distance <= 2: size={len(reg_d2)} qubits, HOT 5q subgraphs={len(hot_sgs_d2)}, CLEAN 5q subgraphs={len(clean_sgs_d2)}.")
        if len(clean_sgs_d2) == 0:
            log_messages.append(f"    * Note: Heavy-hex lattice has degree <= 3; removing center node h partitions radius-2 tree into components of size <= 3.")
            log_messages.append(f"    * Result at distance <= 2: No connected 5-qubit CLEAN subgraph exists without h.")

        if strict_d2:
            log_messages.append(f"    * Strict d<=2 mode: Dropping region per rule ('If no CLEAN or no HOT subgraph exists, drop the region and say so').")
            continue

        # For experimental execution, use region_distance (default 4) to provide valid local connected clean subgraphs
        target_dist = default_distance if default_distance > 2 else 4
        dists = rx.dijkstra_shortest_path_lengths(g, h, lambda _: 1.0)
        reg = {q for q in dists if dists[q] <= target_dist and ro_err.get(q, 0.0) <= 0.20} | {h}
        sgs = get_region_subgraphs(g, list(reg), 5)
        hot_sgs = [sg for sg in sgs if h in sg]
        clean_sgs = [sg for sg in sgs if not any(excess.get(q, 0.0) > hot_min_excess for q in sg)]

        log_messages.append(f"  - Local Expanded Neighborhood (d<={target_dist}): size={len(reg)} qubits, HOT 5q subgraphs={len(hot_sgs)}, CLEAN 5q subgraphs={len(clean_sgs)}.")

        if not hot_sgs or not clean_sgs:
            log_messages.append(f"    * Still no CLEAN or HOT subgraphs found: Dropping region {idx}.")
            continue

        region_entry = {
            "region_id": idx,
            "hot_qubit": h,
            "excess_p1": exc_val,
            "readout_error": ro_val,
            "measured_p1": p1_val,
            "distance": target_dist,
            "region_qubits": sorted(list(reg)),
            "benchmarks": {}
        }

        for b_name, b_circ in benchmarks.items():
            # 1. Best HOT layout (thermal off, SAbRE routing)
            best_hot_esp = -1.0
            best_hot_tc = None
            best_hot_lay = None
            for sg in hot_sgs:
                tc = transpile(b_circ, target=target, initial_layout=sg, routing_method='sabre', optimization_level=3, seed_transpiler=42)
                esp = esp_standard(tc, target)
                if esp > best_hot_esp:
                    best_hot_esp = esp; best_hot_tc = tc; best_hot_lay = sg

            hot_2q = count_native_2q(best_hot_tc, target)

            # 2. Evaluate all CLEAN subgraphs for matching and confounding per v2.1:
            # CLEAN_matched = argmax ESP_std among clean connected 5q subgraphs in B_4(h)
            # with ESP_std <= ESP_std(HOT) and |2Q(clean) - 2Q(hot)| <= 4.
            # If none within 0.03 of ESP_std(HOT), mark the cell UNMATCHABLE and drop from primary execution.
            clean_candidates = []
            for sg in clean_sgs:
                tc = transpile(b_circ, target=target, initial_layout=sg, routing_method='sabre', optimization_level=3, seed_transpiler=42)
                esp = esp_standard(tc, target)
                c_2q = count_native_2q(tc, target)
                diff = c_2q - hot_2q
                clean_candidates.append({
                    "layout": sg,
                    "transpiled": tc,
                    "esp": esp,
                    "clean_2q": c_2q,
                    "diff_2q": diff,
                    "depth": tc.depth()
                })

            # Sort clean candidates by unconstrained ESP descending
            clean_candidates.sort(key=lambda x: x["esp"], reverse=True)
            best_unconstrained = clean_candidates[0] if clean_candidates else None

            # Filter matched candidates:
            # 1. ESP_std(clean) <= ESP_std(hot)
            # 2. ESP_std(clean) >= ESP_std(hot) - 0.03
            # 3. |2Q(clean) - 2Q(hot)| <= 4
            matched = [
                c for c in clean_candidates
                if c["esp"] <= best_hot_esp and c["esp"] >= (best_hot_esp - 0.03) and abs(c["diff_2q"]) <= 4
            ]

            if matched:
                matched.sort(key=lambda x: x["esp"], reverse=True)
                chosen = matched[0]
                status = "PRIMARY"
                is_confounded = False
                conf_reason = "Matched (|Delta 2Q| <= 4, ESP <= HOT and within 0.03)"
                is_executed = True
            else:
                is_executed = False
                non_conf_candidates = [c for c in clean_candidates if abs(c["diff_2q"]) <= 4]
                if non_conf_candidates:
                    status = "UNMATCHABLE"
                    is_confounded = False
                    conf_reason = "No clean subgraph with ESP <= HOT within 0.03"
                    chosen = non_conf_candidates[0]
                else:
                    status = "CONFOUNDED"
                    is_confounded = True
                    conf_reason = "|Delta 2Q| > 4 for all clean subgraphs"
                    chosen = best_unconstrained

            best_clean_lay = chosen["layout"] if chosen else None
            best_clean_tc = chosen["transpiled"] if chosen else None
            best_clean_esp = chosen["esp"] if chosen else -1.0
            clean_2q = chosen["clean_2q"] if chosen else 0
            diff_2q = clean_2q - hot_2q
            clean_depth = chosen["depth"] if chosen else 0
            hot_depth = best_hot_tc.depth()

            # 3. What thermal ESP (measured excess) would choose in the region
            best_th_esp = -1.0
            best_th_lay = None
            for sg in sgs:
                tc = transpile(b_circ, target=target, initial_layout=sg, routing_method='sabre', optimization_level=3, seed_transpiler=42)
                esp_th = esp_thermal(tc, target, p1=p1_est, mode='excess_over_readout')
                if esp_th > best_th_esp:
                    best_th_esp = esp_th; best_th_lay = sg

            th_equals_clean = (best_th_lay == best_clean_lay)

            region_entry["benchmarks"][b_name] = {
                "status": status,
                "executed": is_executed,
                "is_primary": (status == "PRIMARY"),
                "hot_layout": best_hot_lay,
                "clean_layout": best_clean_lay,
                "thermal_esp_layout": best_th_lay,
                "thermal_equals_clean": th_equals_clean,
                "hot_esp": best_hot_esp,
                "clean_esp": best_clean_esp,
                "hot_2q": hot_2q,
                "clean_2q": clean_2q,
                "diff_2q": diff_2q,
                "confounded": is_confounded,
                "confounded_reason": conf_reason,
                "hot_depth": hot_depth,
                "clean_depth": clean_depth,
                "diff_depth": clean_depth - hot_depth,
                "hot_transpiled": best_hot_tc,
                "clean_transpiled": best_clean_tc,
                "logical_circuit": b_circ,
            }

        regions.append(region_entry)

    return regions, log_messages, selected_hot

def parse_iso_ts(ts_str):
    try:
        return datetime.datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

def write_confounding_table(plan_dict, out_path="results/hardware_extra/confounding_table.txt"):
    lines = []
    lines.append("=" * 95)
    lines.append("HARDWARE EXTRA COMPARATOR & CONFOUNDING TABLE (CHECKPOINT V / v2.1)")
    lines.append(f"Backend: {plan_dict.get('backend')} | Calibration: {plan_dict.get('calibration_timestamp')}")
    lines.append("=" * 95)
    hdr = f"{'Region':<7} {'Bench':<7} {'HOT ESP':<9} {'CLN ESP':<9} {'hot_2q':<7} {'cln_2q':<7} {'d2q':<6} {'Status':<13} {'Execution':<13} {'Reason'}"
    lines.append(hdr)
    lines.append("-" * 95)
    for r in plan_dict.get("regions", []):
        r_id = r["region_id"]
        for b_name, b_val in r.get("benchmarks", {}).items():
            diff = b_val.get("diff_2q", 0)
            diff_str = f"{diff:+d}"
            status = b_val.get("status", "PRIMARY" if not b_val.get("confounded") else "CONFOUNDED")
            exec_str = "EXECUTED" if b_val.get("executed", False) else "NOT EXECUTED"
            reason_str = b_val.get("confounded_reason", "None")
            h_esp = f"{b_val.get('hot_esp', 0.0):.4f}"
            c_esp = f"{b_val.get('clean_esp', 0.0):.4f}"
            lines.append(
                f"R{r_id:<6} {b_name:<7} {h_esp:<9} {c_esp:<9} {b_val.get('hot_2q', 0):<7} {b_val.get('clean_2q', 0):<7} {diff_str:<6} {status:<13} {exec_str:<13} {reason_str}"
            )
    lines.append("-" * 95)
    table_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table_text)
    return table_text

def run_hotspot(args):
    config = load_config()
    shots = int(config.get("shots", 8192))
    repetitions = int(config.get("repetitions", 2))
    allowance_remaining = float(config.get("allowance_remaining_s", 279.0))
    reserve_s = float(config.get("reserve_s", 150.0))
    extra_run_cap_s = float(config.get("extra_run_cap_s", 110.0))

    # Step 1.5: Target mode resolution
    target_mode = getattr(args, "target", "fake")
    expected_backend = config.get("extra_backend", "ibm_marrakesh")
    persistent_hot_qubits = None

    backend = get_hotspot_backend(config, local=args.local, target=target_mode)
    backend_name = backend.name
    cal_timestamp = get_calibration_timestamp(backend)
    target = backend.target

    # Step 1.6: Allowance checked timestamp verification
    checked_at_str = config.get("allowance_checked_at")
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    if not checked_at_str:
        if target_mode == "live" or args.execute:
            raise ValueError("Refusal: 'allowance_checked_at' missing in config.yaml. Dashboard allowance must be verified within 24 hours.")
        else:
            print("  [WARNING] 'allowance_checked_at' missing in config.yaml.")
            is_allowance_stale = True
    else:
        checked_at_dt = parse_iso_ts(checked_at_str)
        allowance_age_sec = abs((now_utc - checked_at_dt).total_seconds())
        is_allowance_stale = (allowance_age_sec > 86400)
        if target_mode == "live" or args.execute:
            if is_allowance_stale:
                raise ValueError(
                    f"Refusal: 'allowance_checked_at' ({checked_at_str}) is older than 24 hours "
                    f"({allowance_age_sec/3600:.1f} h ago). Re-enter dashboard allowance manually in config.yaml."
                )
        else:
            if is_allowance_stale:
                print(f"  [WARNING] Stored allowance was checked at {checked_at_str} ({allowance_age_sec/3600:.1f} h ago > 24h).")

    # Step 2: p1 data loading and validation
    if target_mode == "live":
        if args.local:
            raise ValueError("Refusal: Cannot use --local with --target live. Live target requires connection to IBM Quantum.")

        import glob
        p1v2_files = sorted(glob.glob("results/hardware_v2/p1v2_*.json"))
        if not p1v2_files:
            raise ValueError(f"Refusal: No p1v2 characterization file found in results/hardware_v2/ for backend '{backend_name}'. Run scripts/measure_p1_v2.py first.")
        newest_p1_file = Path(p1v2_files[-1])
        with open(newest_p1_file, "r", encoding="utf-8") as f:
            p1_data = json.load(f)

        p1_backend = p1_data.get("backend", "")
        if p1_backend.lower().startswith("fake") or "fake" in str(newest_p1_file).lower() or "offline_rehearsal" in str(newest_p1_file):
            raise ValueError(f"Refusal: Cannot build live plan from a fake p1v2 file ({newest_p1_file}). Run scripts/measure_p1_v2.py on live target first.")

        if p1_backend != backend_name and p1_backend != expected_backend:
            raise ValueError(f"Refusal: p1v2 backend '{p1_backend}' does not match target backend '{backend_name}'.")

        p1_exec_ts = parse_iso_ts(p1_data.get("execution_timestamp", ""))
        if abs((now_utc - p1_exec_ts).total_seconds()) > 86400:
            raise ValueError(f"Refusal: p1v2 file '{newest_p1_file}' is older than 24 hours. Re-run scripts/measure_p1_v2.py.")

        p1_cal_ts = p1_data.get("calibration_timestamp")
        cal_dt = parse_iso_ts(cal_timestamp)
        cal_age_h = (now_utc - cal_dt).total_seconds() / 3600.0
        next_cal_dt = cal_dt + datetime.timedelta(days=1)
        print("=" * 80)
        print("LIVE TARGET CALIBRATION METRICS:")
        print(f"  Live Target Calibration:    {cal_timestamp}")
        print(f"  Calibration Age:            {cal_age_h:.2f} hours")
        print(f"  Estimated Next Calibration: {next_cal_dt.isoformat()}")
        print("=" * 80)

        if p1_cal_ts != cal_timestamp:
            raise ValueError(
                f"Refusal: p1v2 calibration timestamp '{p1_cal_ts}' "
                f"does not match live target calibration timestamp '{cal_timestamp}'. Re-run scripts/measure_p1_v2.py on the fresh calibration."
            )

        from scripts.analyze_p1_v2 import analyze_p1_v2
        analysis_res = analyze_p1_v2(str(newest_p1_file))
        persistent_hot = analysis_res.get("persistent_hot", [])
        if len(persistent_hot) < 1:
            print("\n[REFUSAL RESULT] Zero persistent hot qubits identified under p1v2 characterization.")
            print("Note: The non-persistence of hot spots across isolated blocks is itself an empirical finding.")
            raise ValueError("Refusal: Cannot build hot-spot plan because zero persistent hot qubits were found in p1v2.")

        persistent_hot_qubits = persistent_hot
        p1_ts_str = p1_data.get("execution_timestamp", "")
    else:
        # Default fake target (FakeMarrakesh)
        if args.execute and not args.local:
            raise ValueError("Refusal: Execution on fake target is prohibited. Live hardware execution requires --target live.")
        newest_p1_file, p1_data, p1_ts_str = get_newest_p1_data(backend_name)
        p1_backend = p1_data.get("backend", "")

    # Item 3: Live backend assertions
    if target_mode == "live" and not args.local:
        if p1_backend != expected_backend:
            raise ValueError(
                f"Refusal: Newest p1 file backend '{p1_backend}' != extra_backend '{expected_backend}'."
            )
        dt_cal = parse_iso_ts(cal_timestamp)
        today_utc = datetime.datetime.now(datetime.timezone.utc).date()
        today_local = datetime.date.today()
        cal_date = dt_cal.date()
        if cal_date != today_local and cal_date != today_utc:
            raise ValueError(
                f"Refusal: Backend calibration date ({cal_date}) is not today's date ({today_local}). Live run requires today's calibration."
            )

    pilot_scale_path = Path("results/hardware_extra/pilot_scale.json")
    plan_path = Path("results/hardware_extra/plan.json")

    # Fast early refusal gates for --execute mode
    if args.execute:
        # 1. Pilot file existence and validation
        if not pilot_scale_path.exists():
            raise ValueError(f"Refusal: Pilot file '{pilot_scale_path}' missing. Run --pilot first.")
        with open(pilot_scale_path, "r", encoding="utf-8") as f:
            pilot_data = json.load(f)

        # 2. Plan file existence (needed for plan created_timestamp comparison)
        if not plan_path.exists():
            raise ValueError(f"Refusal: Plan file '{plan_path}' missing. Run dry run first.")
        with open(plan_path, "r", encoding="utf-8") as f:
            saved_plan = json.load(f)

        # Pilot Gate: same backend as extra_backend, non-empty job_id, within 24 h, timestamp after backend calibration
        pilot_backend = pilot_data.get("backend")
        if not pilot_backend:
            raise ValueError("Refusal: Pilot file missing 'backend'.")
        if not args.local:
            if pilot_backend != expected_backend:
                raise ValueError(
                    f"Refusal: Pilot backend '{pilot_backend}' does not match extra_backend '{expected_backend}'."
                )
        else:
            if pilot_backend != backend_name and pilot_backend != expected_backend:
                raise ValueError(
                    f"Refusal: Pilot backend '{pilot_backend}' does not match target backend '{backend_name}'."
                )

        if not pilot_data.get("job_id") or not str(pilot_data.get("job_id")).strip():
            raise ValueError("Refusal: Pilot file missing job_id. Real QPU pilot run required.")

        pilot_ts = float(pilot_data.get("timestamp", 0.0))
        cal_dt = parse_iso_ts(cal_timestamp)
        cal_epoch = cal_dt.timestamp()
        if pilot_ts <= cal_epoch:
            raise ValueError(
                f"Refusal: Pilot timestamp ({pilot_ts}) is older than or equal to backend calibration timestamp ({cal_epoch} / {cal_timestamp}). Run --pilot after backend calibration."
            )

        if abs(time.time() - pilot_ts) > 86400:
            raise ValueError(f"Refusal: Pilot scale file is older than 24 hours. Run --pilot again.")

        # 3. Calibration timestamp unchanged
        if cal_timestamp != saved_plan.get("calibration_timestamp"):
            raise ValueError(
                f"Refusal: Calibration timestamp changed since planning "
                f"({cal_timestamp} != {saved_plan.get('calibration_timestamp')}). Re-run dry run."
            )

        # 4. Approval seconds exact match
        required_approve = math.ceil(float(saved_plan.get("total_estimate_seconds", 108.0)))
        if args.approve_seconds is None:
            raise ValueError(f"Refusal: Must provide --approve-seconds {required_approve}.")
        if int(args.approve_seconds) != required_approve:
            raise ValueError(
                f"Refusal: --approve-seconds {args.approve_seconds} does not match required {required_approve}."
            )

        # 5. Allowance reserve buffer check
        pilot_used_time = float(pilot_data.get("pilot_usage", 4.0))
        p1v2_num_pubs = 24 if target.num_qubits >= 127 else 20
        scale = get_scale_estimate()
        p1_est_time = p1v2_num_pubs * (scale * shots)
        projected_spend = pilot_used_time + p1_est_time + float(saved_plan.get("total_estimate_seconds", 108.0))
        if allowance_remaining - projected_spend < reserve_s:
            raise ValueError(
                f"Refusal: Insufficient allowance. Remaining ({allowance_remaining:.1f} s) - "
                f"projected spend ({projected_spend:.1f} s) < reserve ({reserve_s:.1f} s)."
            )

    # Layout Selection
    regions, log_messages, selected_hot = select_hotspot_regions_and_layouts(
        backend, p1_data, config, strict_d2=args.strict_d2, persistent_hot_qubits=persistent_hot_qubits
    )

    # Build circuit list for execution / de-dup:
    # DROP confounded and unmatchable cells from execution (only execute PRIMARY cells)
    circuit_items = []
    pm = generate_preset_pass_manager(target=target, optimization_level=0)

    for reg in regions:
        r_id = reg["region_id"]
        h_q = reg["hot_qubit"]
        for b_name, b_data in reg["benchmarks"].items():
            if not b_data.get("executed", False):
                continue
            # HOT circuit
            isa_hot = pm.run(b_data["hot_transpiled"])
            validate_isa(isa_hot, target)
            circuit_items.append({
                "region_id": r_id,
                "hot_qubit": h_q,
                "benchmark": b_name,
                "condition": "HOT",
                "layout": b_data["hot_layout"],
                "transpiled_circuit": isa_hot,
            })
            # CLEAN circuit
            isa_clean = pm.run(b_data["clean_transpiled"])
            validate_isa(isa_clean, target)
            circuit_items.append({
                "region_id": r_id,
                "hot_qubit": h_q,
                "benchmark": b_name,
                "condition": "CLEAN",
                "layout": b_data["clean_layout"],
                "transpiled_circuit": isa_clean,
            })

    # De-duplication mapping
    unique_circuits = []
    unique_signatures = {}
    circuit_to_unique_idx = []

    for item in circuit_items:
        sig = tuple(
            (inst.operation.name, tuple(item['transpiled_circuit'].find_bit(q).index for q in inst.qubits))
            for inst in item['transpiled_circuit'].data
        )
        if sig not in unique_signatures:
            unique_signatures[sig] = len(unique_circuits)
            unique_circuits.append(item['transpiled_circuit'])
        circuit_to_unique_idx.append(unique_signatures[sig])

    count_before = len(circuit_items)
    count_after = len(unique_circuits)

    # Scale and time estimation
    pilot_scale_path = Path("results/hardware_extra/pilot_scale.json")
    if pilot_scale_path.exists():
        with open(pilot_scale_path, "r", encoding="utf-8") as f:
            ps_data = json.load(f)
        scale_per_shot = ps_data.get("qpu_seconds_per_shot", 3.0 / shots)
        pilot_usage = ps_data.get("pilot_usage", 4.0)
    else:
        # Check standard pilot scale
        std_scale = Path("results/hardware/qpu_scale.json")
        if std_scale.exists():
            with open(std_scale, "r", encoding="utf-8") as f:
                ps_data = json.load(f)
            scale_per_shot = ps_data.get("qpu_seconds_per_shot", 3.0 / shots)
            pilot_usage = ps_data.get("pilot_usage", 4.0)
        else:
            scale_per_shot = 3.0 / shots
            pilot_usage = 4.0

    est_per_circuit = shots * scale_per_shot
    est_per_job = count_after * est_per_circuit
    total_estimate_seconds = est_per_job * repetitions

    # Dynamic measure_p1_v2 dry-run estimate (24 PUBs for heavy-hex 156q, 20 PUBs for 5q)
    p1v2_num_pubs = 24 if target.num_qubits >= 127 else 20
    p1v2_estimate_seconds = p1v2_num_pubs * est_per_circuit

    spend_scen1 = pilot_usage + p1v2_estimate_seconds
    rem_scen1 = allowance_remaining - spend_scen1

    spend_scen2 = pilot_usage + p1v2_estimate_seconds + total_estimate_seconds
    rem_scen2 = allowance_remaining - spend_scen2

    buffer_scen3 = rem_scen2 - reserve_s
    remaining_after_run = rem_scen2

    # Power gate count
    num_primary_cells = sum(
        1 for reg in regions
        for b_name, b_data in reg["benchmarks"].items()
        if b_data.get("status") == "PRIMARY"
    )
    if num_primary_cells < 4:
        power_status = f"INCONCLUSIVE (underpowered: {num_primary_cells} primary cells < 4 required)"
    else:
        power_status = f"ADEQUATE ({num_primary_cells} primary cells >= 4 required)"

    current_hashes = [hash_circuit(c) for c in unique_circuits]

    # Save plan.json
    plan_path = Path("results/hardware_extra/plan.json")
    plan_path.parent.mkdir(parents=True, exist_ok=True)

    # Serializable region layout data
    serializable_regions = []
    for r in regions:
        r_copy = {k: v for k, v in r.items() if k != "benchmarks"}
        r_copy["benchmarks"] = {}
        for b_name, b_val in r["benchmarks"].items():
            r_copy["benchmarks"][b_name] = {
                k: v for k, v in b_val.items() if not k.endswith("_transpiled") and k != "logical_circuit"
            }
        serializable_regions.append(r_copy)

    # Enforce extra_run_cap_s in dry run (Item 2)
    if total_estimate_seconds > extra_run_cap_s:
        raise ValueError(
            f"Refusal: Total estimated runtime ({total_estimate_seconds:.1f} s) exceeds extra_run_cap_s ({extra_run_cap_s:.1f} s)."
        )

    # Budget gate for execution mode or live target
    if (target_mode == "live" or args.execute) and buffer_scen3 < 0:
        raise ValueError(
            f"Refusal: Insufficient allowance. Remaining ({allowance_remaining:.1f} s) - "
            f"projected spend ({spend_scen2:.1f} s) < reserve ({reserve_s:.1f} s). "
            f"Buffer is {buffer_scen3:.1f} s."
        )

    plan_dict = {
        "created_timestamp": 1760164800.0,
        "created_at": "2026-10-11T12:00:00+00:00",
        "excess_metric_label": "excess P(1|0) over isolated control qubits median or calibrated prob_meas1_prep0",
        "confounding_rule": "Cells with |native 2q difference HOT vs CLEAN| > 4 or unmatchable ESP are dropped from execution.",
        "decision_rule": "hot-spot penalty confirmed for a benchmark only if among primary cells (matched & non-confounded), pooled CLEAN - HOT Newcombe 95% CI lower bound > 0 and diff >= 0.5 pt in >= 50% of cells with N_primary >= 4.",
        "comparator_rule": "CLEAN_matched = argmax ESP_std among clean connected 5q subgraphs in B_4(h) with ESP_std <= ESP_std(HOT), within 0.03, and |Delta 2Q| <= 4; else UNMATCHABLE (not executed).",
        "confounding_threshold_2q": 4,
        "power_status": power_status,
        "primary_cell_count": num_primary_cells,
        "backend": backend_name,
        "calibration_timestamp": cal_timestamp,
        "p1_source_file": str(newest_p1_file).replace("\\", "/"),
        "p1_timestamp": p1_ts_str,
        "circuit_hashes": current_hashes,
        "circuit_count_before": count_before,
        "circuit_count_after": count_after,
        "per_circuit_estimate_seconds": est_per_circuit,
        "per_job_estimate_seconds": est_per_job,
        "total_estimate_seconds": total_estimate_seconds,
        "p1v2_estimate_seconds": p1v2_estimate_seconds,
        "allowance_remaining_s": allowance_remaining,
        "reserve_s": reserve_s,
        "extra_run_cap_s": extra_run_cap_s,
        "remaining_after_run_s": remaining_after_run,
        "budget_scenarios": {
            "scenario1_p1v2_only": {"spend": spend_scen1, "remaining": rem_scen1},
            "scenario2_p1v2_plus_plan": {"spend": spend_scen2, "remaining": rem_scen2},
            "scenario3_with_reserve": {"spend": spend_scen2, "remaining_buffer": buffer_scen3, "reserve_s": reserve_s}
        },
        "shots": shots,
        "repetitions": repetitions,
        "regions": serializable_regions,
    }
    plan_dict["plan_hash"] = compute_plan_hash(plan_dict)

    if not args.execute and not args.pilot:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_dict, f, indent=2)
        write_confounding_table(plan_dict)

    # Generate printable summary report
    lines = []
    lines.append("=" * 80)
    if target_mode != "live":
        lines.append("[OFFLINE DRY RUN ON FakeMarrakesh - NOT EVIDENCE - DO NOT CITE]")
    lines.append("PRE-REGISTERED HARDWARE HOT-SPOT STRESS TEST: DRY RUN & EXECUTION PLAN")
    lines.append(f"Timestamp: {datetime.datetime.now(datetime.timezone.utc).isoformat()}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("1. BACKEND & CALIBRATION METADATA")
    lines.append("-" * 80)
    lines.append(f"  Backend:                  {backend_name}")
    lines.append(f"  Calibration Timestamp:    {cal_timestamp}")
    lines.append(f"  Newest p1 Source:         {newest_p1_file.as_posix()}")
    lines.append(f"  Excess Metric Label:      excess P(1|0) over control median or calibrated prob_meas1_prep0")
    lines.append("")
    lines.append("2. REGION AND LAYOUT SELECTION LOGS")
    lines.append("-" * 80)
    for msg in log_messages:
        lines.append(f"  {msg}")
    lines.append("")
    lines.append("3. PER-REGION LAYOUT & TRANSPILATION TABLE")
    lines.append("-" * 80)
    lines.append(f"  Power Gate Status:        {power_status}")
    lines.append(f"  Primary Cells Count:      {num_primary_cells}")
    lines.append("-" * 80)
    if not regions:
        lines.append("  [NOTE] Zero regions survived layout filtering (all candidate regions dropped).")
    else:
        hdr = f"{'Region':<7} {'Bench':<7} {'HOT Layout':<18} {'CLN Layout':<18} {'ESP(Hot)':<9} {'ESP(Cln)':<9} {'2Q(H/C)':<9} {'Status':<13} {'Execution':<13}"
        lines.append(hdr)
        lines.append("-" * 115)
        for r in regions:
            r_id = r["region_id"]
            for b_name, b_val in r["benchmarks"].items():
                hot_lay_s = str(b_val["hot_layout"])
                cln_lay_s = str(b_val["clean_layout"])
                q2_s = f"{b_val['hot_2q']}/{b_val['clean_2q']} ({b_val['diff_2q']:+d})"
                status_s = b_val.get("status", "PRIMARY")
                exec_s = "EXECUTED" if b_val.get("executed", False) else "NOT EXECUTED"
                lines.append(
                    f"R{r_id:<6} {b_name:<7} {hot_lay_s:<18} {cln_lay_s:<18} "
                    f"{b_val['hot_esp']:<9.4f} {b_val['clean_esp']:<9.4f} "
                    f"{q2_s:<9} {status_s:<13} {exec_s:<13}"
                )
        lines.append("-" * 115)
    lines.append("")
    lines.append("4. CIRCUIT COUNT & BUDGET SCENARIOS (Checkpoint V)")
    lines.append("-" * 80)
    lines.append(f"  Circuit Count Before De-dup:   {count_before}")
    lines.append(f"  Circuit Count After De-dup:    {count_after}")
    lines.append(f"  Estimated Seconds Per Circuit: {est_per_circuit:.3f} s")
    lines.append(f"  Estimated Seconds Per Job:     {est_per_job:.3f} s ({count_after} circuits x {est_per_circuit:.3f} s)")
    lines.append(f"  Total Estimated Plan Seconds:  {total_estimate_seconds:.3f} s ({repetitions} reps)")
    lines.append(f"  Extra Run Budget Cap:          {extra_run_cap_s:.1f} s")
    lines.append(f"  Cap Verification:              {'PASS' if total_estimate_seconds <= extra_run_cap_s else 'FAIL'} ({total_estimate_seconds:.1f} s <= {extra_run_cap_s:.1f} s)")
    lines.append(f"  Dashboard Allowance Remaining: {allowance_remaining:.1f} s")
    lines.append(f"  Reserve Buffer Requirement:    {reserve_s:.1f} s")
    lines.append(f"  measure_p1_v2 Dry-Run Estimate:{p1v2_estimate_seconds:.1f} s ({p1v2_num_pubs} PUBs @ {est_per_circuit:.2f} s/PUB)")
    lines.append("")
    lines.append("  BUDGET SCENARIOS:")
    lines.append(f"  - Scenario 1 (p1v2 only):      Spend: {spend_scen1:.1f} s | Remaining: {rem_scen1:.1f} s")
    lines.append(f"  - Scenario 2 (p1v2 + plan):    Spend: {spend_scen2:.1f} s | Remaining: {rem_scen2:.1f} s")
    lines.append(f"  - Scenario 3 (with reserve):   Remaining Buffer after {reserve_s:.0f}s reserve: {buffer_scen3:+.1f} s ({'PASS' if buffer_scen3 >= 0 else 'FAIL'})")
    lines.append("")
    lines.append("5. PLAN INTEGRITY & ARTIFACTS")
    lines.append("-" * 80)
    lines.append(f"  Plan Path:                     {plan_path.as_posix()}")
    hyphenated_plan_hash = "-".join(plan_dict['plan_hash'][i:i+16] for i in range(0, len(plan_dict['plan_hash']), 16))
    lines.append(f"  Plan Hash:                     {hyphenated_plan_hash}")
    lines.append("=" * 80)

    summary_text = "\n".join(lines) + "\n"
    summary_path = Path("results/hardware_extra/hotspot_plan_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print(summary_text)

    # -------------------------------------------------------------------------
    # Mode Handling: --pilot
    # -------------------------------------------------------------------------
    if args.pilot:
        if not unique_circuits:
            raise ValueError("Refusal: Cannot run pilot because zero unique circuits exist in the plan.")
        print("\n[PILOT MODE] Running ONE circuit to establish empirical shot scaling...")
        pilot_circuit = unique_circuits[0]

        if not args.local:
            service = get_ibm_service()
            sampler = Sampler(mode=backend)
            sampler.options.dynamical_decoupling.enable = False
            sampler.options.twirling.enable_gates = False
            job = sampler.run([pilot_circuit], shots=shots)
            j_id = job.job_id()
            print(f"Pilot Job ID: {j_id}")
            job.result()
            usage = extract_qpu_seconds(job, 3.0)
        else:
            print("[PILOT LOCAL REHEARSAL] Simulating pilot run on local backend.")
            j_id = f"pilot_local_{int(time.time())}"
            usage = 3.0

        pilot_scale_data = {
            "backend": backend_name,
            "job_id": j_id,
            "timestamp": time.time(),
            "iso_time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "qpu_seconds_per_shot": usage / shots,
            "pilot_usage": usage,
            "pilot_shots": shots,
            "circuit_hash": hash_circuit(pilot_circuit)
        }
        with open(pilot_scale_path, "w", encoding="utf-8") as f:
            json.dump(pilot_scale_data, f, indent=2)
        print(f"Pilot scale saved to {pilot_scale_path} (Job ID: {j_id}, Usage: {usage:.2f} s, Seconds/shot: {usage/shots:.6f}).")
        return

    # -------------------------------------------------------------------------
    # Mode Handling: Dry Run vs Execute
    # -------------------------------------------------------------------------
    if not args.execute and not args.local:
        print("\n[DRY RUN COMPLETE] Plan and layouts pre-registered.")
        p1_approve = math.ceil(4096 * 2 * scale_per_shot)
        p1v2_approve = math.ceil(p1v2_estimate_seconds)
        hw_approve = math.ceil(total_estimate_seconds)
        print("\nExact operational command sequence with calculated approval budgets:")
        print(f"  python scripts/measure_p1_v2.py --execute --approve-seconds <N>")
        print(f"  python scripts/analyze_p1_v2.py")
        print(f"  # If zero persistent hot qubits, STOP and report non-persistence as the finding.")
        print(f"  python scripts/run_hotspot_hw.py --target live --dry-run")
        print(f"  python scripts/run_hotspot_hw.py --target live --execute --approve-seconds <M>")
        print(f"  python scripts/analyze_hotspot_v2.py")
        return

    # -------------------------------------------------------------------------
    # Mode Handling: --execute (Gated Verification)
    # -------------------------------------------------------------------------
    if args.execute:
        print("\n[EXECUTE MODE GATED CHECKS]")

        # 1. Pilot file existence and 24h validity
        if not pilot_scale_path.exists():
            raise ValueError(f"Refusal: Pilot file '{pilot_scale_path}' missing. Run --pilot first.")
        with open(pilot_scale_path, "r", encoding="utf-8") as f:
            pilot_data = json.load(f)
        if time.time() - pilot_data.get("timestamp", 0) > 86400:
            raise ValueError(f"Refusal: Pilot scale file is older than 24 hours. Run --pilot again.")
        print("  [PASS] Pilot scale file verified (< 24h old).")

        # 2. Plan file existence and hash integrity
        if not plan_path.exists():
            raise ValueError(f"Refusal: Plan file '{plan_path}' missing. Run dry run first.")
        with open(plan_path, "r", encoding="utf-8") as f:
            saved_plan = json.load(f)
        if plan_dict["plan_hash"] != saved_plan.get("plan_hash"):
            raise ValueError("Refusal: Current plan hash does not match saved results/hardware_extra/plan.json!")
        if current_hashes != saved_plan.get("circuit_hashes"):
            raise ValueError("Refusal: Current transpiled circuit hashes differ from saved plan.json!")
        print("  [PASS] Plan hash and circuit hash integrity verified.")

        # 3. Calibration timestamp unchanged
        if cal_timestamp != saved_plan.get("calibration_timestamp"):
            raise ValueError(
                f"Refusal: Calibration timestamp changed since planning "
                f"({cal_timestamp} != {saved_plan.get('calibration_timestamp')}). Re-run dry run."
            )
        print("  [PASS] Calibration timestamp unchanged since planning.")

        # 4. Approval seconds exact match
        required_approve = math.ceil(total_estimate_seconds)
        if args.approve_seconds is None:
            raise ValueError(f"Refusal: Must provide --approve-seconds {required_approve}.")
        if int(args.approve_seconds) != required_approve:
            raise ValueError(
                f"Refusal: --approve-seconds {args.approve_seconds} does not match required {required_approve}."
            )
        print(f"  [PASS] Approved seconds ({args.approve_seconds}) exactly matches estimate ({required_approve}).")

        # 5. Allowance reserve buffer check
        pilot_used_time = float(pilot_data.get("pilot_usage", 4.0))
        p1_est_time = 4.0
        projected_spend = pilot_used_time + p1_est_time + total_estimate_seconds
        if allowance_remaining - projected_spend < reserve_s:
            raise ValueError(
                f"Refusal: Insufficient allowance. Remaining ({allowance_remaining:.1f} s) - "
                f"projected spend ({projected_spend:.1f} s) < reserve ({reserve_s:.1f} s)."
            )
        print(f"  [PASS] Projected remaining allowance ({allowance_remaining - projected_spend:.1f} s) >= reserve ({reserve_s:.1f} s).")

        if not args.local:
            print("\nSubmitting jobs to QPU...")
            sampler = Sampler(mode=backend)
            sampler.options.dynamical_decoupling.enable = False
            sampler.options.twirling.enable_gates = False

            raw_dir = Path("results/hardware_extra/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            job_ids = []

            for rep in range(repetitions):
                print(f"Submitting repetition {rep} ({count_after} unique circuits)...")
                job = sampler.run(unique_circuits, shots=shots)
                j_id = job.job_id()
                job_ids.append(j_id)
                print(f"  Job ID (rep {rep}): {j_id}")
                res = job.result()

                for item_idx, item in enumerate(circuit_items):
                    u_idx = circuit_to_unique_idx[item_idx]
                    pub_res = res[u_idx].data
                    counts = pub_res.meas.get_counts() if hasattr(pub_res, 'meas') else pub_res.c.get_counts()
                    raw_fn = raw_dir / f"r{item['region_id']}_{item['benchmark']}_{item['condition']}_rep{rep}.json"
                    with open(raw_fn, "w", encoding="utf-8") as f:
                        json.dump(counts, f, indent=2)

            print("\nExecution complete. Raw counts written to results/hardware_extra/raw/.")
        else:
            print("\n[LOCAL REHEARSAL COMPLETE] All gated refusal checks verified successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run Hotspot Hardware Stress Test")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and write results/hardware_extra/plan.json (default)")
    parser.add_argument("--target", choices=["fake", "live"], default="fake", help="Target mode: fake (offline rehearsal on FakeMarrakesh) or live (real QPU target with p1v2)")
    parser.add_argument("--pilot", action="store_true", help="Run 1 circuit to calibrate timing scale")
    parser.add_argument("--execute", action="store_true", help="Execute planned circuits with gating checks")
    parser.add_argument("--approve-seconds", type=int, default=None, help="Approved budget seconds for execution")
    parser.add_argument("--local", action="store_true", help="Run locally using FakeMarrakesh without submitting to live QPU")
    parser.add_argument("--strict-d2", action="store_true", help="Strictly enforce distance <= 2 (dropping regions with no clean 5q subgraphs)")
    args = parser.parse_args()

    run_hotspot(args)

if __name__ == "__main__":
    main()

