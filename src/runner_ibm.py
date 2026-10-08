import os
import json
import time
import math
from qiskit_ibm_runtime import QiskitRuntimeService
from pathlib import Path

def get_ibm_service():
    """
    Initialize QiskitRuntimeService using saved account ONLY.
    Never accepts token/CRN.
    """
    instance = os.environ.get("QISKIT_IBM_INSTANCE", None)
    return QiskitRuntimeService(channel="ibm_quantum_platform", instance=instance)

def get_backend(service, config):
    """
    Selects backend pinned in config or falls back to lowest pending jobs.
    """
    pinned = config.get("backend") or config.get("backends", {}).get("backend")
    if pinned:
        backends = service.backends()
        matching = [b for b in backends if b.name == pinned]
        if not matching:
            raise ValueError(f"Pinned backend '{pinned}' not found among available backends.")
        backend = matching[0]
        status = backend.status()
        if not getattr(status, "operational", False):
            raise ValueError(f"Pinned backend '{pinned}' is not operational (status: {status}).")
        return backend

    sel = config.get("backends", {}).get("backend_selection", {})
    min_q = sel.get("min_num_qubits", 127)
    sim = sel.get("simulator", False)
    op = sel.get("operational", True)
    
    backends = service.backends(operational=op, simulator=sim, min_num_qubits=min_q)
    if not backends:
        raise ValueError(f"No backends found matching criteria.")
    
    best_backend = min(backends, key=lambda b: b.status().pending_jobs)
    return best_backend

def get_calibration_timestamp(backend):
    props = backend.properties()
    if props and props.last_update_date:
        return props.last_update_date.isoformat()
    return "unknown_timestamp"

def check_pilot_and_approval(total_estimated_seconds, approve_seconds, scale_file_path="results/hardware/qpu_scale.json"):
    if not os.path.exists(scale_file_path):
        raise ValueError("Pilot scale file not found. Run --pilot first.")
    
    with open(scale_file_path, "r") as f:
        scale_data = json.load(f)
        
    if time.time() - scale_data.get("timestamp", 0) > 86400:
        raise ValueError("Pilot scale file is older than 24h. Run --pilot again.")
        
    if approve_seconds is None:
        raise ValueError(f"Must provide --approve-seconds {math.ceil(total_estimated_seconds)}")
        
    if approve_seconds != math.ceil(total_estimated_seconds):
        raise ValueError(f"--approve-seconds {approve_seconds} must exactly match required {math.ceil(total_estimated_seconds)}")
        
def get_scale_estimate(scale_file_path="results/hardware/qpu_scale.json"):
    if not os.path.exists(scale_file_path):
        return 3.0 / 8192 # fallback per-shot estimate
    with open(scale_file_path, "r") as f:
        scale_data = json.load(f)
    return scale_data.get("qpu_seconds_per_shot", 3.0 / 8192)

def extract_qpu_seconds(job, estimate):
    usage_time = None
    if hasattr(job, 'usage'):
        usage_time = job.usage()
    if isinstance(usage_time, float):
        return usage_time
    print(f"Warning: job.usage() returned None or missing, falling back to estimate {estimate}")
    return estimate
