import sys
from pathlib import Path
from qiskit import QuantumCircuit

sys.path.append(str(Path(__file__).parent.parent))
from src.runner_ibm import get_ibm_service, submit_hardware_job

def build_p1_circuit(num_qubits: int) -> QuantumCircuit:
    \"\"\"Build a circuit that simply delays and then measures all qubits.\"\"\"
    qc = QuantumCircuit(num_qubits)
    # Give some idle time to let qubits decay
    qc.delay(1000, range(num_qubits), unit='dt')
    qc.measure_all()
    return qc

def main(dry_run=True):
    service = get_ibm_service()
    backend_name = "ibm_kyoto" # default or from config
    backend = service.backend(backend_name)
    num_qubits = backend.target.num_qubits
    
    print(f"Building p1 measurement circuit for {backend_name} ({num_qubits} qubits)...")
    qc = build_p1_circuit(num_qubits)
    
    if dry_run:
        print("[DRY RUN] Will not submit. Run with '--execute' to submit to QPU.")
        return
        
    print("Submitting p1 measurement job...")
    job = submit_hardware_job(qc, backend_name, shots=4096, dry_run=False)
    print(f"Job submitted! Job ID: {job.job_id()}")

if __name__ == "__main__":
    dry_run = '--execute' not in sys.argv
    main(dry_run)
