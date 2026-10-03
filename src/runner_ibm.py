import os
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

def get_ibm_service():
    \"\"\"
    Initialize QiskitRuntimeService using saved account.
    Does NOT use .env or print credentials.
    \"\"\"
    try:
        # Use ibm_quantum_platform if available, else ibm_cloud
        service = QiskitRuntimeService(channel='ibm_quantum_platform') # uses saved account
        return service
    except Exception:
        try:
            service = QiskitRuntimeService()
            return service
        except Exception as e:
            raise RuntimeError("Could not initialize QiskitRuntimeService. Please ensure an account is saved locally.") from e

def submit_hardware_job(circuit, backend_name, shots=8192, dry_run=True):
    \"\"\"
    Submit a circuit to IBM Quantum Runtime SamplerV2.
    \"\"\"
    service = get_ibm_service()
    backend = service.backend(backend_name)
    target = backend.target
    
    pm = generate_preset_pass_manager(target=target, optimization_level=0)
    isa_circuit = pm.run(circuit)
    
    if dry_run:
        print(f"[DRY RUN] Would submit circuit to {backend_name} with {shots} shots.")
        return None
        
    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=shots)
    return job
