import sys
import itertools
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from qiskit_aer import AerSimulator
from src.metrics.esp import esp_standard

def run_regression():
    backend = FakeVigoV2()
    target = backend.target
    
    qc = QuantumCircuit(5)
    for i in range(4): qc.cx(i, i+1)
    qc.measure_all()
    
    sim = AerSimulator()
    layouts = list(itertools.permutations(range(5)))
    print(f"Testing {len(layouts)} layouts...")
    
    for perm in layouts:
        tc = transpile(qc, backend, initial_layout=list(perm), optimization_level=0)
        counts = sim.run(tc).result().get_counts()
        fid = counts.get('00000', 0) / sum(counts.values())
        print(f"Layout {perm}: Fidelity {fid:.2f}")
        assert fid == 1.0, f"Fidelity is {fid} for {perm}"
        
    print("All 120 layouts passed with fidelity 1.0.")

if __name__ == '__main__':
    run_regression()
