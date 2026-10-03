import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
from src.metrics.esp import esp_standard, esp_thermal, get_p1
from src.mappers.mappers import get_connected_subgraphs

def debug(hot_q):
    backend = FakeVigoV2()
    target = backend.target
    temps_mk = {q: 15 for q in range(target.num_qubits)}
    temps_mk[hot_q] = 120
    
    print(f"\n--- Hot Qubit: Q{hot_q} ---")
    print("p1 per qubit:")
    for q in range(target.num_qubits):
        print(f"  Q{q}: {get_p1(q, target, temps_mk=temps_mk):.4f}")
        
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    
    sgs = get_connected_subgraphs(target, 3)
    best_std = -1.0; argmax_std = None
    best_th = -1.0; argmax_th = None
    
    for sg in sgs:
        tc = transpile(qc, target=target, initial_layout=sg, optimization_level=1, seed_transpiler=42)
        e_std = esp_standard(tc, target)
        e_th = esp_thermal(tc, target, p1=temps_mk, mode='raw_upper_bound')
        
        print(f"Subgraph {sg}: esp_standard={e_std:.4f}, esp_thermal={e_th:.4f}")
        if e_std > best_std: best_std = e_std; argmax_std = sg
        if e_th > best_th: best_th = e_th; argmax_th = sg
        
    print(f"\nArgmax esp_standard: {argmax_std} ({best_std:.4f})")
    print(f"Argmax esp_thermal: {argmax_th} ({best_th:.4f})")
    print(f"Argmax changed: {argmax_std != argmax_th}")

if __name__ == '__main__':
    debug(4)
    debug(0)
