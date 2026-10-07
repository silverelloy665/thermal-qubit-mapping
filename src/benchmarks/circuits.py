import random
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT, QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp

def get_ghz(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def get_bv(secret_string: str) -> QuantumCircuit:
    n = len(secret_string)
    qc = QuantumCircuit(n + 1, n)
    # init target
    qc.x(n)
    qc.h(range(n + 1))
    qc.barrier()
    for i, bit in enumerate(reversed(secret_string)):
        if bit == '1':
            qc.cx(i, n)
    qc.barrier()
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc

def get_qft(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n)
    # Known input state: prepare |++...++>
    qc.h(range(n))
    qc.compose(QFT(num_qubits=n, approximation_degree=0, do_swaps=True), inplace=True)
    qc.measure_all()
    return qc

def get_qaoa(n: int) -> QuantumCircuit:
    # QAOA on a ring graph
    op_list = []
    for i in range(n):
        pauli_str = ['I'] * n
        pauli_str[i] = 'Z'
        pauli_str[(i + 1) % n] = 'Z'
        op_list.append("".join(reversed(pauli_str))) # Qiskit endianness
    
    hamiltonian = SparsePauliOp(op_list)
    ansatz = QAOAAnsatz(hamiltonian, reps=1)
    bound = ansatz.assign_parameters([0.5, 0.5])
    qc = QuantumCircuit(n)
    qc.compose(bound, inplace=True)
    qc.measure_all()
    return qc

def get_mirror(n: int, seed: int = 42) -> QuantumCircuit:
    """
    Mirror benchmark circuit:
    Random layers of CX + single-qubit gates on all N qubits with a fixed seed,
    including long-range CX gates that force SWAPs on Vigo and Guadalupe,
    followed by its exact inverse. Ideal output is strictly |0...0>.
    """
    rng = random.Random(seed + n)
    qc = QuantumCircuit(n)
    fwd = QuantumCircuit(n)
    
    # Layer 1: single qubit rotations
    for i in range(n):
        fwd.rx(rng.uniform(0.1, 2.0), i)
        fwd.rz(rng.uniform(0.1, 2.0), i)
        
    # Layer 2: long-range CX forcing SWAPs on linear / T-shaped topologies
    fwd.cx(0, n - 1)
    if n >= 4:
        fwd.cx(1, n - 2)
    if n >= 5:
        fwd.cx(0, n // 2)
        fwd.cx(1, n - 1)
        
    # Layer 3: intermediate single qubit rotations
    for i in range(n):
        fwd.ry(rng.uniform(0.1, 2.0), i)
        
    # Layer 4: nearest-neighbor ladder CX + long-range CX
    for i in range(n - 1):
        fwd.cx(i, i + 1)
    fwd.cx(0, n - 1)
    
    # Layer 5: inverse of forward circuit
    inv = fwd.inverse()
    qc.compose(fwd, inplace=True)
    qc.barrier()
    qc.compose(inv, inplace=True)
    qc.measure_all()
    return qc

# Aliases for backwards compatibility
def get_routing(n: int) -> QuantumCircuit:
    return get_mirror(n)

def get_routing_old(n: int) -> QuantumCircuit:
    return get_mirror(n)

# Fix BV to use n qubits total (n-1 data, 1 target)
def get_bv_n(n: int) -> QuantumCircuit:
    if n < 2:
        return get_ghz(n)
    secret_string = "1" + "0" * (n-3) + "1" if n >= 3 else "1"
    return get_bv(secret_string)

def get_all_benchmarks(n: int):
    return {
        'ghz': get_ghz(n),
        'bv': get_bv_n(n),
        'qft': get_qft(n),
        'mirror': get_mirror(n),
    }
