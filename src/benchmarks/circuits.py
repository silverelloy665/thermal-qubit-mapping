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
    # bind parameters to random values or fixed
    bound = ansatz.assign_parameters([0.5, 0.5])
    qc = QuantumCircuit(n)
    qc.compose(bound, inplace=True)
    qc.measure_all()
    return qc

def get_routing_pressure(n: int) -> QuantumCircuit:
    """Circuit with long range CX gates to stress routing."""
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n // 2):
        qc.cx(i, n - 1 - i)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    qc.measure_all()
    return qc

def get_all_benchmarks(n: int):
    return {
        'ghz': get_ghz(n),
        'bv': get_bv_n(n),
        'qft': get_qft(n),
        'qaoa': get_qaoa(n),
        'routing': get_routing_pressure(n)
    }

# Fix BV to use n qubits total (n-1 data, 1 target)
def get_bv_n(n: int) -> QuantumCircuit:
    if n < 2:
        return get_ghz(n) # fallback
    secret_string = "1" + "0" * (n-3) + "1" if n >= 3 else "1"
    return get_bv(secret_string)
