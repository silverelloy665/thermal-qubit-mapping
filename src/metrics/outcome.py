import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, hellinger_fidelity

def compute_ideal_distribution(circuit: QuantumCircuit) -> dict[str, float]:
    """
    Computes exact ideal distribution via Statevector of the logical circuit.
    Removes final measurements and filters probabilities <= 1e-10.
    """
    qc_no_meas = circuit.remove_final_measurements(inplace=False)
    sv = Statevector(qc_no_meas)
    probs = sv.probabilities_dict()
    return {str(k): float(v) for k, v in probs.items() if v > 1e-10}

def extract_bv_secret(circuit: QuantumCircuit) -> str:
    """
    Extracts the secret bitstring from a Bernstein-Vazirani QuantumCircuit.
    In the standard BV pattern, the ancilla qubit is the last qubit (index num_clbits),
    and CX(i, ancilla) gates indicate that the i-th bit (from right, 0-indexed) is '1'.
    """
    num_clbits = circuit.num_clbits
    bits = ['0'] * num_clbits
    ancilla_idx = num_clbits
    for inst in circuit.data:
        if inst.operation.name == 'cx':
            ctrl = circuit.find_bit(inst.qubits[0]).index
            tgt = circuit.find_bit(inst.qubits[1]).index
            if tgt == ancilla_idx and ctrl < num_clbits:
                bits[num_clbits - 1 - ctrl] = '1'
    return ''.join(bits)

def get_ideal_support(benchmark_name: str, circuit: QuantumCircuit) -> list[str] | None:
    """
    Returns the ideal support bitstrings for benchmarks whose metric is success_prob:
    - GHZ: P(0..0) + P(1..1)
    - BV: P(secret)
    - mirror: P(0..0)
    Returns None if the benchmark uses Hellinger fidelity vs exact ideal distribution.
    """
    b = benchmark_name.lower().strip()
    if b == 'ghz':
        n = circuit.num_clbits if circuit.num_clbits > 0 else circuit.num_qubits
        return ['0' * n, '1' * n]
    elif b == 'bv':
        secret = extract_bv_secret(circuit)
        return [secret]
    elif b == 'mirror':
        n = circuit.num_clbits if circuit.num_clbits > 0 else circuit.num_qubits
        return ['0' * n]
    return None

def compute_outcome_metric(
    benchmark_name: str,
    counts: dict[str, int | float],
    logical_circuit: QuantumCircuit,
    ideal_distribution: dict[str, float] | None = None,
    shots: int | float | None = None
) -> tuple[str, float]:
    """
    Computes (metric_type, metric_val) for a circuit execution:
    - If benchmark is ghz, bv, or mirror:
      metric_type = 'success_prob'
      metric_val = total probability on ideal support
    - Otherwise:
      metric_type = 'fidelity'
      metric_val = Hellinger fidelity vs exact ideal distribution
    """
    if shots is None:
        shots = sum(counts.values())
    if shots <= 0:
        return ('success_prob' if get_ideal_support(benchmark_name, logical_circuit) is not None else 'fidelity', 0.0)

    support = get_ideal_support(benchmark_name, logical_circuit)
    if support is not None:
        total_prob = sum(counts.get(s, 0) for s in support) / float(shots)
        return ('success_prob', float(np.clip(total_prob, 0.0, 1.0)))
    else:
        if ideal_distribution is None:
            ideal_distribution = compute_ideal_distribution(logical_circuit)
        norm_counts = {str(k): float(v) / float(shots) for k, v in counts.items()}
        fid = hellinger_fidelity(ideal_distribution, norm_counts)
        return ('fidelity', float(fid))

