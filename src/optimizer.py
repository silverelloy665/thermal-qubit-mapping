import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

# ── Fixed linear coupling map ──────────────────────────────
EDGES = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]]
COUPLING = CouplingMap(EDGES)

# ── Thermal data (hardcoded from thermal_noise.csv) ────────
# Sorted best→worst by stability: q2,q0,q4,q1,q3
THERMAL_ORDER = [2, 0, 4, 1, 3]   # physical qubit indices
THERMAL_SCORES = {0: 0.40, 1: 0.65, 2: 0.15, 3: 0.85, 4: 0.50}
# score = 0.5*(temp/20) + 0.5*(error/0.004) from CSV

def count_swaps(circuit):
    """Count swap gates in a transpiled circuit."""
    ops = circuit.count_ops()
    return ops.get('swap', 0)

def compute_noise_score(layout_dict):
    """
    layout_dict: {logical_qubit_int: physical_qubit_int}
    Returns average thermal score of assigned physical qubits.
    """
    if not layout_dict:
        return 0.5
    total = sum(THERMAL_SCORES.get(phys, 0.5) 
                for phys in layout_dict.values())
    return round(total / len(layout_dict), 3)

def run_random_mapping(circuit):
    """Worst-case mapping: reverse order layout."""
    n = circuit.num_qubits
    # Create layout: logical qubits to physical qubits (reversed order = worst)
    qubits = circuit.qubits
    layout = {qubits[i]: (n - 1 - i) for i in range(n)}
    layout_dict = {i: (n - 1 - i) for i in range(n)}  # for noise calculation
    print(f"  Random layout: {layout_dict}")
    
    tc = transpile(
        circuit,
        coupling_map=COUPLING,
        initial_layout=layout,
        routing_method='sabre',
        optimization_level=0,
        seed_transpiler=12345
    )
    swaps = count_swaps(tc)
    noise = compute_noise_score(layout_dict)
    print(f"  Random ops: {tc.count_ops()}")
    print(f"  Random swaps={swaps}, noise={noise}")
    return swaps, noise

def run_sabre_mapping(circuit):
    """SABRE automatic layout + routing."""
    tc = transpile(
        circuit,
        coupling_map=COUPLING,
        routing_method='sabre',
        layout_method='sabre',
        optimization_level=1,
        seed_transpiler=42
    )
    # Extract final layout from transpiled circuit
    try:
        final_layout = tc.layout.final_layout
        if final_layout is not None:
            # final_layout maps Qubit → physical index, we need logical index → physical
            layout_dict = {}
            for logical_idx in range(circuit.num_qubits):
                logical_qubit = circuit.qubits[logical_idx]
                phys_qubit = final_layout[logical_qubit]
                # phys_qubit might be a Qubit or an int, extract index
                if isinstance(phys_qubit, int):
                    layout_dict[logical_idx] = phys_qubit
                else:
                    layout_dict[logical_idx] = phys_qubit.index
        else:
            layout_dict = {i: i for i in range(circuit.num_qubits)}
    except Exception as e:
        print(f"  [Warning] Could not extract SABRE layout: {e}")
        layout_dict = {i: i for i in range(circuit.num_qubits)}
    
    print(f"  SABRE layout: {layout_dict}")
    swaps = count_swaps(tc)
    noise = compute_noise_score(layout_dict)
    print(f"  SABRE ops: {tc.count_ops()}")
    print(f"  SABRE swaps={swaps}, noise={noise}")
    return swaps, noise

def run_thermal_mapping(circuit):
    """Thermal-aware: assign coolest qubits first."""
    n = circuit.num_qubits
    # Map logical 0→best physical, 1→second best, etc.
    qubits = circuit.qubits
    layout = {qubits[i]: THERMAL_ORDER[i] for i in range(n)}
    layout_dict = {i: THERMAL_ORDER[i] for i in range(n)}  # for noise calculation
    print(f"  Thermal layout: {layout_dict}")
    
    tc = transpile(
        circuit,
        coupling_map=COUPLING,
        initial_layout=layout,
        routing_method='sabre',
        optimization_level=1,
        seed_transpiler=42
    )
    swaps = count_swaps(tc)
    noise = compute_noise_score(layout_dict)
    print(f"  Thermal ops: {tc.count_ops()}")
    print(f"  Thermal swaps={swaps}, noise={noise}")
    return swaps, noise

