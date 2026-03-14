
"""
Circuit Loader Module

Provides functions to load various quantum circuits including Bell states, GHZ states,
Quantum Fourier Transform, and random circuits. All circuits include measurement operations.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.random import random_circuit
from typing import Optional


def load_bell_circuit() -> QuantumCircuit:
    """
    Load a 2-qubit Bell state (maximally entangled state).
    
    Creates the circuit: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    
    Returns:
        QuantumCircuit: 2-qubit Bell state circuit with measurements.
    """
    try:
        qc = QuantumCircuit(2, 2, name='Bell')
        qc.h(0)
        qc.cx(0, 1)
        qc.measure(range(2), range(2))
        return qc
    except Exception as e:
        raise RuntimeError(f"Error creating Bell circuit: {str(e)}")


def load_ghz_circuit(n: int) -> QuantumCircuit:
    """
    Load an n-qubit GHZ state (Greenberger-Horne-Zeilinger state).
    
    Creates the circuit: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    
    Args:
        n (int): Number of qubits (must be >= 2).
        
    Returns:
        QuantumCircuit: n-qubit GHZ state circuit with measurements.
        
    Raises:
        ValueError: If n < 2.
    """
    try:
        if n < 2:
            raise ValueError(f"GHZ state requires at least 2 qubits, got {n}")
        
        qc = QuantumCircuit(n, n, name=f'GHZ_{n}')
        qc.h(0)
        
        for i in range(n - 1):
            qc.cx(i, i + 1)
        
        qc.measure(range(n), range(n))
        return qc
    except Exception as e:
        raise RuntimeError(f"Error creating GHZ circuit: {str(e)}")


def load_qft_circuit(n: int) -> QuantumCircuit:
    """
    Load an n-qubit Quantum Fourier Transform (QFT) circuit.
    
    Implements the standard QFT algorithm with all necessary phase rotations.
    
    Args:
        n (int): Number of qubits (must be >= 1).
        
    Returns:
        QuantumCircuit: n-qubit QFT circuit with measurements.
        
    Raises:
        ValueError: If n < 1.
    """
    try:
        if n < 1:
            raise ValueError(f"QFT requires at least 1 qubit, got {n}")
        
        qc = QuantumCircuit(n, n, name=f'QFT_{n}')
        
        # QFT implementation
        for j in range(n):
            qc.h(j)
            for k in range(j + 1, n):
                angle = 2 * np.pi / (2 ** (k - j + 1))
                qc.cp(angle, k, j)
        
        # Swap qubits
        for i in range(n // 2):
            qc.swap(i, n - 1 - i)
        
        qc.measure(range(n), range(n))
        return qc
    except Exception as e:
        raise RuntimeError(f"Error creating QFT circuit: {str(e)}")


def load_random_circuit(n: int, depth: int) -> QuantumCircuit:
    """
    Load a random quantum circuit using Qiskit's random_circuit generator.
    
    Creates a random circuit with specified number of qubits and depth,
    using a variety of gates (RZGate, SXGate, CXGate).
    
    Args:
        n (int): Number of qubits (must be >= 1).
        depth (int): Circuit depth (must be >= 1).
        
    Returns:
        QuantumCircuit: Random quantum circuit with measurements.
        
    Raises:
        ValueError: If n < 1 or depth < 1.
    """
    try:
        if n < 1:
            raise ValueError(f"Random circuit requires at least 1 qubit, got {n}")
        if depth < 1:
            raise ValueError(f"Random circuit requires depth >= 1, got {depth}")
        
        qc = random_circuit(n, depth, measure=False, seed=42)
        qc.name = f'Random_{n}q_{depth}d'
        qc.measure_all()
        return qc
    except Exception as e:
        raise RuntimeError(f"Error creating random circuit: {str(e)}")


def measure_all(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Add measurement operations to all qubits in a circuit if not already present.
    
    Args:
        qc (QuantumCircuit): Input quantum circuit.
        
    Returns:
        QuantumCircuit: Circuit with measurements added.
    """
    try:
        qc_copy = qc.copy()
        if qc.num_clbits == 0:
            qc_copy.measure_all()
        return qc_copy
    except Exception as e:
        raise RuntimeError(f"Error adding measurements: {str(e)}")


def load_routing_pressure_circuit() -> QuantumCircuit:
    """
    Load a 5-qubit circuit designed to force SWAP gates on linear topology.
    
    This circuit has CX gates between distant qubits that cannot be executed
    without SWAP operations on a linear coupling map [0-1-2-3-4].
    
    Returns:
        QuantumCircuit: 5-qubit circuit with high routing pressure.
    """
    try:
        qc = QuantumCircuit(5, 5, name='RoutingPressure')
        qc.h(0)
        qc.cx(0, 4)   # Far apart: 0 to 4 (distance 4) — forces SWAPs
        qc.cx(1, 3)   # Far apart: 1 to 3 (distance 2) — may force SWAPs
        qc.cx(0, 3)   # Distance 3 — forces SWAPs
        qc.cx(2, 4)   # Distance 2 — may force SWAPs
        qc.cx(1, 4)   # Distance 3 — forces SWAPs
        qc.measure(range(5), range(5))
        return qc
    except Exception as e:
        raise RuntimeError(f"Error creating routing pressure circuit: {str(e)}")


if __name__ == "__main__":
    # Test circuit loading
    print("Testing circuit loader...")
    
    bell = load_bell_circuit()
    print(f"Bell circuit: {bell.num_qubits} qubits, depth: {bell.depth()}")
    
    ghz = load_ghz_circuit(3)
    print(f"GHZ-3 circuit: {ghz.num_qubits} qubits, depth: {ghz.depth()}")
    
    qft = load_qft_circuit(3)
    print(f"QFT-3 circuit: {qft.num_qubits} qubits, depth: {qft.depth()}")
    
    random = load_random_circuit(5, 3)
    print(f"Random circuit: {random.num_qubits} qubits, depth: {random.depth()}")
    
    print("All tests passed!")
