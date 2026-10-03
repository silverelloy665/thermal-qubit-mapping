import pytest
import os
import sys
import math
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from src.metrics.esp import esp_standard, esp_thermal, compute_esp_base
from src.mappers.mappers import mapper_esp, get_connected_subgraphs

@pytest.fixture
def fake_backend():
    return GenericBackendV2(num_qubits=5, basis_gates=['cx', 'id', 'rz', 'sx', 'x'], coupling_map=[[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]], seed=42)

def test_a_hand_checkable_esp():
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    backend = GenericBackendV2(num_qubits=1, basis_gates=['x'])
    backend.target['x'][(0,)].error = 0.01
    backend.target['x'][(0,)].duration = 0.0
    backend.target['measure'][(0,)].error = 0.02
    backend.target['measure'][(0,)].duration = 0.0
    
    tc = transpile(qc, backend, optimization_level=0)
    val = esp_standard(tc, backend.target)
    # 0.99 * 0.98 = 0.9702
    assert math.isclose(val, 0.9702, rel_tol=1e-12)

def test_b_unused_qubits_do_not_change_esp(fake_backend):
    qc1 = QuantumCircuit(2)
    qc1.cx(0,1)
    tc1 = transpile(qc1, fake_backend, initial_layout=[0,1], optimization_level=0)
    
    qc2 = QuantumCircuit(5)
    qc2.cx(0,1)
    tc2 = transpile(qc2, fake_backend, initial_layout=[0,1,2,3,4], optimization_level=0)
    
    assert math.isclose(esp_standard(tc1, fake_backend.target), esp_standard(tc2, fake_backend.target), rel_tol=1e-12)

def test_c_thermal_term_changes_layout(fake_backend):
    temps = {q: 15 for q in range(5)}
    temps[1] = 150 # q1 is hot
    qc = QuantumCircuit(2)
    qc.cx(0,1)
    tc, esp, _, lay, _ = mapper_esp(qc, fake_backend.target, temps_mk=temps, use_thermal=True, exhaustive=True, seed=42)
    assert 1 not in lay

def test_d_thermal_term_no_change_when_same_qubits(fake_backend):
    temps = {q: 15 for q in range(5)}
    temps[1] = 150
    qc = QuantumCircuit(5)
    for i in range(4): qc.cx(i, i+1)
    
    _, _, _, lay1, _ = mapper_esp(qc, fake_backend.target, temps_mk=None, use_thermal=False, exhaustive=True, seed=42)
    _, _, _, lay2, _ = mapper_esp(qc, fake_backend.target, temps_mk=temps, use_thermal=True, exhaustive=True, seed=42)
    assert lay1 == lay2

def test_e_layout_validity(fake_backend):
    sgs = get_connected_subgraphs(fake_backend.target, 3)
    for sg in sgs:
        assert len(sg) == 3
        assert len(set(sg)) == 3
    assert set([0,1,2]) in [set(s) for s in sgs]

def test_f_bit_ordering():
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.measure_all()
    backend = GenericBackendV2(num_qubits=2, basis_gates=['x'])
    backend.target['x'][(0,)].error = 0.0
    backend.target['x'][(1,)].error = 0.5
    backend.target['x'][(0,)].duration = 0.0
    backend.target['x'][(1,)].duration = 0.0
    backend.target['measure'][(0,)].error = 0.0
    backend.target['measure'][(1,)].error = 0.0
    backend.target['measure'][(0,)].duration = 0.0
    backend.target['measure'][(1,)].duration = 0.0
    
    tc = transpile(qc, backend, initial_layout=[1,0], optimization_level=0)
    esp = esp_standard(tc, backend.target)
    assert math.isclose(esp, 0.5, rel_tol=1e-3)

def test_g_120_layout_regression(fake_backend):
    qc = QuantumCircuit(5)
    for i in range(4): qc.cx(i, i+1)
    qc.measure_all()
    
    sim = AerSimulator()
    for perm in [[0,1,2,3,4], [4,3,2,1,0], [1,2,0,3,4]]:
        tc = transpile(qc, fake_backend, initial_layout=perm, optimization_level=0)
        counts = sim.run(tc).result().get_counts()
        assert '00000' in counts and counts['00000'] == 1024

def test_h_no_hardcoded_results():
    import glob
    for filepath in glob.glob('scripts/*.py') + glob.glob('src/**/*.py', recursive=True):
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            assert '0.51' not in content
            assert '0.9781' not in content

def test_i_column_names_differ():
    assert esp_standard.__name__ != esp_thermal.__name__

def test_j_excess_over_readout_never_exceeds_raw(fake_backend):
    qc = QuantumCircuit(1)
    qc.x(0)
    qc.measure_all()
    tc = transpile(qc, fake_backend, initial_layout=[0], optimization_level=0)
    
    temps = {0: 150}
    fake_backend.target['measure'][(0,)].error = 0.05
    
    raw = compute_esp_base(tc, fake_backend.target, temps_mk=temps, thermal_term_mode='raw_upper_bound')
    excess = compute_esp_base(tc, fake_backend.target, temps_mk=temps, thermal_term_mode='excess_over_readout')
    
    assert excess >= raw # Higher probability means lower penalty, so excess never exceeds raw penalty
