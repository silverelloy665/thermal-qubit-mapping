import numpy as np
import yaml
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

H_PLANCK = 6.62607015e-34
K_BOLTZMANN = 1.380649e-23

def load_config():
    p = Path(__file__).parent.parent.parent / 'config.yaml'
    if not p.exists(): return {}
    with open(p, 'r') as f:
        return yaml.safe_load(f)

def calculate_thermal_population(freq_hz: float, temp_mk: float) -> float:
    if temp_mk <= 0: return 0.0
    temp_k = temp_mk / 1000.0
    exponent = (H_PLANCK * freq_hz) / (K_BOLTZMANN * temp_k)
    if exponent > 700: return 0.0
    return 1.0 / (1.0 + np.exp(exponent))

def get_p1(q: int, target: Target, p1: dict = None, temps_mk: dict = None) -> float:
    if p1 and q in p1: return p1[q]
    if temps_mk and q in temps_mk:
        freq = getattr(target.qubit_properties[q], 'frequency', None) if getattr(target, 'qubit_properties', None) else None
        if freq is not None:
            return calculate_thermal_population(freq, temps_mk[q])
    return 0.0

def get_active_qubits(circuit: QuantumCircuit) -> set:
    active = set()
    for inst in circuit.data:
        if inst.operation.name not in ['barrier', 'delay']:
            active.update(circuit.find_bit(q).index for q in inst.qubits)
    return active

def get_native_2q_gates(target: Target) -> list:
    native = []
    if not target: return ['cx']
    for inst_name, _ in target.instructions:
        if inst_name in ['cx', 'cz', 'ecr'] and inst_name not in native:
            native.append(inst_name)
    return native if native else ['cx']

def count_native_2q(circuit: QuantumCircuit, target: Target) -> int:
    native_names = get_native_2q_gates(target)
    count = 0
    for k, v in circuit.count_ops().items():
        if k in native_names:
            count += v
    return count

def compute_esp_base(circuit: QuantumCircuit, target: Target, p1: dict = None, temps_mk: dict = None, thermal_term_mode: str = 'none') -> float:
    prob_success = 1.0
    active_qubits = get_active_qubits(circuit)
    if not active_qubits: return 1.0
    
    qubit_active_duration = {q: 0.0 for q in range(target.num_qubits)} if target else {}
    readout_errors = {}
    
    if target:
        for inst in circuit.data:
            op = inst.operation
            qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
            if op.name in ['barrier', 'delay', 'measure']: continue
            try:
                props = target[op.name].get(qargs, None)
                if props:
                    if getattr(props, 'error', None): prob_success *= max(0.0, 1.0 - props.error)
                    if getattr(props, 'duration', None):
                        for q in qargs: qubit_active_duration[q] += props.duration
            except KeyError: pass
            
        for q in active_qubits:
            try:
                props = target['measure'].get((q,), None)
                if props and getattr(props, 'error', None):
                    readout_errors[q] = props.error
                    prob_success *= max(0.0, 1.0 - props.error)
            except KeyError: pass
            
        total_duration = max(qubit_active_duration.values()) if qubit_active_duration else 0.0
        
        idle_prob = 1.0
        for q in active_qubits:
            try:
                t1 = getattr(target.qubit_properties[q], 't1', None)
                t2 = getattr(target.qubit_properties[q], 't2', None)
            except (KeyError, AttributeError, TypeError):
                t1 = t2 = None
            idle_time = max(0.0, total_duration - qubit_active_duration.get(q, 0.0))
            if t1 and t1 > 0: idle_prob *= np.exp(-idle_time / t1)
            if t2 and t2 > 0: idle_prob *= np.exp(-idle_time / t2)
        prob_success *= idle_prob

    if thermal_term_mode == 'none': return prob_success
    
    if thermal_term_mode in ['raw_upper_bound', 'excess_over_readout']:
        thermal_prob = 1.0
        for q in active_qubits:
            p1_val = get_p1(q, target, p1, temps_mk)
            if p1_val > 0:
                if thermal_term_mode == 'excess_over_readout':
                    ro = readout_errors.get(q, 0.0)
                    penalty = max(0.0, p1_val - ro)
                    thermal_prob *= ((1.0 - penalty) ** 2)
                else:
                    thermal_prob *= ((1.0 - p1_val) ** 2)
        return prob_success * thermal_prob
        
    return prob_success

def esp_standard(circuit: QuantumCircuit, target: Target) -> float:
    return compute_esp_base(circuit, target, thermal_term_mode='none')

def esp_thermal(circuit: QuantumCircuit, target: Target, p1: dict = None, mode: str = None) -> float:
    if not mode:
        config = load_config()
        mode = config.get('experiment', {}).get('thermal_term_mode', 'raw_upper_bound')
    # Backward compatibility with temps_mk if passed instead of p1 dict (for older scripts)
    temps_mk = p1 if isinstance(p1, dict) and any(v > 1.0 for v in p1.values()) else None
    p1_dict = None if temps_mk else p1
    return compute_esp_base(circuit, target, p1=p1_dict, temps_mk=temps_mk, thermal_term_mode=mode)

def esp_thermal_gate(circuit: QuantumCircuit, target: Target, p1: dict = None) -> float:
    esp = esp_standard(circuit, target)
    if not target: return esp
    temps_mk = p1 if isinstance(p1, dict) and any(v > 1.0 for v in p1.values()) else None
    p1_dict = None if temps_mk else p1
    
    thermal_prob = 1.0
    for inst in circuit.data:
        op = inst.operation
        qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
        if op.name in ['barrier', 'delay', 'measure']: continue
        try:
            props = target[op.name].get(qargs, None)
            if props and getattr(props, 'duration', 0.0) > 0:
                for q in qargs:
                    p1_val = get_p1(q, target, p1_dict, temps_mk)
                    if p1_val > 0:
                        t1 = getattr(target.qubit_properties[q], 't1', 1e-3)
                        if not t1 or t1 <= 0: t1 = 1e-3
                        thermal_prob *= max(0.0, 1.0 - p1_val * (1.0 - np.exp(-props.duration / t1)))
        except (KeyError, AttributeError, TypeError): pass
    return esp * thermal_prob
