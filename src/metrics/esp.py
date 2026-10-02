import numpy as np
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

H_PLANCK = 6.62607015e-34 # J s
K_BOLTZMANN = 1.380649e-23 # J / K

def calculate_thermal_population(freq_hz: float, temp_mk: float) -> float:
    """Calculate excited state population p1 for a given frequency and temperature in mK."""
    if temp_mk <= 0:
        return 0.0
    temp_k = temp_mk * 1e-3
    energy = H_PLANCK * freq_hz
    kt = K_BOLTZMANN * temp_k
    exponent = energy / kt
    if exponent > 700: 
        return 0.0
    return 1.0 / (1.0 + np.exp(exponent))

def compute_esp(circuit: QuantumCircuit, target: Target, temps_mk: dict = None) -> tuple[float, float]:
    """
    Computes Estimated Success Probability (ESP) of a transpiled circuit.
    
    Args:
        circuit: A routed and transpiled QuantumCircuit (must use backend basis gates).
        target: The backend Target.
        temps_mk: Dictionary mapping physical qubit index to temperature in millikelvin.
    
    Returns:
        (esp_standard, esp_thermal): Tuple of ESP without and with the thermal penalty.
    """
    prob_success = 1.0
    
    # 1. Gate errors and active durations
    qubit_active_duration = {q: 0.0 for q in range(circuit.num_qubits)}
    
    for inst in circuit.data:
        op = inst.operation
        qargs = inst.qubits
        
        # Ignore barriers and logical directives
        if op.name in ['barrier', 'measure']:
            continue
            
        phys_qubits = tuple(circuit.find_bit(q).index for q in qargs)
        
        # Get gate properties from target
        try:
            props = target[op.name].get(phys_qubits, None)
            if props is None:
                continue
            
            error = getattr(props, 'error', 0.0)
            duration = getattr(props, 'duration', 0.0)
            
            if error is not None:
                prob_success *= max(0.0, 1.0 - error)
                
            if duration is not None:
                for q in phys_qubits:
                    qubit_active_duration[q] += duration
        except KeyError:
            pass # Gate not in target (e.g. ideal simulator)

    # 2. Readout errors
    for inst in circuit.data:
        op = inst.operation
        qargs = inst.qubits
        if op.name == 'measure':
            phys_q = circuit.find_bit(qargs[0]).index
            try:
                props = target['measure'].get((phys_q,), None)
                if props is not None and getattr(props, 'error', None) is not None:
                    prob_success *= max(0.0, 1.0 - props.error)
            except KeyError:
                pass

    # 3. Idle decoherence
    # We estimate circuit duration as max active time (rough approximation if not fully scheduled)
    total_duration = max(qubit_active_duration.values()) if qubit_active_duration else 0.0
    
    idle_prob = 1.0
    for q in range(circuit.num_qubits):
        try:
            t1 = getattr(target.qubit_properties[q], 't1', None)
            t2 = getattr(target.qubit_properties[q], 't2', None)
        except (KeyError, TypeError):
            t1 = t2 = None
            
        idle_time = max(0.0, total_duration - qubit_active_duration[q])
        
        if t1 is not None and t1 > 0:
            idle_prob *= np.exp(-idle_time / t1)
        if t2 is not None and t2 > 0:
            idle_prob *= np.exp(-idle_time / t2)
            
    prob_success *= idle_prob
    esp_standard = prob_success

    # 4. Thermal penalty (Affects initialization)
    esp_thermal = esp_standard
    if temps_mk is not None:
        thermal_prob = 1.0
        for q in range(circuit.num_qubits):
            if q in temps_mk:
                try:
                    freq = getattr(target.qubit_properties[q], 'frequency', None)
                    if freq is not None:
                        p1 = calculate_thermal_population(freq, temps_mk[q])
                        # Probability of starting in the correct |0> state is (1 - p1)
                        # We also suffer this penalty at readout (simplified model)
                        thermal_prob *= ((1.0 - p1) ** 2)
                except (KeyError, TypeError):
                    pass
        esp_thermal *= thermal_prob

    return esp_standard, esp_thermal
