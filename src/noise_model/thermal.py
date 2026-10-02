from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
from qiskit.transpiler import Target
from src.metrics.esp import calculate_thermal_population

def build_thermal_noise_model(target: Target, temps_mk: dict = None) -> NoiseModel:
    noise_model = NoiseModel()
    num_qubits = target.num_qubits
    
    seen_1q = set()
    seen_2q = set()
    
    for q in range(num_qubits):
        q_props = target.qubit_properties[q]
        if q_props is None:
            continue
            
        t1 = getattr(q_props, 't1', None)
        t2 = getattr(q_props, 't2', None)
        freq = getattr(q_props, 'frequency', None)
        
        p1 = 0.0
        if temps_mk and q in temps_mk and freq is not None:
            p1 = calculate_thermal_population(freq, temps_mk[q])
            
        for inst, qargs in target.instructions:
            inst_name = inst.name
            if inst_name in ['delay', 'barrier', 'measure']:
                continue
                
            if len(qargs) == 1 and qargs[0] == q:
                key = (inst_name, q)
                if key in seen_1q:
                    continue
                seen_1q.add(key)
                
                props = target[inst_name].get((q,), None)
                if props is not None and getattr(props, 'duration', None) is not None:
                    dur = props.duration
                    if t1 and t2 and dur > 0:
                        t_err = thermal_relaxation_error(t1, t2, dur, p1)
                        noise_model.add_quantum_error(t_err, inst_name, [q])
                        
        for inst, qargs in target.instructions:
            inst_name = inst.name
            if inst_name in ['cx', 'ecr', 'cz'] and len(qargs) == 2 and qargs[0] == q:
                q1, q2 = qargs
                key = (inst_name, q1, q2)
                if key in seen_2q:
                    continue
                seen_2q.add(key)
                
                props = target[inst_name].get((q1, q2), None)
                if props is not None:
                    dur = getattr(props, 'duration', 0.0)
                    if t1 and t2 and dur > 0:
                        t_err1 = thermal_relaxation_error(t1, t2, dur, p1)
                        
                        t1_2 = getattr(target.qubit_properties[q2], 't1', t1)
                        t2_2 = getattr(target.qubit_properties[q2], 't2', t2)
                        f2 = getattr(target.qubit_properties[q2], 'frequency', freq)
                        p1_2 = calculate_thermal_population(f2, temps_mk.get(q2, 0.0)) if (temps_mk and f2) else 0.0
                        t_err2 = thermal_relaxation_error(t1_2, t2_2, dur, p1_2)
                        
                        t_err_2q = t_err1.tensor(t_err2)
                        noise_model.add_quantum_error(t_err_2q, inst_name, [q1, q2])

        try:
            measure_props = target['measure'].get((q,), None)
            if measure_props and getattr(measure_props, 'duration', None):
                dur = measure_props.duration
                if t1 and t2 and dur > 0:
                    t_err = thermal_relaxation_error(t1, t2, dur, p1)
                    noise_model.add_quantum_error(t_err, 'measure', [q])
        except KeyError:
            pass

    return noise_model
