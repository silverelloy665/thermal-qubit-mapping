from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error, ReadoutError, pauli_error
from qiskit.transpiler import Target
from src.metrics.esp import get_p1

def build_thermal_noise_model(target: Target, temps_mk: dict = None, p1_dict: dict = None) -> NoiseModel:
    noise_model = NoiseModel()
    seen_1q = set()
    seen_2q = set()
    
    # Add thermal initialization via reset + bitflip
    for q in range(target.num_qubits):
        p1 = get_p1(q, target, p1=p1_dict, temps_mk=temps_mk)
        if p1 > 0:
            err = pauli_error([('X', p1), ('I', 1 - p1)])
            noise_model.add_quantum_error(err, "reset", [q])

    for q in range(target.num_qubits):
        q_props = target.qubit_properties[q] if getattr(target, 'qubit_properties', None) else None
        t1 = getattr(q_props, 't1', None) if q_props else None
        t2 = getattr(q_props, 't2', None) if q_props else None
        p1 = get_p1(q, target, p1=p1_dict, temps_mk=temps_mk)
            
        for inst, qargs in target.instructions:
            inst_name = inst.name
            if inst_name in ['delay', 'barrier', 'measure', 'reset']: continue
                
            if len(qargs) == 1 and qargs[0] == q:
                key = (inst_name, q)
                if key in seen_1q: continue
                seen_1q.add(key)
                props = target[inst_name].get((q,), None)
                if props is not None:
                    dur = getattr(props, 'duration', 0.0)
                    err_rate = getattr(props, 'error', 0.0)
                    if t1 and t2 and dur > 0:
                        t_err = thermal_relaxation_error(t1, t2, dur, p1)
                        if err_rate > 0: t_err = t_err.compose(depolarizing_error(err_rate, 1))
                        noise_model.add_quantum_error(t_err, inst_name, [q])
                        
        for inst, qargs in target.instructions:
            inst_name = inst.name
            if inst_name in ['cx', 'ecr', 'cz'] and len(qargs) == 2 and qargs[0] == q:
                q1, q2 = qargs
                key = (inst_name, q1, q2)
                if key in seen_2q: continue
                seen_2q.add(key)
                props = target[inst_name].get((q1, q2), None)
                if props is not None:
                    dur = getattr(props, 'duration', 0.0)
                    err_rate = getattr(props, 'error', 0.0)
                    if t1 and t2 and dur > 0:
                        t_err1 = thermal_relaxation_error(t1, t2, dur, p1)
                        q2_props = target.qubit_properties[q2] if getattr(target, 'qubit_properties', None) else None
                        t1_2 = getattr(q2_props, 't1', t1) if q2_props else t1
                        t2_2 = getattr(q2_props, 't2', t2) if q2_props else t2
                        p1_2 = get_p1(q2, target, p1=p1_dict, temps_mk=temps_mk)
                        t_err2 = thermal_relaxation_error(t1_2, t2_2, dur, p1_2)
                        t_err_2q = t_err1.tensor(t_err2)
                        if err_rate > 0: t_err_2q = t_err_2q.compose(depolarizing_error(err_rate, 2))
                        noise_model.add_quantum_error(t_err_2q, inst_name, [q1, q2])

        try:
            measure_props = target['measure'].get((q,), None)
            if measure_props:
                prob = getattr(measure_props, 'error', 0.0)
                if prob > 0:
                    ro_err = ReadoutError([[1 - prob, prob], [prob, 1 - prob]])
                    noise_model.add_readout_error(ro_err, [q])
        except KeyError: pass

    return noise_model

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.providers.fake_provider import GenericBackendV2
    backend = GenericBackendV2(num_qubits=2)
    nm = build_thermal_noise_model(backend.target, p1_dict={0: 0.12})
    qc = QuantumCircuit(2)
    qc.reset(0)
    qc.measure_all()
    tc = transpile(qc, backend, optimization_level=0)
    sim = AerSimulator(noise_model=nm)
    counts = sim.run(tc, shots=10000, seed_simulator=42).result().get_counts()
    print("Counts for p1=0.12:", counts)
