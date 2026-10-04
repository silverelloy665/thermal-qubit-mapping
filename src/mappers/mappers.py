import itertools
import random
import time
import math
import numpy as np
from typing import List, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target
import rustworkx as rx
from src.metrics.esp import esp_standard, esp_thermal, get_p1, load_config

def get_active_qubits(qc):
    active = set()
    for inst in qc.data:
        if inst.operation.name not in ['barrier', 'delay', 'measure']:
            for q in inst.qubits:
                active.add(qc.find_bit(q).index)
    return active

def get_connected_subgraphs(target: Target, k: int) -> List[List[int]]:
    num_qubits = target.num_qubits
    if k > num_qubits: return []
    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_qubits))
    edges = []
    for (q1, q2) in target.build_coupling_map().get_edges():
        edges.append((q1, q2))
    graph.add_edges_from_no_data(edges)
    
    subgraphs = set()
    def dfs(current_subgraph, neighbors):
        if len(current_subgraph) == k:
            subgraphs.add(tuple(sorted(current_subgraph)))
            return
        for node in neighbors:
            new_subgraph = current_subgraph | {node}
            new_neighbors = (neighbors | set(graph.neighbors(node))) - new_subgraph
            dfs(new_subgraph, new_neighbors)
    for i in range(num_qubits):
        dfs({i}, set(graph.neighbors(i)))
    return [list(sg) for sg in subgraphs]

def rank_subgraphs(subgraphs: List[List[int]], target: Target, use_thermal: bool, temps_mk: dict = None, p1: dict = None) -> List[List[int]]:
    scored = []
    for sg in subgraphs:
        score = 1.0
        for q in sg:
            try:
                err = target['measure'].get((q,), None)
                if err and getattr(err, 'error', None): score *= max(0.0, 1.0 - err.error)
            except KeyError: pass
        if use_thermal:
            for q in sg:
                p = get_p1(q, target, p1, temps_mk)
                if p > 0: score *= ((1.0 - p) ** 2)
        for i in range(len(sg)):
            for j in range(i+1, len(sg)):
                q1, q2 = sg[i], sg[j]
                for inst_name in ['cx', 'ecr', 'cz']:
                    try:
                        err1 = getattr(target[inst_name].get((q1, q2), None), 'error', 0.0) or 0.0
                        err2 = getattr(target[inst_name].get((q2, q1), None), 'error', 0.0) or 0.0
                        avg = (err1 + err2) / 2.0
                        if avg > 0:
                            score *= max(0.0, 1.0 - avg)
                            break
                    except KeyError: pass
        scored.append((score, sg))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:50]]

def _get_layout(tc: QuantumCircuit, original_circuit: QuantumCircuit) -> list:
    if getattr(tc, 'layout', None) and getattr(tc.layout, 'initial_layout', None):
        try:
            return [tc.layout.initial_layout[q] for q in original_circuit.qubits]
        except KeyError:
            pass
    raise ValueError("Circuit layout could not be determined. The transpiler did not attach an initial_layout.")

def mapper_random(circuit: QuantumCircuit, target: Target, num_draws: int = 50, seed: int = 42, routing_seeds: int = 3, opt_level: int = 3) -> List[Tuple[QuantumCircuit, float, float, list, int]]:
    random.seed(seed)
    subgraphs = get_connected_subgraphs(target, circuit.num_qubits)
    if not subgraphs: subgraphs = [list(range(circuit.num_qubits))]
    
    results = []
    for i in range(num_draws):
        start = time.time()
        sg = random.choice(subgraphs)
        perm = list(sg)
        random.shuffle(perm)
        
        best_tc = None
        best_esp = -1.0
        best_r_seed = 0
        for r_seed in range(routing_seeds):
            tc = transpile(circuit, target=target, initial_layout=perm, routing_method='sabre', optimization_level=opt_level, seed_transpiler=r_seed)
            esp = esp_standard(tc, target)
            if esp > best_esp:
                best_esp = esp; best_tc = tc; best_r_seed = r_seed
        
        lay = _get_layout(best_tc, circuit)
        active_qubits = set(get_active_qubits(best_tc))
        if active_qubits:
            assert active_qubits.issubset(set(lay)), f"Active qubits {active_qubits} exceed chosen layout {lay}"
        results.append((best_tc, best_esp, time.time() - start, lay, best_r_seed))
    return results

def mapper_qiskit_default(circuit: QuantumCircuit, target: Target, level: int, seed: int = 42) -> Tuple[QuantumCircuit, float, float, list, int]:
    start = time.time()
    tc = transpile(circuit, target=target, optimization_level=level, seed_transpiler=seed)
    esp = esp_standard(tc, target)
    lay = _get_layout(tc, circuit)
    return tc, esp, time.time() - start, lay, seed

def mapper_esp(circuit: QuantumCircuit, target: Target, temps_mk: dict = None, p1: dict = None,
               use_thermal: bool = False, exhaustive: bool = False, seed: int = 42, routing_seeds: int = 3, max_candidate_layouts: int = 200, opt_level: int = 3) -> Tuple[QuantumCircuit, float, float, list, int]:
    """
    Heuristic search mapper optimizing for ESP.
    Randomly samples up to `max_candidate_layouts` connected subgraphs (or exhaustive if specified)
    and evaluates each layout to pick the highest ESP configuration.
    """
    start_time = time.time()
    subgraphs = get_connected_subgraphs(target, circuit.num_qubits)
    subgraphs = rank_subgraphs(subgraphs, target, use_thermal, temps_mk, p1)
    
    search_layouts = []
    if exhaustive and len(subgraphs) * math.factorial(circuit.num_qubits) <= max_candidate_layouts:
        for sg in subgraphs:
            for perm in itertools.permutations(sg):
                search_layouts.append(list(perm))
    else:
        random.seed(seed)
        seen = set()
        for _ in range(max_candidate_layouts * 10):
            if len(search_layouts) >= max_candidate_layouts: break
            sg = random.choice(subgraphs)
            perm = list(sg)
            random.shuffle(perm)
            if tuple(perm) not in seen:
                seen.add(tuple(perm))
                search_layouts.append(perm)
                
    best_tc = None
    best_esp = -1.0
    best_r_seed = 0
    
    eval_fn = lambda c: esp_thermal(c, target, p1, mode=None) if use_thermal else esp_standard(c, target)
    
    for layout in search_layouts:
        for r_seed in range(routing_seeds):
            tc = transpile(circuit, target=target, initial_layout=layout, routing_method='sabre', optimization_level=opt_level, seed_transpiler=r_seed)
            score = eval_fn(tc)
            if score > best_esp:
                best_esp = score; best_tc = tc; best_r_seed = r_seed
                
    if best_tc is None:
        best_tc = transpile(circuit, target=target, optimization_level=opt_level, seed_transpiler=seed)
        best_esp = eval_fn(best_tc)
        
    lay = _get_layout(best_tc, circuit)
    active_qubits = set(get_active_qubits(best_tc))
    if active_qubits:
        assert active_qubits.issubset(set(lay)), f"Active qubits {active_qubits} exceed chosen layout {lay}"
    return best_tc, best_esp, time.time() - start_time, lay, best_r_seed
