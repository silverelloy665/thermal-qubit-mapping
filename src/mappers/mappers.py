import itertools
import random
import time
import math
import numpy as np
from typing import List, Dict, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target
import rustworkx as rx

from src.metrics.esp import compute_esp

def get_connected_subgraphs(target: Target, k: int) -> List[List[int]]:
    num_qubits = target.num_qubits
    if k > num_qubits:
        return []
    
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

def transpile_with_routing_seeds(circuit: QuantumCircuit, target: Target, layout: list, routing_seeds: int = 3, use_thermal: bool = False, temps_mk: dict = None) -> Tuple[QuantumCircuit, float]:
    best_tc = None
    best_esp = -1.0
    
    for r_seed in range(routing_seeds):
        tc = transpile(
            circuit,
            target=target,
            initial_layout=layout,
            routing_method='sabre',
            optimization_level=1,
            seed_transpiler=r_seed
        )
        esp_std, esp_th = compute_esp(tc, target, temps_mk)
        score = esp_th if use_thermal else esp_std
        if score > best_esp:
            best_esp = score
            best_tc = tc
            
    return best_tc, best_esp

def mapper_random(circuit: QuantumCircuit, target: Target, num_draws: int = 50, seed: int = 42, routing_seeds: int = 3) -> List[QuantumCircuit]:
    random.seed(seed)
    subgraphs = get_connected_subgraphs(target, circuit.num_qubits)
    if not subgraphs:
        subgraphs = [random.sample(range(target.num_qubits), circuit.num_qubits) for _ in range(num_draws)]
    
    circuits = []
    for i in range(num_draws):
        sg = random.choice(subgraphs)
        perm = list(sg)
        random.shuffle(perm)
        tc, _ = transpile_with_routing_seeds(circuit, target, perm, routing_seeds=routing_seeds)
        circuits.append(tc)
    return circuits

def mapper_qiskit_default(circuit: QuantumCircuit, target: Target, level: int, seed: int = 42) -> QuantumCircuit:
    return transpile(
        circuit,
        target=target,
        optimization_level=level,
        seed_transpiler=seed
    )

def mapper_esp(circuit: QuantumCircuit, target: Target, temps_mk: dict = None, 
               use_thermal: bool = False, exhaustive: bool = False, seed: int = 42, routing_seeds: int = 3, max_candidate_layouts: int = 200) -> Tuple[QuantumCircuit, float, float]:
    start_time = time.time()
    subgraphs = get_connected_subgraphs(target, circuit.num_qubits)
    best_tc = None
    best_esp = -1.0
    
    search_layouts = []
    if exhaustive and len(subgraphs) * math.factorial(circuit.num_qubits) < 5000:
        for sg in subgraphs:
            for p in itertools.permutations(sg):
                search_layouts.append(list(p))
    else:
        random.seed(seed)
        for _ in range(max_candidate_layouts):
            sg = random.choice(subgraphs)
            perm = list(sg)
            random.shuffle(perm)
            search_layouts.append(perm)
            
    unique_layouts = []
    seen = set()
    for lay in search_layouts:
        t = tuple(lay)
        if t not in seen:
            seen.add(t)
            unique_layouts.append(lay)
            
    for layout in unique_layouts:
        tc, score = transpile_with_routing_seeds(circuit, target, layout, routing_seeds, use_thermal, temps_mk)
        if score > best_esp:
            best_esp = score
            best_tc = tc
            
    if best_tc is None:
        best_tc = transpile(circuit, target=target, optimization_level=1, seed_transpiler=seed)
        best_esp = compute_esp(best_tc, target, temps_mk)[1 if use_thermal else 0]
        
    elapsed = time.time() - start_time
    return best_tc, best_esp, elapsed
