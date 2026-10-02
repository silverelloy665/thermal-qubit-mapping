import itertools
import random
import numpy as np
from typing import List, Dict, Tuple
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import Target
import rustworkx as rx
from qiskit.transpiler.passes import VF2Layout

from src.metrics.esp import compute_esp

def get_connected_subgraphs(target: Target, k: int) -> List[List[int]]:
    """Find all connected subgraphs of size k in the backend target."""
    num_qubits = target.num_qubits
    if k > num_qubits:
        return []
    
    # Build graph
    graph = rx.PyGraph()
    graph.add_nodes_from(range(num_qubits))
    edges = []
    for (q1, q2) in target.build_coupling_map().get_edges():
        edges.append((q1, q2))
    graph.add_edges_from_no_data(edges)
    
    # Simple BFS/DFS to find connected subgraphs of size k
    # For small k (like 5), this is fast.
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

def transpile_with_layout(circuit: QuantumCircuit, target: Target, layout: list, seed: int = 42) -> QuantumCircuit:
    """Transpile a circuit with a specific initial layout."""
    return transpile(
        circuit,
        target=target,
        initial_layout=layout,
        routing_method='sabre',
        optimization_level=1,
        seed_transpiler=seed
    )

def mapper_random(circuit: QuantumCircuit, target: Target, num_draws: int = 50, seed: int = 42) -> List[QuantumCircuit]:
    """Generates random valid initial layouts (connected subgraphs)."""
    random.seed(seed)
    subgraphs = get_connected_subgraphs(target, circuit.num_qubits)
    if not subgraphs:
        # Fallback to completely random choices
        subgraphs = [random.sample(range(target.num_qubits), circuit.num_qubits) for _ in range(num_draws)]
    
    circuits = []
    for i in range(num_draws):
        sg = random.choice(subgraphs)
        perm = list(sg)
        random.shuffle(perm)
        tc = transpile_with_layout(circuit, target, perm, seed=seed+i)
        circuits.append(tc)
    return circuits

def mapper_qiskit_default(circuit: QuantumCircuit, target: Target, level: int, seed: int = 42) -> QuantumCircuit:
    """Qiskit default transpilation at specified optimization level."""
    return transpile(
        circuit,
        target=target,
        optimization_level=level,
        seed_transpiler=seed
    )

def mapper_esp(circuit: QuantumCircuit, target: Target, temps_mk: dict = None, 
               use_thermal: bool = False, exhaustive: bool = False, seed: int = 42) -> Tuple[QuantumCircuit, float]:
    """
    Search for the best layout using ESP.
    If exhaustive=True, search all permutations of all connected subgraphs.
    If exhaustive=False, search random permutations over subgraphs (simulated annealing or random shots).
    """
    subgraphs = get_connected_subgraphs(target, circuit.num_qubits)
    best_tc = None
    best_esp = -1.0
    
    # Determine the search space
    search_layouts = []
    if exhaustive:
        for sg in subgraphs:
            for p in itertools.permutations(sg):
                search_layouts.append(list(p))
    else:
        # Random search over subgraphs
        random.seed(seed)
        for _ in range(50):
            sg = random.choice(subgraphs)
            perm = list(sg)
            random.shuffle(perm)
            search_layouts.append(perm)
            
    # Remove duplicates
    unique_layouts = []
    seen = set()
    for lay in search_layouts:
        t = tuple(lay)
        if t not in seen:
            seen.add(t)
            unique_layouts.append(lay)
            
    for layout in unique_layouts:
        tc = transpile_with_layout(circuit, target, layout, seed=seed)
        esp_std, esp_th = compute_esp(tc, target, temps_mk)
        score = esp_th if use_thermal else esp_std
        
        if score > best_esp:
            best_esp = score
            best_tc = tc
            
    if best_tc is None: # Fallback if no subgraphs found
        best_tc = transpile(circuit, target=target, optimization_level=1, seed_transpiler=seed)
        best_esp = compute_esp(best_tc, target, temps_mk)[1 if use_thermal else 0]
        
    return best_tc, best_esp
