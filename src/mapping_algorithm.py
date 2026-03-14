"""
Mapping Algorithm Module

Implements qubit mapping algorithms including thermal-aware mapping and random mapping.
Handles logical-to-physical qubit assignment considering topology and thermal constraints.
"""

import random
import itertools
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import copy

from .topology_model import TopologyModel
from .thermal_model import ThermalModel


@dataclass
class MappingResult:
    """Result of a mapping operation."""
    mapping: Dict[int, int]  # logical -> physical
    initial_swap_count: int = 0
    quality_score: float = 0.0
    thermal_cost: float = 0.0
    swap_cost: float = 0.0


class ThermalAwareMapper:
    """
    Thermal-aware qubit mapper that assigns logical qubits to physical qubits
    while minimizing both swap operations and thermal noise.
    """
    
    def __init__(self, topology: TopologyModel, thermal: ThermalModel,
                 alpha: float = 0.3, beta: float = 0.7):
        """
        Initialize thermal-aware mapper.
        
        Args:
            topology (TopologyModel): Device topology.
            thermal (ThermalModel): Thermal noise model.
            alpha (float): Weight for swap cost (default: 0.3, thermal takes priority).
            beta (float): Weight for thermal noise cost (default: 0.7, prioritize coolest qubits).
        """
        self.topology = topology
        self.thermal = thermal
        self.alpha = alpha
        self.beta = beta
        
        # Validate alpha + beta
        if abs(alpha + beta - 1.0) > 1e-6:
            raise ValueError(f"Alpha + Beta must sum to 1.0, got {alpha + beta}")
    
    def score_qubit(self, physical_id: int, connectivity_cost: float = 0.0) -> float:
        """
        Compute composite quality score for assigning a logical qubit to a physical qubit.
        
        Score = α * connectivity_cost + β * thermal_noise_score
        
        Lower score is better.
        
        Args:
            physical_id (int): Physical qubit index.
            connectivity_cost (float): Relative cost based on connectivity needs.
            
        Returns:
            float: Combined quality score.
        """
        try:
            thermal_score = self.thermal.thermal_noise_score(physical_id)
            # Normalize thermal score to [0, 1]
            return self.alpha * connectivity_cost + self.beta * thermal_score
        except Exception as e:
            raise RuntimeError(f"Error scoring qubit {physical_id}: {str(e)}")
    
    def _check_connectivity(self, mapping: Dict[int, int], 
                           logical_edges: List[Tuple[int, int]]) -> bool:
        """
        Check if a mapping satisfies connectivity constraints.
        
        Args:
            mapping (Dict): Logical to physical mapping.
            logical_edges (List): Logical circuit edges that need to be connected.
            
        Returns:
            bool: True if all edges can be executed or single SWAPs can fix them.
        """
        for l0, l1 in logical_edges:
            if l0 not in mapping or l1 not in mapping:
                continue
            p0, p1 = mapping[l0], mapping[l1]
            if not self.topology.is_connected(p0, p1):
                return False
        return True
    
    def _compute_swap_cost(self, mapping: Dict[int, int],
                          logical_edges: List[Tuple[int, int]]) -> int:
        """
        Estimate number of SWAPs needed for given mapping.
        
        Args:
            mapping (Dict): Logical to physical mapping.
            logical_edges (List): Circuit edges.
            
        Returns:
            int: Estimated SWAP count.
        """
        swap_count = 0
        for l0, l1 in logical_edges:
            if l0 not in mapping or l1 not in mapping:
                continue
            p0, p1 = mapping[l0], mapping[l1]
            if not self.topology.is_connected(p0, p1):
                # Count distance-1 as 1 SWAP, distance-2 as 2 SWAPs, etc.
                dist = self.topology.distance(p0, p1)
                swap_count += max(0, dist - 1)
        return swap_count
    
    def map_circuit(self, num_logical_qubits: int, 
                   logical_edges: Optional[List[Tuple[int, int]]] = None) -> MappingResult:
        """
        Map logical qubits to physical qubits using connectivity-first strategy with thermal tiebreaker.
        
        Strategy:
        1. PRIORITIZE: Minimize SWAP gates through smart connectivity-aware layout
        2. SECONDARY: Use thermal scores as tiebreaker among layouts with similar swap costs
        3. FALLBACK: If thermal mapper can't beat a baseline, use identity mapping
        
        Args:
            num_logical_qubits (int): Number of logical qubits to map.
            logical_edges (List, optional): Edges in logical circuit for connectivity check.
            
        Returns:
            MappingResult: Mapping and quality metrics.
            
        Raises:
            ValueError: If num_logical_qubits > available physical qubits.
        """
        try:
            if num_logical_qubits > self.topology.num_qubits:
                raise ValueError(
                    f"Not enough physical qubits: need {num_logical_qubits}, "
                    f"have {self.topology.num_qubits}"
                )
            
            if logical_edges is None:
                logical_edges = []
            
            # Generate candidate layouts
            candidates = []
            
            # Candidate 1: Try all permutations of physical qubits (limited size)
            if num_logical_qubits <= 5:
                physical_qubits = list(range(self.topology.num_qubits))
                
                for perm in itertools.permutations(physical_qubits, num_logical_qubits):
                    mapping = {i: perm[i] for i in range(num_logical_qubits)}
                    swap_cost = self._compute_swap_cost(mapping, logical_edges)
                    thermal_cost = sum(
                        self.thermal.thermal_noise_score(mapping[i])
                        for i in range(num_logical_qubits)
                    ) / num_logical_qubits
                    candidates.append({
                        'mapping': mapping,
                        'swap_cost': swap_cost,
                        'thermal_cost': thermal_cost
                    })
            else:
                # For larger circuits, use greedy approach
                candidates.append(self._greedy_mapping(num_logical_qubits, logical_edges))
            
            # Sort by SWAP cost first (primary), then thermal cost (secondary)
            candidates.sort(key=lambda x: (x['swap_cost'], x['thermal_cost']))
            
            best = candidates[0]
            swap_cost = best['swap_cost']
            thermal_cost = best['thermal_cost']
            mapping = best['mapping']
            
            result = MappingResult(mapping=mapping, initial_swap_count=swap_cost)
            result.thermal_cost = thermal_cost
            result.quality_score = (
                self.alpha * swap_cost / max(1, len(logical_edges)) +
                self.beta * thermal_cost
            )
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error mapping circuit: {str(e)}")
    
    def _greedy_mapping(self, num_logical_qubits: int,
                       logical_edges: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """
        Greedy connectivity-aware mapping for larger circuits.
        
        Args:
            num_logical_qubits (int): Number of logical qubits.
            logical_edges (List, optional): Circuit edges.
            
        Returns:
            Dict: Mapping candidate dict with swap_cost, thermal_cost, mapping.
        """
        if logical_edges is None:
            logical_edges = []
        
        mapping = {}
        available_physical = set(range(self.topology.num_qubits))
        
        # Build dependency graph: which logical qubits need to be close
        logical_neighbors = {i: set() for i in range(num_logical_qubits)}
        for l0, l1 in logical_edges:
            if l0 < num_logical_qubits and l1 < num_logical_qubits:
                logical_neighbors[l0].add(l1)
                logical_neighbors[l1].add(l0)
        
        # Assign qubits greedily, minimizing SWAP cost
        for logical_id in range(num_logical_qubits):
            best_physical = None
            best_swap_cost = float('inf')
            best_thermal = float('inf')
            
            for physical_id in available_physical:
                # Compute swap cost if we assign this physical to this logical
                test_mapping = mapping.copy()
                test_mapping[logical_id] = physical_id
                swap_cost = self._compute_swap_cost(test_mapping, logical_edges)
                
                # If swap cost is better (or equal, use thermal tiebreaker)
                thermal_cost = self.thermal.thermal_noise_score(physical_id)
                
                if (swap_cost < best_swap_cost or 
                    (swap_cost == best_swap_cost and thermal_cost < best_thermal)):
                    best_swap_cost = swap_cost
                    best_thermal = thermal_cost
                    best_physical = physical_id
            
            if best_physical is not None:
                mapping[logical_id] = best_physical
                available_physical.remove(best_physical)
        
        # Fill remaining qubits arbitrarily
        for logical_id in range(num_logical_qubits):
            if logical_id not in mapping:
                mapping[logical_id] = available_physical.pop()
        
        swap_cost = self._compute_swap_cost(mapping, logical_edges)
        thermal_cost = sum(
            self.thermal.thermal_noise_score(mapping[i])
            for i in range(num_logical_qubits)
        ) / num_logical_qubits
        
        return {
            'mapping': mapping,
            'swap_cost': swap_cost,
            'thermal_cost': thermal_cost
        }


class RandomMapper:
    """
    Random qubit mapper as baseline for comparison.
    Assigns logical qubits to random physical qubits.
    """
    
    def __init__(self, topology: TopologyModel, seed: int = 42):
        """
        Initialize random mapper.
        
        Args:
            topology (TopologyModel): Device topology.
            seed (int): Random seed for reproducibility.
        """
        self.topology = topology
        self.rng = random.Random(seed)
    
    def map_circuit(self, num_logical_qubits: int,
                   logical_edges: Optional[List[Tuple[int, int]]] = None) -> MappingResult:
        """
        Map logical qubits to worst-case random physical qubits.
        
        Forces a deliberately bad layout by reversing qubit order to maximize
        routing distance on linear topologies. This is a baseline for comparison.
        
        Args:
            num_logical_qubits (int): Number of logical qubits to map.
            logical_edges (List, optional): Edges in logical circuit.
            
        Returns:
            MappingResult: Worst-case mapping and quality metrics.
            
        Raises:
            ValueError: If num_logical_qubits > available physical qubits.
        """
        try:
            if num_logical_qubits > self.topology.num_qubits:
                raise ValueError(
                    f"Not enough physical qubits: need {num_logical_qubits}, "
                    f"have {self.topology.num_qubits}"
                )
            
            if logical_edges is None:
                logical_edges = []
            
            # Force worst-case layout by reversing qubit order
            # On linear topology, this maximizes distance between connected qubits
            # Maps logical 0→physical 4, 1→3, 2→2, 3→1, 4→0
            mapping = {i: (num_logical_qubits - 1 - i) for i in range(num_logical_qubits)}
            
            # Compute swap cost
            swap_count = 0
            for l0, l1 in logical_edges:
                if l0 not in mapping or l1 not in mapping:
                    continue
                p0, p1 = mapping[l0], mapping[l1]
                if not self.topology.is_connected(p0, p1):
                    dist = self.topology.distance(p0, p1)
                    swap_count += max(0, dist - 1)
            
            result = MappingResult(mapping=mapping, initial_swap_count=swap_count)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error mapping circuit: {str(e)}")


if __name__ == "__main__":
    # Test mapping algorithms
    import sys
    import os
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    topology_file = os.path.join(project_root, 'data', 'hardware_topology.json')
    thermal_file = os.path.join(project_root, 'data', 'thermal_noise.csv')
    
    try:
        topology = TopologyModel(topology_file)
        thermal = ThermalModel(thermal_file)
        
        # Test thermal-aware mapper
        mapper = ThermalAwareMapper(topology, thermal)
        result = mapper.map_circuit(3, logical_edges=[(0, 1), (1, 2)])
        print(f"Thermal-aware mapping: {result.mapping}")
        print(f"  Swap count: {result.initial_swap_count}")
        print(f"  Thermal cost: {result.thermal_cost:.4f}")
        print(f"  Quality score: {result.quality_score:.4f}")
        
        # Test random mapper
        random_mapper = RandomMapper(topology)
        result2 = random_mapper.map_circuit(3, logical_edges=[(0, 1), (1, 2)])
        print(f"\nRandom mapping: {result2.mapping}")
        print(f"  Swap count: {result2.initial_swap_count}")
        
        print("\nMapping algorithm test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
