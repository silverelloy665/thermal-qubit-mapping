"""
Topology Model Module

Handles hardware topology of quantum devices including graph construction,
connectivity analysis, and visualization.
"""

import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import List, Dict, Tuple, Optional
import os


class TopologyModel:
    """
    Represents the hardware topology of a quantum device as a graph.
    
    Manages qubit connectivity, distance calculations, and provides methods
    for mapping logical qubits to physical qubits based on topology constraints.
    """
    
    def __init__(self, topology_file: str):
        """
        Initialize the topology model from a JSON file.
        
        Args:
            topology_file (str): Path to JSON file containing device topology.
            
        Raises:
            FileNotFoundError: If topology file not found.
            json.JSONDecodeError: If JSON file is invalid.
        """
        try:
            if not os.path.exists(topology_file):
                raise FileNotFoundError(f"Topology file not found: {topology_file}")
            
            with open(topology_file, 'r') as f:
                data = json.load(f)
            
            self.name = data.get('name', 'unknown_device')
            self.num_qubits = data.get('num_qubits', 0)
            self.edges = data.get('edges', [])
            
            # Build networkx graph
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(self.num_qubits))
            self.graph.add_edges_from(self.edges)
            
            # Precompute all shortest paths
            self._compute_shortest_paths()
            
        except Exception as e:
            raise RuntimeError(f"Error initializing topology model: {str(e)}")
    
    def _compute_shortest_paths(self) -> None:
        """Precompute all pairs shortest path distances."""
        try:
            self.distances = dict(nx.all_pairs_shortest_path_length(self.graph))
        except Exception as e:
            raise RuntimeError(f"Error computing shortest paths: {str(e)}")
    
    def get_neighbors(self, qubit: int) -> List[int]:
        """
        Get immediate neighbors of a qubit in the coupling map.
        
        Args:
            qubit (int): Qubit index.
            
        Returns:
            List[int]: List of neighboring qubit indices.
            
        Raises:
            ValueError: If qubit index is out of range.
        """
        try:
            if qubit < 0 or qubit >= self.num_qubits:
                raise ValueError(f"Qubit {qubit} out of range [0, {self.num_qubits-1}]")
            return list(self.graph.neighbors(qubit))
        except Exception as e:
            raise RuntimeError(f"Error getting neighbors for qubit {qubit}: {str(e)}")
    
    def shortest_path(self, q1: int, q2: int) -> List[int]:
        """
        Find the shortest path between two qubits.
        
        Args:
            q1 (int): Source qubit index.
            q2 (int): Target qubit index.
            
        Returns:
            List[int]: List of qubit indices forming the shortest path.
            
        Raises:
            ValueError: If qubits out of range or not connected.
        """
        try:
            if q1 < 0 or q1 >= self.num_qubits or q2 < 0 or q2 >= self.num_qubits:
                raise ValueError(f"Qubit indices out of range")
            
            path = nx.shortest_path(self.graph, q1, q2)
            return path
        except nx.NetworkXNoPath:
            raise ValueError(f"No path between qubits {q1} and {q2}")
        except Exception as e:
            raise RuntimeError(f"Error finding shortest path: {str(e)}")
    
    def distance(self, q1: int, q2: int) -> int:
        """
        Get the graph distance between two qubits.
        
        Args:
            q1 (int): Source qubit index.
            q2 (int): Target qubit index.
            
        Returns:
            int: Distance (number of edges) between qubits.
        """
        try:
            return self.distances[q1][q2]
        except Exception as e:
            raise RuntimeError(f"Error computing distance: {str(e)}")
    
    def get_coupling_map(self) -> List[Tuple[int, int]]:
        """
        Get the coupling map (list of connected qubit pairs).
        
        Returns:
            List[Tuple[int, int]]: List of (source, target) pairs representing device edges.
        """
        return list(self.graph.edges())
    
    def is_connected(self, q1: int, q2: int) -> bool:
        """
        Check if two qubits are directly connected.
        
        Args:
            q1 (int): First qubit index.
            q2 (int): Second qubit index.
            
        Returns:
            bool: True if qubits are directly connected, False otherwise.
        """
        return self.graph.has_edge(q1, q2)
    
    def visualize(self, output_path: Optional[str] = None, 
                 node_colors: Optional[List[float]] = None,
                 colormap: str = 'coolwarm') -> None:
        """
        Visualize the qubit topology graph.
        
        Args:
            output_path (str, optional): Path to save the figure. If None, shows plot.
            node_colors (List[float], optional): Colors for nodes (e.g., temperature values).
            colormap (str): Matplotlib colormap name.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Layout
            pos = nx.spring_layout(self.graph, seed=42, k=2)
            
            # Default colors: by qubit index
            if node_colors is None:
                node_colors = list(range(self.num_qubits))
            
            # Draw
            nodes = nx.draw_networkx_nodes(
                self.graph, pos,
                node_color=node_colors,
                node_size=1500,
                cmap=colormap,
                ax=ax
            )
            
            nx.draw_networkx_edges(self.graph, pos, width=2, ax=ax)
            nx.draw_networkx_labels(self.graph, pos, font_size=12, font_weight='bold', ax=ax)
            
            # Colorbar
            if node_colors is not None:
                cbar = plt.colorbar(nodes, ax=ax, label='Value')
            
            ax.set_title(f"Quantum Device Topology: {self.name}\n({self.num_qubits} Qubits)", 
                        fontsize=14, fontweight='bold')
            ax.axis('off')
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            raise RuntimeError(f"Error visualizing topology: {str(e)}")
    
    def get_graph_stats(self) -> Dict:
        """
        Get statistical information about the topology graph.
        
        Returns:
            Dict: Statistics including density, diameter, clustering coefficient, etc.
        """
        try:
            stats = {
                'num_qubits': self.num_qubits,
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'diameter': nx.diameter(self.graph) if nx.is_connected(self.graph) else float('inf'),
                'is_connected': nx.is_connected(self.graph),
                'avg_clustering': nx.average_clustering(self.graph),
            }
            return stats
        except Exception as e:
            raise RuntimeError(f"Error computing graph stats: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of topology."""
        stats = self.get_graph_stats()
        return (f"TopologyModel({self.name})\n"
                f"  Qubits: {self.num_qubits}\n"
                f"  Edges: {stats['num_edges']}\n"
                f"  Density: {stats['density']:.3f}\n"
                f"  Diameter: {stats['diameter']}")


if __name__ == "__main__":
    # Test topology model
    import sys
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    topology_file = os.path.join(project_root, 'data', 'hardware_topology.json')
    
    print(f"Loading topology from: {topology_file}")
    
    try:
        topo = TopologyModel(topology_file)
        print(topo)
        
        # Test methods
        neighbors = topo.get_neighbors(0)
        print(f"\nNeighbors of qubit 0: {neighbors}")
        
        path = topo.shortest_path(0, 3)
        print(f"Shortest path from 0 to 3: {path}")
        
        dist = topo.distance(0, 3)
        print(f"Distance from 0 to 3: {dist}")
        
        coupling_map = topo.get_coupling_map()
        print(f"Coupling map: {coupling_map}")
        
        print("\nTopology model test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
