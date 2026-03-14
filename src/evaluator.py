"""
Evaluator Module

Evaluates quantum circuit optimization results through simulation,
fidelity computation, and comparative analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .thermal_model import ThermalModel
from .topology_model import TopologyModel


@dataclass
class SimulationResult:
    """Result from circuit simulation."""
    counts: Dict[str, int]
    total_shots: int
    method: str


class Evaluator:
    """
    Evaluates and compares quantum circuit optimization results.
    
    Performs simulations with noise models, computes fidelity metrics,
    and generates comparative reports.
    """
    
    def __init__(self, thermal_model: ThermalModel, 
                 topology_model: Optional[TopologyModel] = None):
        """
        Initialize evaluator.
        
        Args:
            thermal_model (ThermalModel): Thermal noise model for simulations.
            topology_model (TopologyModel, optional): Device topology.
        """
        self.thermal = thermal_model
        self.topology = topology_model
        self.simulator = AerSimulator()
    
    def run_simulation(self, circuit: QuantumCircuit, 
                      noise_model: Optional[NoiseModel] = None,
                      shots: int = 1024, method: str = "default") -> SimulationResult:
        """
        Execute circuit simulation with optional noise.
        
        Args:
            circuit (QuantumCircuit): Circuit to simulate.
            noise_model (NoiseModel, optional): Qiskit noise model.
            shots (int): Number of shots for simulation.
            method (str): Label for the simulation method.
            
        Returns:
            SimulationResult: Simulation counts and metadata.
            
        Raises:
            RuntimeError: If simulation fails.
        """
        try:
            # Run simulation
            if noise_model:
                job = self.simulator.run(circuit, noise_model=noise_model, shots=shots)
            else:
                job = self.simulator.run(circuit, shots=shots)
            
            result = job.result()
            counts = result.get_counts(circuit)
            
            return SimulationResult(
                counts=counts,
                total_shots=shots,
                method=method
            )
            
        except Exception as e:
            raise RuntimeError(f"Error during simulation: {str(e)}")
    
    def compute_fidelity(self, ideal_counts: Dict[str, int],
                        noisy_counts: Dict[str, int]) -> float:
        """
        Compute fidelity between ideal and noisy measurement distributions.
        
        Uses Hellinger fidelity: F = (Σ √(p_i * q_i))^2
        
        Args:
            ideal_counts (Dict): Ideal measurement counts.
            noisy_counts (Dict): Noisy measurement counts.
            
        Returns:
            float: Fidelity in range [0, 1].
        """
        try:
            # Normalize counts to probabilities
            def normalize(counts_dict):
                total = sum(counts_dict.values())
                return {state: count / total for state, count in counts_dict.items()}
            
            ideal_probs = normalize(ideal_counts)
            noisy_probs = normalize(noisy_counts)
            
            # Get all possible states
            all_states = set(ideal_probs.keys()) | set(noisy_probs.keys())
            
            # Compute Hellinger fidelity
            fidelity_sum = 0.0
            for state in all_states:
                p = ideal_probs.get(state, 0.0)
                q = noisy_probs.get(state, 0.0)
                fidelity_sum += np.sqrt(p * q)
            
            fidelity = fidelity_sum ** 2
            return float(max(0.0, min(1.0, fidelity)))  # Clamp to [0, 1]
            
        except Exception as e:
            raise RuntimeError(f"Error computing fidelity: {str(e)}")
    
    def compute_noise_score(self, mapping: Dict[int, int],
                           num_qubits: int) -> float:
        """
        Compute weighted average thermal noise score for a mapping.
        
        Args:
            mapping (Dict): Logical to physical qubit mapping.
            num_qubits (int): Number of logical qubits.
            
        Returns:
            float: Average thermal noise score.
        """
        try:
            if not mapping or num_qubits == 0:
                return 0.0
            
            scores = []
            for logical_id in range(num_qubits):
                physical_id = mapping.get(logical_id, logical_id)
                score = self.thermal.thermal_noise_score(physical_id)
                scores.append(score)
            
            return float(np.mean(scores)) if scores else 0.0
            
        except Exception as e:
            raise RuntimeError(f"Error computing noise score: {str(e)}")
    
    def evaluate_result(self, opt_result: OptimizationResult,
                       ideal_circuit: QuantumCircuit,
                       noise_model: Optional[NoiseModel] = None,
                       shots: int = 1024) -> Dict:
        """
        Evaluate a single optimization result.
        
        Args:
            opt_result (OptimizationResult): Optimization result to evaluate.
            ideal_circuit (QuantumCircuit): Ideal circuit for fidelity reference.
            noise_model (NoiseModel, optional): Noise model for evaluation.
            shots (int): Number of simulation shots.
            
        Returns:
            Dict: Evaluation metrics including fidelity, noise score, etc.
        """
        try:
            # Run simulations
            ideal_sim = self.run_simulation(ideal_circuit, noise_model=None, 
                                           shots=shots, method="ideal")
            noisy_sim = self.run_simulation(opt_result.circuit, noise_model=noise_model,
                                           shots=shots, method=opt_result.method)
            
            # Compute metrics
            fidelity = self.compute_fidelity(ideal_sim.counts, noisy_sim.counts)
            noise_score = self.compute_noise_score(
                opt_result.mapping or {},
                ideal_circuit.num_qubits
            )
            
            return {
                'method': opt_result.method,
                'swap_count': opt_result.swap_count,
                'depth': opt_result.depth,
                'fidelity': fidelity,
                'noise_score': noise_score,
                'ideal_counts': ideal_sim.counts,
                'noisy_counts': noisy_sim.counts,
            }
            
        except Exception as e:
            raise RuntimeError(f"Error evaluating result: {str(e)}")
    
    def compare_methods(self, opt_results: Dict[str, OptimizationResult],
                       ideal_circuit: QuantumCircuit,
                       noise_model: Optional[NoiseModel] = None,
                       shots: int = 1024) -> pd.DataFrame:
        """
        Compare optimization results from multiple methods.
        
        Args:
            opt_results (Dict): Optimization results keyed by method name.
            ideal_circuit (QuantumCircuit): Reference ideal circuit.
            noise_model (NoiseModel, optional): Noise model.
            shots (int): Simulation shots.
            
        Returns:
            pd.DataFrame: Comparison table with metrics for each method.
        """
        try:
            results = []
            
            for method, opt_result in opt_results.items():
                eval_result = self.evaluate_result(
                    opt_result,
                    ideal_circuit,
                    noise_model=noise_model,
                    shots=shots
                )
                results.append(eval_result)
            
            df = pd.DataFrame(results)
            # Sort by method for consistency
            df = df.sort_values('method').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            raise RuntimeError(f"Error comparing methods: {str(e)}")
    
    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """
        Generate formatted text report of comparison results.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe from compare_methods.
            
        Returns:
            str: Formatted report string.
        """
        try:
            report = "\n" + "="*80 + "\n"
            report += "OPTIMIZATION RESULT COMPARISON REPORT\n"
            report += "="*80 + "\n\n"
            
            # Summary table
            report += "Performance Metrics by Method:\n"
            report += "-"*80 + "\n"
            
            # Create display format
            summary_data = []
            for _, row in comparison_df.iterrows():
                summary_data.append({
                    'Method': row['method'].capitalize(),
                    'SWAP Count': int(row['swap_count']),
                    'Depth': int(row['depth']),
                    'Noise Score': f"{row['noise_score']:.4f}",
                    'Fidelity': f"{row['fidelity']:.4f}",
                })
            
            summary_df = pd.DataFrame(summary_data)
            report += summary_df.to_string(index=False)
            report += "\n\n"
            
            # Find best method for each metric
            report += "Best Performance by Metric:\n"
            report += "-"*80 + "\n"
            
            best_swaps_idx = comparison_df['swap_count'].idxmin()
            best_fidelity_idx = comparison_df['fidelity'].idxmax()
            best_noise_idx = comparison_df['noise_score'].idxmin()
            
            report += f"  Least SWAPs: {comparison_df.loc[best_swaps_idx, 'method'].capitalize()} "
            report += f"({int(comparison_df.loc[best_swaps_idx, 'swap_count'])} SWAPs)\n"
            report += f"  Highest Fidelity: {comparison_df.loc[best_fidelity_idx, 'method'].capitalize()} "
            report += f"({comparison_df.loc[best_fidelity_idx, 'fidelity']:.4f})\n"
            report += f"  Lowest Noise: {comparison_df.loc[best_noise_idx, 'method'].capitalize()} "
            report += f"({comparison_df.loc[best_noise_idx, 'noise_score']:.4f})\n"
            
            report += "\n" + "="*80 + "\n"
            
            return report
            
        except Exception as e:
            raise RuntimeError(f"Error generating report: {str(e)}")
    
    def print_comparison_table(self, comparison_df: pd.DataFrame) -> None:
        """
        Print formatted comparison table with Unicode box characters.
        
        Args:
            comparison_df (pd.DataFrame): Comparison dataframe.
        """
        try:
            rows = []
            for _, row in comparison_df.iterrows():
                rows.append({
                    'Method': row['method'].capitalize(),
                    'SWAP Count': int(row['swap_count']),
                    'Noise Score': float(row['noise_score']),
                    'Fidelity': float(row['fidelity']),
                })
            
            # Print table header
            print("\n+----------------------+-----------+-------------+--------------+")
            print("| Method               | SWAP Count| Noise Score | Fidelity     |")
            print("+----------------------+-----------+-------------+--------------+")
            
            # Print rows
            for row in rows:
                method = row['Method'].ljust(20)
                swaps = str(row['SWAP Count']).center(9)
                noise = f"{row['Noise Score']:.2f}".center(11)
                fidelity = f"{row['Fidelity']:.2f}".center(12)
                print(f"| {method} | {swaps} | {noise} | {fidelity} |")
            
            # Print table footer
            print("+----------------------+-----------+-------------+--------------+\n")
            
        except Exception as e:
            print(f"Error printing table: {e}")


if __name__ == "__main__":
    # Test evaluator (requires full setup)
    print("Evaluator module loaded successfully")
