"""
Thermal Model Module

Manages thermal noise parameters and builds Qiskit NoiseModel objects
based on thermal characteristics of quantum devices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error
from typing import Dict, List, Tuple, Optional
import os


class ThermalModel:
    """
    Represents thermal noise characteristics of a quantum device.
    
    Loads thermal parameters (temperature, T1, T2, error rates) and provides
    methods to compute noise scores and build Qiskit NoiseModel for simulations.
    """
    
    def __init__(self, thermal_file: str):
        """
        Initialize thermal model from a CSV file.
        
        Expected CSV columns: qubit, temperature, error_rate, t1_us, t2_us, stability_score
        
        Args:
            thermal_file (str): Path to CSV file containing thermal parameters.
            
        Raises:
            FileNotFoundError: If thermal file not found.
            ValueError: If required columns missing.
        """
        try:
            if not os.path.exists(thermal_file):
                raise FileNotFoundError(f"Thermal file not found: {thermal_file}")
            
            self.data = pd.read_csv(thermal_file)
            
            # Validate required columns
            required_cols = ['qubit', 'temperature', 'error_rate', 't1_us', 't2_us', 'stability_score']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in thermal data: {missing_cols}")
            
            # Parse qubit names (remove 'q' prefix if present)
            self.data['qubit_id'] = self.data['qubit'].apply(
                lambda x: int(x.replace('q', '')) if isinstance(x, str) else int(x)
            )
            
            self.num_qubits = len(self.data)
            self._compute_noise_scores()
            
        except Exception as e:
            raise RuntimeError(f"Error initializing thermal model: {str(e)}")
    
    def _compute_noise_scores(self) -> None:
        """Compute normalized noise scores for each qubit."""
        try:
            # Normalize temperature (0-1)
            temp_min, temp_max = self.data['temperature'].min(), self.data['temperature'].max()
            temp_range = temp_max - temp_min if temp_max > temp_min else 1
            norm_temp = (self.data['temperature'] - temp_min) / temp_range
            
            # Normalize error rate (0-1)
            error_min, error_max = self.data['error_rate'].min(), self.data['error_rate'].max()
            error_range = error_max - error_min if error_max > error_min else 1
            norm_error = (self.data['error_rate'] - error_min) / error_range
            
            # Compute thermal noise score: weighted combination
            # Higher temperature and higher error rate = higher noise
            self.data['computed_noise_score'] = 0.5 * norm_temp + 0.5 * norm_error
            
        except Exception as e:
            raise RuntimeError(f"Error computing noise scores: {str(e)}")
    
    def thermal_noise_score(self, qubit_id: int) -> float:
        """
        Get the thermal noise score for a specific qubit.
        
        Score is computed from temperature and error rate. Higher score = higher noise.
        
        Args:
            qubit_id (int): Qubit index.
            
        Returns:
            float: Noise score in range [0, 1].
            
        Raises:
            ValueError: If qubit_id not found.
        """
        try:
            row = self.data[self.data['qubit_id'] == qubit_id]
            if row.empty:
                raise ValueError(f"Qubit {qubit_id} not found in thermal data")
            return float(row['computed_noise_score'].values[0])
        except Exception as e:
            raise RuntimeError(f"Error getting noise score for qubit {qubit_id}: {str(e)}")
    
    def stability_rank(self) -> List[int]:
        """
        Get qubits ranked by stability (best to worst).
        
        Ranking based on stability_score column (higher = better).
        
        Returns:
            List[int]: Qubit indices sorted from most to least stable.
        """
        try:
            ranked = self.data.sort_values('stability_score', ascending=False)
            return list(ranked['qubit_id'].values)
        except Exception as e:
            raise RuntimeError(f"Error computing stability rank: {str(e)}")
    
    def get_t1(self, qubit_id: int) -> float:
        """Get T1 relaxation time (microseconds) for a qubit."""
        try:
            row = self.data[self.data['qubit_id'] == qubit_id]
            if row.empty:
                raise ValueError(f"Qubit {qubit_id} not found")
            return float(row['t1_us'].values[0])
        except Exception as e:
            raise RuntimeError(f"Error getting T1 for qubit {qubit_id}: {str(e)}")
    
    def get_t2(self, qubit_id: int) -> float:
        """Get T2 dephasing time (microseconds) for a qubit."""
        try:
            row = self.data[self.data['qubit_id'] == qubit_id]
            if row.empty:
                raise ValueError(f"Qubit {qubit_id} not found")
            return float(row['t2_us'].values[0])
        except Exception as e:
            raise RuntimeError(f"Error getting T2 for qubit {qubit_id}: {str(e)}")
    
    def get_error_rate(self, qubit_id: int) -> float:
        """Get error rate for a qubit."""
        try:
            row = self.data[self.data['qubit_id'] == qubit_id]
            if row.empty:
                raise ValueError(f"Qubit {qubit_id} not found")
            return float(row['error_rate'].values[0])
        except Exception as e:
            raise RuntimeError(f"Error getting error rate for qubit {qubit_id}: {str(e)}")
    
    def build_noise_model(self, gate_time_1q: float = 50e-9, gate_time_2q: float = 300e-9) -> NoiseModel:
        """
        Build a Qiskit NoiseModel with thermal relaxation errors.
        
        Properly separates 1-qubit and 2-qubit errors:
        - 1-qubit thermal errors applied to single-qubit gates
        - 2-qubit thermal errors applied to two-qubit gates
        
        Args:
            gate_time_1q (float): Single-qubit gate duration in seconds (default: 50ns).
            gate_time_2q (float): Two-qubit gate duration in seconds (default: 300ns).
            
        Returns:
            NoiseModel: Qiskit noise model with thermal relaxation on all qubits.
        """
        try:
            noise_model = NoiseModel()
            
            # Add 1-qubit errors for single-qubit gates
            for idx in range(self.num_qubits):
                t1 = self.get_t1(idx) * 1e-6  # Convert microseconds to seconds
                t2 = self.get_t2(idx) * 1e-6
                
                # Create 1-qubit thermal relaxation error
                error_1q = thermal_relaxation_error(t1, t2, gate_time_1q)
                
                # Apply ONLY to single-qubit gates
                noise_model.add_quantum_error(error_1q, ['u', 'h', 'x', 'rz', 'sx'], [idx])
            
            # Add 2-qubit errors for two-qubit gates on all connected pairs
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i != j:
                        t1_i = self.get_t1(i) * 1e-6
                        t2_i = self.get_t2(i) * 1e-6
                        t1_j = self.get_t1(j) * 1e-6
                        t2_j = self.get_t2(j) * 1e-6
                        
                        # Create 2-qubit thermal relaxation error by expanding two single-qubit errors
                        error_1q_i = thermal_relaxation_error(t1_i, t2_i, gate_time_2q)
                        error_1q_j = thermal_relaxation_error(t1_j, t2_j, gate_time_2q)
                        error_2q = error_1q_i.expand(error_1q_j)
                        
                        # Apply ONLY to 2-qubit cx gates with explicit qubit pair indices
                        noise_model.add_quantum_error(error_2q, ['cx'], [i, j])
            
            return noise_model
            
        except Exception as e:
            raise RuntimeError(f"Error building noise model: {str(e)}")
    
    def visualize_temperatures(self, output_path: Optional[str] = None) -> None:
        """
        Visualize qubit temperatures as a heatmap.
        
        Args:
            output_path (str, optional): Path to save figure. If None, shows plot.
        """
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Prepare data for heatmap
            temps = self.data.sort_values('qubit_id')[['qubit_id', 'temperature']].reset_index(drop=True)
            
            # Create heatmap
            temp_matrix = temps['temperature'].values.reshape(1, -1)
            sns.heatmap(temp_matrix, annot=True, fmt='.1f', cmap='coolwarm',
                       cbar_kws={'label': 'Temperature (K)'}, ax=ax,
                       xticklabels=[f'q{i}' for i in range(self.num_qubits)])
            
            ax.set_title('Qubit Temperatures', fontsize=14, fontweight='bold')
            ax.set_ylabel('Device')
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            raise RuntimeError(f"Error visualizing temperatures: {str(e)}")
    
    def visualize_noise_scores(self, output_path: Optional[str] = None) -> None:
        """
        Visualize noise scores for all qubits as a bar chart.
        
        Args:
            output_path (str, optional): Path to save figure. If None, shows plot.
        """
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            data_sorted = self.data.sort_values('qubit_id')
            x_labels = [f'q{int(q)}' for q in data_sorted['qubit_id']]
            
            bars = ax.bar(x_labels, data_sorted['computed_noise_score'], 
                         color='coral', edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Qubit', fontsize=12, fontweight='bold')
            ax.set_ylabel('Noise Score', fontsize=12, fontweight='bold')
            ax.set_title('Thermal Noise Score by Qubit', fontsize=14, fontweight='bold')
            ax.set_ylim(0, max(data_sorted['computed_noise_score']) * 1.15)
            ax.grid(axis='y', alpha=0.3)
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            raise RuntimeError(f"Error visualizing noise scores: {str(e)}")
    
    def get_statistics(self) -> Dict:
        """
        Get statistical summary of thermal parameters.
        
        Returns:
            Dict: Statistics including mean, std, min, max for temperature, error rate, etc.
        """
        try:
            stats = {
                'num_qubits': self.num_qubits,
                'temperature_mean': float(self.data['temperature'].mean()),
                'temperature_std': float(self.data['temperature'].std()),
                'temperature_min': float(self.data['temperature'].min()),
                'temperature_max': float(self.data['temperature'].max()),
                'error_rate_mean': float(self.data['error_rate'].mean()),
                'error_rate_std': float(self.data['error_rate'].std()),
                'stability_mean': float(self.data['stability_score'].mean()),
            }
            return stats
        except Exception as e:
            raise RuntimeError(f"Error computing statistics: {str(e)}")
    
    def __str__(self) -> str:
        """String representation of thermal model."""
        stats = self.get_statistics()
        return (f"ThermalModel({self.num_qubits} qubits)\n"
                f"  Temp: {stats['temperature_mean']:.1f} +/- {stats['temperature_std']:.1f} K\n"
                f"  Error Rate: {stats['error_rate_mean']:.4f} +/- {stats['error_rate_std']:.4f}\n"
                f"  Stability: {stats['stability_mean']:.2f}")


if __name__ == "__main__":
    # Test thermal model
    import sys
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    thermal_file = os.path.join(project_root, 'data', 'thermal_noise.csv')
    
    print(f"Loading thermal model from: {thermal_file}")
    
    try:
        thermal = ThermalModel(thermal_file)
        print(thermal)
        
        # Test methods
        print(f"\nStability ranking (best to worst): {thermal.stability_rank()}")
        
        print(f"Noise score for q0: {thermal.thermal_noise_score(0):.4f}")
        print(f"T1 for q1: {thermal.get_t1(1)} us")
        print(f"T2 for q2: {thermal.get_t2(2)} us")
        
        noise_model = thermal.build_noise_model()
        print(f"\nNoise model built successfully")
        
        print("\nThermal model test passed!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
