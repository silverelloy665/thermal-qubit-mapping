"""
Thermal-Aware Qubit Mapping Optimization

A complete quantum computing application that implements thermal-aware qubit mapping
to optimize circuit execution on NISQ devices, considering thermal noise and hardware topology.
"""

__version__ = "1.0.0"
__author__ = "Quantum Computing Team"

# Import main components
from . import circuit_loader
from . import topology_model
from . import thermal_model
from . import mapping_algorithm
from . import optimizer
# from . import evaluator  # Disabled: not used in new optimizer pipeline

__all__ = [
    'circuit_loader',
    'topology_model',
    'thermal_model',
    'mapping_algorithm',
    'optimizer',
    # 'evaluator',
]
