# Thermal Qubit Mapping

This repository contains the codebase for the project "ESP-based noise-aware qubit mapping with a thermal (excited-state population p1) term", targeting the Qiskit Fall Fest 2026 @ MPSTME, track "Noise-Aware Quantum Computing".

## Project Overview

Quantum bits (qubits) implemented using superconducting transmon technology are subject to environmental noise. One significant source of error is thermal relaxation and excitation, leading to non-zero excited-state populations ($p_1$). Traditional qubit mapping and routing algorithms optimize for metrics like gate count or standard Estimated Success Probability (ESP), but often neglect the spatially and temporally varying thermal noise across the quantum processing unit (QPU).

This project extends the standard ESP metric to incorporate a thermal penalty term. By evaluating the thermal profile of a real IBM Quantum device, we can intelligently map circuits onto physical qubits that not only have high gate fidelities but also lower thermal populations, thereby maximizing the true fidelity of the executed quantum circuit.

### Key Contributions
1. **Thermal ESP Metric**: We define `esp_thermal` which scales the standard ESP by the expected thermal errors. We explore two formulations:
   - `raw_upper_bound`: Penalizes every operation based on $p_1$.
   - `excess_over_readout`: Penalizes only the thermal error that exceeds the calibrated readout error.
2. **Thermal-Aware Mapper**: A heuristic search mapper that explores connected subgraphs of the QPU, evaluating both standard and thermal ESP to find the optimal layout.
3. **Simulation and Hardware Validation**: We validate the mappers using `Qiskit Aer` density matrix simulations with injected thermal noise models, followed by real hardware executions via `qiskit-ibm-runtime` (SamplerV2).

## Repository Structure

- `src/benchmarks/`: Quantum circuits for evaluation (GHZ, BV, QFT, QAOA, Routing Pressure).
- `src/mappers/`: Implementation of `mapper_qiskit_default`, `mapper_random`, and `mapper_esp`.
- `src/metrics/`: Functions to calculate standard ESP and thermal ESP (`esp.py`).
- `src/noise_model/`: Tools to build custom Qiskit Aer noise models reflecting specific thermal configurations (`thermal.py`).
- `scripts/`: Executable scripts for running simulations (`run_sim.py`), exact sanity checks (`exact_sanity_check.py`), density matrix validations (`density_matrix_check.py`), and hardware submissions (`run_hardware.py`).
- `tests/`: Comprehensive `pytest` suite ensuring metric and mapping correctness.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/silverelloy665/thermal-qubit-mapping.git
   cd thermal-qubit-mapping
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Or .venv\\Scripts\\activate on Windows
   pip install -r requirements.txt
   ```

3. **Configure IBM Quantum Credentials**:
   Do not put your credentials in code or `.env` files tracked by git. Instead, save your account locally using the Qiskit Runtime Service:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<YOUR_TOKEN>", overwrite=True)
   ```

## Running Simulations

The simulation suite sweeps across different benchmarks, noise configurations, and mappers.
```bash
python scripts/run_sim.py
```
This generates a CSV output in `results/sim/`.

## Running Hardware Jobs

Hardware jobs are subject to a strict budget constraint (e.g., 600 QPU seconds).
To perform a dry run and estimate QPU usage:
```bash
python scripts/run_hardware.py
```
To explicitly execute the jobs on the QPU, pass the execution flags:
```bash
python scripts/run_hardware.py --execute --approve-seconds
```

## Testing

Run the test suite to verify metric bounds, mapper connectivity, and distribution correctness:
```bash
pytest -v tests/test_all.py
```
