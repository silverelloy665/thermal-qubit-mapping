# Thermal Qubit Mapping: Noise-Aware Mapping with Excited-State Population ($p_1$)

[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen.svg)]()
[![Backend](https://img.shields.io/badge/IBM%20Quantum-ibm__marrakesh%20(156q)-blue.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Heron%20r2%20(CZ)-purple.svg)]()
[![Budget](https://img.shields.io/badge/QPU%20Cap-350s%20(250s%20reserve)-orange.svg)]()

This repository contains the full research implementation, simulation benchmarks, and live IBM Quantum hardware verification for **"ESP-Based Noise-Aware Qubit Mapping with an Excited-State Population ($p_1$) Thermal Term"**, prepared for the **Qiskit Fall Fest 2026 @ MPSTME** (Track: *Noise-Aware Quantum Computing*).

---

## 1. Project Overview & Motivation

Superconducting transmon quantum processors are intrinsically susceptible to environmental thermal fluctuations and non-equilibrium quasiparticle excitations. While transmon devices are refrigerated to dilution-refrigerator temperatures (~15 mK), physical qubits experience non-zero steady-state excited-state populations ($p_1 = P(1|0)$). 

Standard quantum compilers and noise-aware routing algorithms (such as standard Estimated Success Probability or ESP) incorporate single-qubit and two-qubit gate error rates ($\epsilon_{1q}, \epsilon_{2q}$) and readout assignment errors ($\epsilon_{ro}$). However, they routinely treat thermal excitation as either uniform or implicitly subsumed into relaxation times ($T_1$). In multi-qubit QPUs, local thermal hot-spots, asymmetric drive line heating, and quasiparticle poisoning generate spatial and temporal heterogeneity in $p_1$.

This project introduces a **Thermal-Aware ESP metric** that penalizes qubit layouts mapped onto physically hot or thermally fluctuating subgraphs. We evaluate this metric through:
1. **Extensive Aer Simulations**: Modeling synthetic thermal noise models across varying temperature profiles (15 mK to 50 mK) and testing over 20+ unit test constraints.
2. **Real Hardware Benchmarking on IBM Quantum**: Executing 4 canonical quantum algorithms (GHZ, Bernstein-Vazirani, Quantum Fourier Transform, and Mirror Random Circuits) across 6 mapping strategies on the 156-qubit Heron r2 processor `ibm_marrakesh`.

---

## 2. Theoretical Framework & Metric Formulations

### Standard Estimated Success Probability (ESP)
The baseline ESP of a transpiled quantum circuit mapped to a physical layout is computed as the product of gate and measurement fidelities:
$$\text{ESP}_{\text{std}} = \prod_{g \in \mathcal{G}_{1q}} (1 - \epsilon_g) \prod_{g \in \mathcal{G}_{2q}} (1 - \epsilon_g) \prod_{q \in \mathcal{Q}_{\text{active}}} (1 - \epsilon_{\text{ro}, q})$$

### Thermal Penalty Formulations
To penalize thermal excitation while preventing double-counting of readout assignment errors, we formulate two modes:

1. **Excess Over Readout (`excess_over_readout`, Recommended)**:
   Because laboratory measurement of $P(1|0)$ naturally contains readout misclassification errors ($\epsilon_{\text{ro}}$), the true excess thermal population is bounded by:
   $$p_{1, \text{excess}}(q) = \max(0, P(1|0)_q - \epsilon_{\text{ro}, q})$$
   $$\text{ESP}_{\text{thermal}} = \text{ESP}_{\text{std}} \times \prod_{q \in \mathcal{Q}_{\text{active}}} (1 - p_{1, \text{excess}}(q))$$

2. **Raw Upper Bound (`raw_upper_bound`)**:
   Treats the measured excited-state population as an independent upper-bound penalty on qubit initialization:
   $$\text{ESP}_{\text{thermal}} = \text{ESP}_{\text{std}} \times \prod_{q \in \mathcal{Q}_{\text{active}}} (1 - p_1(q))$$

---

## 3. Real IBM Quantum Hardware Execution (`ibm_marrakesh`)

The live QPU validation was conducted on IBM's flagship 156-qubit Heron r2 quantum processor **`ibm_marrakesh`** using Qiskit Runtime SamplerV2.

### Execution Provenance & Job Tracking

| Phase | Job ID | Backend | Circuits / PUBs | Shots | QPU Time | Status |
|---|---|---|---|---|---|---|
| **Pilot Run** | `db3ur4slf4us73c1v5pg` | `ibm_marrakesh` | 1 circuit | 8,192 | ~3.0 s | `DONE` |
| **Thermal $P_1$ Calibration** | `db3usd2mb58s7388dskg` | `ibm_marrakesh` | 2 PUBs ($|0\rangle, X|0\rangle$) | 4,096 | ~3.0 s | `DONE` |
| **Hardware Repetition 0** | `db3uv1cvf2bc73ctqt1g` | `ibm_marrakesh` | 17 de-duplicated circuits | 8,192 | 51.0 s | `DONE` |
| **Hardware Repetition 1** | `db3uvecvf2bc73ctqto0` | `ibm_marrakesh` | 17 de-duplicated circuits | 8,192 | 51.0 s | `DONE` |

- **Total Execution Time**: **102.0 s** (benchmark circuits) + ~6.0 s (pilot & calibration) = **~108 s total QPU time**.
- **Budget Compliance**: Configured budget cap of **350 s** strictly observed, leaving the **250 s reserve budget completely intact**.
- **Execution Mode**: Dedicated job mode, dynamical decoupling disabled, twirling disabled, preserving raw physical device noise characteristics.

---

## 4. Hardware Benchmark Results (Summary Table)

Below is the consolidated performance across all 48 experimental evaluations on `ibm_marrakesh` (8,192 shots $\times$ 2 repetitions = 16,384 total shots per data point, with 95% Wilson binomial confidence intervals):

| Benchmark | Method | Success / Fidelity (Mean ± 95% CI) | 2Q Gates (Native CZ) | Depth | Physical Layout |
|---|---|:---:|:---:|:---:|:---:|
| **GHZ-5** | **Random** | $74.23\% \pm 0.67\%$ | 17 | 32 | `[80, 61, 62, 76, 81]` |
| | **Qiskit L1** | $91.46\% \pm 0.43\%$ | 4 | 20 | `[0, 1, 2, 3, 4]` |
| | **Qiskit L3** | $91.60\% \pm 0.43\%$ | 4 | 16 | `[33, 34, 35, 19, 15]` |
| | **Qiskit L3_best3** | $91.60\% \pm 0.43\%$ | 4 | 16 | `[33, 34, 35, 19, 15]` |
| | **ESP (no thermal)** | **$90.28\% \pm 0.45\%$** | 4 | 16 | `[32, 33, 34, 35, 19]` |
| | **Thermal ESP** | **$90.28\% \pm 0.45\%$** | 4 | 16 | `[32, 33, 34, 35, 19]` |
| **BV-5** | **Random** | $86.35\% \pm 0.53\%$ | 2 | 14 | `[80, 61, 62, 76, 81]` |
| | **Qiskit L1** | $91.43\% \pm 0.43\%$ | 2 | 15 | `[3, 153, 6, 1, 2]` |
| | **Qiskit L3** | $91.17\% \pm 0.44\%$ | 2 | 14 | `[3, 153, 6, 1, 2]` |
| | **Qiskit L3_best3** | $91.17\% \pm 0.44\%$ | 2 | 14 | `[3, 153, 6, 1, 2]` |
| | **ESP (no thermal)** | **$86.32\% \pm 0.53\%$** | 2 | 14 | `[34, 19, 35, 32, 33]` |
| | **Thermal ESP** | **$86.32\% \pm 0.53\%$** | 2 | 14 | `[34, 19, 35, 32, 33]` |
| **QFT-5** | **Random** | $64.07\% \pm 0.74\%$ | 39 | 107 | `[80, 61, 62, 76, 81]` |
| | **Qiskit L1** | $85.97\% \pm 0.54\%$ | 41 | 122 | `[55, 39, 52, 54, 53]` |
| | **Qiskit L3** | $81.93\% \pm 0.59\%$ | 30 | 83 | `[55, 52, 39, 54, 53]` |
| | **Qiskit L3_best3** | $81.93\% \pm 0.59\%$ | 30 | 83 | `[55, 52, 39, 54, 53]` |
| | **ESP (no thermal)** | **$74.07\% \pm 0.67\%$** | 39 | 92 | `[35, 19, 34, 15, 14]` |
| | **Thermal ESP** | **$74.07\% \pm 0.67\%$** | 39 | 92 | `[35, 19, 34, 15, 14]` |
| **Mirror-5** | **Random** | $50.18\% \pm 0.76\%$ | 52 | 124 | `[80, 61, 62, 76, 81]` |
| | **Qiskit L1** | $75.51\% \pm 0.66\%$ | 45 | 112 | `[4, 2, 16, 1, 3]` |
| | **Qiskit L3** | $69.50\% \pm 0.71\%$ | 38 | 86 | `[54, 53, 39, 52, 55]` |
| | **Qiskit L3_best3** | $68.86\% \pm 0.71\%$ | 38 | 86 | `[54, 53, 39, 52, 55]` |
| | **ESP (no thermal)** | **$73.91\% \pm 0.67\%$** | 46 | 110 | `[39, 53, 55, 54, 33]` |
| | **Thermal ESP** | **$73.91\% \pm 0.67\%$** | 46 | 110 | `[39, 53, 55, 54, 33]` |

---

## 5. Key Scientific Insights

### Consolidated Benchmark Comparison (Mean ± 95% Wilson CI)

| Benchmark | Random | Qiskit L1 | Qiskit L3 | Qiskit L3_best3 | ESP (Plain / Thermal) |
|---|:---:|:---:|:---:|:---:|:---:|
| **GHZ-5** | $74.23\% \pm 0.67\%$ | $91.46\% \pm 0.43\%$ | $91.60\% \pm 0.42\%$ | $91.60\% \pm 0.42\%$ | $90.28\% \pm 0.45\%$ |
| **BV-5** | $86.35\% \pm 0.53\%$ | $91.43\% \pm 0.43\%$ | $91.17\% \pm 0.43\%$ | $91.17\% \pm 0.43\%$ | $86.32\% \pm 0.53\%$ |
| **QFT-5** | $64.07\% \pm 0.73\%$ | $85.97\% \pm 0.53\%$ | $81.93\% \pm 0.59\%$ | $81.93\% \pm 0.59\%$ | $74.07\% \pm 0.67\%$ |
| **Mirror-5** | $50.18\% \pm 0.77\%$ | $75.51\% \pm 0.66\%$ | $69.50\% \pm 0.70\%$ | $68.86\% \pm 0.71\%$ | $73.91\% \pm 0.67\%$ |

> [!NOTE]
> **Experimental Setup & Equivalence Notes**:
> - **Shared circuit => thermal == ESP by construction**: On `ibm_marrakesh`, thermal ESP and standard ESP selected the exact same physical layouts (`shared_circuit=True`) because measured $P(1|0)$ across the chosen active qubits remained near or below calibrated readout error $\epsilon_{\text{ro}}$ (excess $<0.02$ on all chosen qubits; e.g., mirror layout includes qubit 54 with excess 0.012). This identity is by construction of the de-duplicated circuit pipeline and does not represent independent empirical convergence.
> - **Heron r2 Native CZ Architecture**: `ibm_marrakesh` natively executes Controlled-Z (CZ) two-qubit operations, compiling GHZ-5 into exactly 4 CZ gates at minimal circuit depth 16.
> - **Scope & Limits**: All hardware measurements were conducted on one backend (`ibm_marrakesh`), during one calibration day, with one fixed transpiled layout per cell across 2 repetitions of 8,192 shots (16,384 total pooled shots per cell). Consequently, confidence intervals quantify shot noise only.
> - **Baseline Comparisons**: ESP significantly outperforms random qubit mapping by **+16.05 pts** on GHZ-5 ($90.28\%$ vs. $74.23\%$) and by **+23.73 pts** on Mirror-5 ($73.91\%$ vs. $50.18\%$). Compared against Qiskit L3, ESP yields -1.32 pts (GHZ), -4.85 pts (BV), -7.87 pts (QFT), and +4.41 pts (Mirror-5).

### Thermal-Gradient Sensitivity Sweep Results (`results/sim/thermal_sweep.csv`)

- **Baseline Comparative Performance vs Qiskit L3**:
  - At zero excess, ESP trails L3 by 8.0, 2.7, 7.5, and 8.8 pts (GHZ, BV, QFT, Mirror) on `FakeGuadalupeV2`.
  - In non-zero thermal excess regimes, thermal ESP beats L3 with a 95% bootstrap CI strictly greater than zero in only **1 of 72 hot cells** (GHZ-5, 2 hot qubits, $15\%$ excess: $+5.28\text{ pts}$, 95% CI $[+0.35, +9.81]$). By point estimate, thermal ESP beats L3 in **6 of 12 cells at 15% excess** (GHZ with 1, 2, 3 hot qubits; BV with 2, 3 hot qubits; QFT with 3 hot qubits); Mirror-5 never beats L3 across any parameter cell. The committed thermal_sweep.csv was generated before L3 transpile seeding; its L3 columns drifted by up to 0.07 pt in an unseeded rerun (Checkpoint K); seeded reruns reproduce the committed values on the tested cell (ghz, 2 hot, 0.15).
- **Physical Prevalence of High-Thermal Regimes**: On `ibm_marrakesh`, only **4 of 156 qubits** exhibit $p_{1, \text{excess}} > 0.10$ (qubits 125, 22, 26, 11); qubit 125 has readout error 0.0096 and excess 0.41 (thermal excess), whereas the readout-failure qubits ($\epsilon_{\text{ro}} > 0.20$) are 94 and 130. Consequently, regimes where thermal ESP provides definitive routing advantages over L3 are physically rare.
- **Guadalupe Practical Detection First Crossings**: On the 16-qubit `FakeGuadalupeV2` model, thermal ESP achieves practical sensitivity ($\text{gain} \ge 0.5\text{ pt}$ and 95% bootstrap CI lower bound $> 0$) at first crossing:
  - **GHZ-5**: $p_{1, \text{excess}} = 0.005$ across 1, 2, and 3 hot qubits (+1.26 pts, +2.40 pts, +1.82 pts).
  - **BV-5**: $p_{1, \text{excess}} = 0.020$ (1 hot qubit, +1.50 pts), $0.050$ (2 hot qubits, +1.91 pts), and $0.005$ (3 hot qubits, +0.57 pts).
  - **QFT-5**: $p_{1, \text{excess}} = 0.005$ (1 hot qubit, +2.38 pts) and $0.010$ (2 and 3 hot qubits, +2.36 pts, +1.56 pts).
  - **Mirror-5**: $p_{1, \text{excess}} = 0.020$ (1 and 3 hot qubits, +1.31 pts, +2.30 pts) and $0.100$ (2 hot qubits, +2.51 pts). Note that Mirror-5 (3 hot) loses the effect again at 0.05 (gain -0.12 pts with CI overlapping zero).
- **Vigo Topologically Uninformative**: On `FakeVigoV2` (5 qubits total), mapping $N=5$ benchmarks provides **no layout freedom**—all 5 qubits $\{0, 1, 2, 3, 4\}$ must be chosen, yielding identical layouts and exactly 0.00 gain across all noise profiles.
- **Mirror Benchmark Sensitivity Noise**: Due to deep multi-layer compilation variance, Mirror-5 exhibits high sensitivity noise; its 95% bootstrap confidence intervals overlap zero below $\sim 10\%$ excess population for 2 hot qubits.
- **Circularity Caveat**: The Aer noise simulator injects non-equilibrium populations via state-preparation bit-flip mixtures ($X$ with probability $p_{1, \text{excess}}$), which directly aligns with the penalization objective optimized by Thermal ESP.

### Methodological Limitations & Thermal Sensitivity Bounds

- **Vigo $N=5$ has no layout freedom**: On `FakeVigoV2` (5 qubits total), mapping any 5-qubit benchmark requires utilizing the entire chip $\{0, 1, 2, 3, 4\}$; there is no subgraph freedom to route around hot spots regardless of injected thermal excess.
- **10 seeds not 20**: The thermal sensitivity sweep evaluated 10 profile seeds per cell rather than 20 to operate within runtime budget constraints.
- **Hot qubits forced into the default layout in $\ge 50\%$ of draws**: To test evasion efficacy, at least half of all random seed draws explicitly place at least one hot qubit inside the default plain-ESP layout.
- **First crossing definition**: The pre-registered sensitivity first crossing is the smallest $p_{1, \text{excess}}$ where the paired-gain 95% bootstrap CI lower bound is strictly $> 0$.
- **Practical significance threshold**: Paired fidelity gains below 0.5 pt are marked "not practically meaningful".

---

## 6. Presentation Assets for Slides 10 & 11

The repository includes high-resolution publication assets specifically tailored for the presentation slides:

- **Slide 10 Transpiled Circuit Diagram**:
  `results/figures/slide10_transpiled_circuit.png`
  High-resolution rendering of GHZ-5 transpiled onto physical qubits `{19, 32, 33, 34, 35}` of `ibm_marrakesh`, highlighting the native CZ gate decomposition.
- **Slide 11 Real Hardware Benchmark Bar Chart**:
  `results/figures/slide11_hardware_results_chart.png`
  Publication-grade bar chart matching the slide background (`#F3F4F1`) with 95% Wilson confidence intervals, comparing Random (74.2%), Qiskit L1 (91.5%), Qiskit L3 (91.6%), ESP (90.3%), and Thermal ESP (90.3%).

---

## 7. Repository Structure

```
thermal-qubit-mapping/
├── config.yaml                    # Budget caps, reserve, and backend configurations
├── README.md                      # Complete project documentation
├── requirements.txt               # Pinned Python package dependencies
├── src/
│   ├── benchmarks/circuits.py     # GHZ, BV, QFT, and Mirror circuit generators
│   ├── mappers/mappers.py         # Random, Qiskit default/best3, and ESP heuristic mappers
│   ├── metrics/esp.py             # Standard and Thermal ESP implementations
│   ├── noise_model/thermal.py     # Thermal noise simulation generators
│   └── runner_ibm.py              # Secure Qiskit Runtime Service and backend loader
├── scripts/
│   ├── run_sim.py                 # Aer simulation sweep runner
│   ├── run_hardware.py            # Hardware runner with pilot & approve-seconds gate
│   ├── measure_p1.py              # Excited-state population measurement on QPU
│   ├── generate_slide_assets.py   # Regenerates diagram and plot for presentation slides
│   ├── audit_repo.py              # Strict repository verification script
│   └── verify_commit.py           # Automated size-shrink and regression guard
├── results/
│   ├── figures/                   # High-res slide charts and circuit diagrams
│   └── hardware/                  # Raw counts, runtime_table.csv, plan, and provenance sidecars
└── tests/
    └── test_all.py                # 20 rigorous pytest unit tests
```

---

## 8. Reproducibility & Commands

### 1. Run Unit Tests (20 Tests)
```bash
pytest -v tests/test_all.py
```

### 2. Run Repository Audit
```bash
python scripts/audit_repo.py
```

### 3. Run Simulation Suite
```bash
python scripts/run_sim.py
```

### 4. Execute Hardware Validation Protocol
```bash
# Step 1: Dry run to estimate QPU runtime and generate plan.json
python scripts/run_hardware.py

# Step 2: Single-circuit pilot calibration (~3s QPU)
python scripts/run_hardware.py --pilot

# Step 3: Measure QPU excited-state population p1
python scripts/measure_p1.py --execute --approve-seconds <N>

# Step 4: Re-run dry run to update plan with pilot scaling
python scripts/run_hardware.py

# Step 5: Full execution on IBM Quantum QPU
python scripts/run_hardware.py --execute --approve-seconds <M>
```

---

## 9. Security & Governance

- **Zero Hardcoded Secrets**: Credentials are authenticated exclusively via locally saved Qiskit accounts (`QiskitRuntimeService.save_account`) under channel `ibm_quantum_platform`.
- **Untracked Environment Files**: `.env` is strictly untracked and barred by `.gitignore`.
- **Approval Gate Enforcement**: Execution on the real QPU strictly requires explicit `--approve-seconds` and a valid pilot scaling file `< 24h` old.
