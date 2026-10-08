import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from qiskit import transpile
from src.runner_ibm import get_ibm_service
from src.benchmarks.circuits import get_ghz

def main():
    service = get_ibm_service()
    backend = service.backend("ibm_marrakesh")
    target = backend.target
    
    # 1. Transpile GHZ-5 onto the ESP chosen layout on ibm_marrakesh
    layout_esp = [32, 33, 34, 35, 19]
    qc_ghz = get_ghz(5)
    tc_esp = transpile(qc_ghz, target=target, initial_layout=layout_esp, optimization_level=3)
    
    # Draw circuit and save
    os.makedirs("results/figures", exist_ok=True)
    fig_circ = tc_esp.draw(output='mpl', style='iqp', plot_barriers=True, fold=25)
    circ_path = "results/figures/slide10_transpiled_circuit.png"
    fig_circ.savefig(circ_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_circ)
    print(f"Saved circuit diagram to {circ_path}")
    
    # Extract circuit stats
    ops = tc_esp.count_ops()
    two_q_gates = ops.get('cz', ops.get('cx', ops.get('ecr', 0)))
    depth_val = tc_esp.depth()
    print(f"GHZ-5 ESP on ibm_marrakesh -> 2Q gates: {two_q_gates}, Depth: {depth_val}, Layout: {layout_esp}")
    
    # 2. Read real hardware results from runtime_table.csv
    df = pd.read_csv("results/hardware/runtime_table.csv")
    sub = df[df['benchmark'] == 'ghz']
    
    methods_order = ['random', 'qiskit_L1', 'qiskit_L3', 'esp_no_thermal', 'esp_thermal']
    display_names = ['Random', 'Qiskit L1', 'Qiskit L3', 'ESP', 'Thermal ESP']
    
    means = []
    cis = []
    for m in methods_order:
        vals = sub[sub['method'] == m]['metric_val'].values
        p_mean = np.mean(vals)
        # 16,384 total shots
        se = np.sqrt(p_mean * (1.0 - p_mean) / 16384)
        means.append(p_mean)
        cis.append(1.96 * se)
        
    # Generate presentation bar chart
    bg_color = '#F3F4F1'
    text_color = '#1E1E1E'
    purple_highlight = '#8B5CF6'
    slate_bar = '#64748B'
    blue_bar = '#3B82F6'
    
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=bg_color)
    ax.set_facecolor(bg_color)
    
    colors = [slate_bar, slate_bar, slate_bar, blue_bar, purple_highlight]
    bars = ax.bar(display_names, means, yerr=cis, capsize=5, color=colors,
                  width=0.55, error_kw={'elinewidth': 1.8, 'ecolor': '#333333', 'capthick': 1.8})
                  
    ax.set_ylim(0.65, 0.98)
    ax.set_ylabel('Success Probability', fontsize=12, fontweight='bold', color=text_color)
    ax.set_title('Success Probability by Method (95% CI)\nIBM Quantum Marrakesh (156 Qubits) · GHZ-5',
                 fontsize=13, fontweight='bold', color=text_color, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='#CBD5E1')
    ax.tick_params(colors=text_color, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#94A3B8')
        
    for bar, m_val, ci_val in zip(bars, means, cis):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, y + ci_val + 0.008,
                f"{m_val*100:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold', color=text_color)
                
    plt.tight_layout()
    chart_path = "results/figures/slide11_hardware_results_chart.png"
    fig.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor=bg_color)
    plt.close(fig)
    print(f"Saved results chart to {chart_path}")

if __name__ == "__main__":
    main()
