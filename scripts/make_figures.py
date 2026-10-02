import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from qiskit_ibm_runtime.fake_provider import FakeVigoV2
import networkx as nx

sns.set_theme(style="whitegrid")

def plot_bar_charts(df, out_dir):
    metrics = ['fidelity', 'esp', 'cx_count', 'depth']
    titles = ['Fidelity (Hellinger)', 'Estimated Success Probability (ESP)', 'CX Gate Count', 'Circuit Depth']
    
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(10, 6))
        # Bootstrapped CI is default in seaborn barplot
        sns.barplot(data=df, x='benchmark', y=metric, hue='method', capsize=.1, errorbar=('ci', 95))
        plt.title(f"{title} by Method across Benchmarks")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_comparison.png", dpi=300)
        plt.savefig(out_dir / f"{metric}_comparison.svg")
        plt.close()

def plot_scatter(df, out_dir):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='esp', y='fidelity', hue='method', style='benchmark', s=100)
    
    # correlation
    corr = df['esp'].corr(df['fidelity'])
    plt.title(f"Predicted ESP vs Measured Fidelity (r = {corr:.3f})")
    
    # y=x line
    min_val = min(df['esp'].min(), df['fidelity'].min())
    max_val = max(df['esp'].max(), df['fidelity'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_dir / "esp_vs_fidelity.png", dpi=300)
    plt.savefig(out_dir / "esp_vs_fidelity.svg")
    plt.close()

def plot_heatmap(out_dir):
    backend = FakeVigoV2()
    target = backend.target
    
    num_qubits = target.num_qubits
    data = []
    
    for q in range(num_qubits):
        q_props = target.qubit_properties[q]
        t1 = getattr(q_props, 't1', 0) * 1e6 # us
        t2 = getattr(q_props, 't2', 0) * 1e6 # us
        
        readout_error = 0
        try:
            m_props = target['measure'].get((q,), None)
            if m_props:
                readout_error = getattr(m_props, 'error', 0)
        except KeyError:
            pass
            
        data.append({
            'Qubit': q,
            'T1 (us)': t1,
            'T2 (us)': t2,
            'Readout Error': readout_error
        })
        
    df_props = pd.DataFrame(data).set_index('Qubit')
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_props, annot=True, fmt=".3g", cmap="YlOrRd")
    plt.title("Backend Calibration Heatmap (FakeVigoV2)")
    plt.tight_layout()
    plt.savefig(out_dir / "calibration_heatmap.png", dpi=300)
    plt.close()

def main():
    res_dir = Path("results")
    out_dir = res_dir / "figures"
    os.makedirs(out_dir, exist_ok=True)
    
    sim_csv = res_dir / "sim" / "phase_a_results.csv"
    if sim_csv.exists():
        df = pd.read_csv(sim_csv)
        plot_bar_charts(df, out_dir)
        plot_scatter(df, out_dir)
        plot_heatmap(out_dir)
        print("Figures generated in results/figures/")
    else:
        print(f"Results not found at {sim_csv}")

if __name__ == "__main__":
    main()
