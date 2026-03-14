import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="darkgrid")
plt.ioff()  # Turn off interactive mode

def generate_all_plots(methods, swap_counts, noise_scores, 
                       fidelity, save_dir="results/graphs"):
    
    os.makedirs(save_dir, exist_ok=True)
    colors_swap    = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    colors_noise   = ['#FFA07A', '#C0C0C0', '#DDA0DD']
    colors_fidelity= ['#98FB98', '#FFDAB9', '#FFB6C1']
    
    print(f"\n{'='*80}")
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # ── PLOT 1: SWAP Count ─────────────────────────────────
    print("\n  Generating SWAP count comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, swap_counts, color=colors_swap, 
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, swap_counts):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + 0.1,
                str(val), ha='center', va='bottom', 
                fontsize=13, fontweight='bold')
    ax.set_title('SWAP Gate Count Comparison', 
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('SWAP Count', fontsize=12)
    ax.set_ylim(0, max(swap_counts) * 1.25)
    plt.tight_layout()
    swap_path = os.path.join(save_dir, 'swap_comparison.png')
    plt.savefig(swap_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved: {swap_path}")

    # ── PLOT 2: Noise Score ────────────────────────────────
    print("\n  Generating noise score comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, noise_scores, color=colors_noise,
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, noise_scores):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    ax.set_title('Average Thermal Noise Score Comparison',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Noise Score', fontsize=12)
    ax.set_ylim(0, max(noise_scores) * 1.25)
    plt.tight_layout()
    noise_path = os.path.join(save_dir, 'noise_score.png')
    plt.savefig(noise_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved: {noise_path}")

    # ── PLOT 3: Fidelity ───────────────────────────────────
    print("\n  Generating fidelity comparison...")
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(methods, fidelity, color=colors_fidelity,
                  edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, fidelity):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold')
    ax.set_title('Fidelity Comparison (Ideal vs Noisy)',
                 fontsize=15, fontweight='bold', pad=15)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_ylim(0.9, 1.0)
    plt.tight_layout()
    fidelity_path = os.path.join(save_dir, 'fidelity_comparison.png')
    plt.savefig(fidelity_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved: {fidelity_path}")

    # ── PLOT 4: Combined Summary ───────────────────────────
    print("\n  Generating combined summary...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Thermal-Aware Qubit Mapping — Full Summary',
                 fontsize=16, fontweight='bold')
    
    datasets = [
        (swap_counts,  'SWAP Count',   colors_swap,     'integer'),
        (noise_scores, 'Noise Score',  colors_noise,    'float'),
        (fidelity,     'Fidelity',     colors_fidelity, 'float'),
    ]
    titles = ['SWAP Gate Count', 'Thermal Noise Score', 'Fidelity']
    
    for ax, (data, ylabel, cols, fmt), title in \
            zip(axes, datasets, titles):
        bars = ax.bar(methods, data, color=cols,
                      edgecolor='black', linewidth=1.2)
        for bar, val in zip(bars, data):
            label = str(val) if fmt=='integer' else f'{val:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(data)*0.02,
                    label, ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0, max(data) * 1.3)
        if ylabel == 'Fidelity':
            ax.set_ylim(0.9, 1.01)
    
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'full_summary.png')
    plt.savefig(summary_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved: {summary_path}")
    
    print(f"\n[OK] 4 plots saved to {save_dir}/")
    print("="*80 + "\n")
