import pandas as pd
import numpy as np
import scipy.stats as stats
import argparse
import os

def bootstrap_ci(data, n_resamples=1000):
    if len(data) < 2: return data.mean(), data.mean()
    res = stats.bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=n_resamples, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

def bootstrap_paired(d1, d2, n_resamples=1000):
    diff = d1 - d2
    if len(diff) < 2: return diff.mean(), diff.mean()
    res = stats.bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_resamples, method='percentile')
    return res.confidence_interval.low, res.confidence_interval.high

def analyze():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='results/sim/phase_3_sweep.csv')
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"File {args.file} not found.")
        return
        
    df = pd.read_csv(args.file)
    
    for device in df['device'].unique():
        for n_val in sorted(df['N'].unique()):
            for b_name in df['benchmark'].unique():
                for metric_type in ['success_prob', 'fidelity']:
                    sub = df[(df['device'] == device) & (df['N'] == n_val) & (df['benchmark'] == b_name) & (df['metric_type'] == metric_type)]
                    if sub.empty: continue
                    
                    print(f"\n{'='*60}")
                    print(f"Device: {device} | N: {n_val} | Benchmark: {b_name} | Metric: {metric_type}")
                    print(f"{'='*60}")
                    
                    # Group data
                    methods = sub['method'].unique()
                    
                    # For paired difference vs qiskit_L3, we need to align by sim_seed
                    # Wait, profile-dependent vs profile-independent methods!
                    # L0/L1/L3, esp_no_thermal are profile-independent. They were run once per sim_seed and broadcasted?
                    # Actually, in the CSV they have a 'profile_seed' column. But they are identical across profile_seeds!
                    # To avoid artificial replication, we should drop duplicates for profile-independent methods.
                    
                    # L3 baseline for paired diff
                    l3_data = sub[sub['method'] == 'qiskit_L3'].drop_duplicates(subset=['sim_seed', 'draw_id'])
                    
                    print(f"{'Method':<16} | {'Mean':<6} | {'95% CI':<15} | {'Diff vs L3 (CI)':<20} | {'Wilcoxon p':<10} | {'Map sec':<8}")
                    print("-" * 85)
                    
                    for m in methods:
                        m_data = sub[sub['method'] == m]
                        
                        if m in ['qiskit_L0', 'qiskit_L1', 'qiskit_L3', 'esp_no_thermal']:
                            m_data = m_data.drop_duplicates(subset=['sim_seed', 'draw_id'])
                            
                        vals = m_data['metric_val'].values
                        if len(vals) == 0: continue
                        
                        mean_val = vals.mean()
                        ci_low, ci_high = bootstrap_ci(vals)
                        map_sec = m_data['mapping_seconds'].mean()
                        
                        diff_str = "N/A"
                        p_val_str = "N/A"
                        
                        if m != 'qiskit_L3' and not l3_data.empty:
                            # Align by sim_seed (and draw_id)
                            # For profile-dependent, we average over profile_seeds first per sim_seed?
                            # Or we replicate L3? The prompt says: "Do not replicate profile-independent methods... across thermal profiles when bootstrapping".
                            # Wait, for PAIRED diff, we need equal length. We can average the profile-dependent methods over profile_seeds for each sim_seed, then pair with L3's sim_seeds!
                            m_agg = m_data.groupby(['sim_seed', 'draw_id'])['metric_val'].mean().reset_index()
                            merged = pd.merge(m_agg, l3_data, on=['sim_seed', 'draw_id'], suffixes=('_m', '_l3'))
                            if len(merged) >= 2:
                                d1 = merged['metric_val_m'].values
                                d2 = merged['metric_val_l3'].values
                                mean_diff = (d1 - d2).mean()
                                d_low, d_high = bootstrap_paired(d1, d2)
                                diff_str = f"{mean_diff:+.3f} [{d_low:+.3f}, {d_high:+.3f}]"
                                
                                # Wilcoxon
                                try:
                                    if np.all(d1 == d2): p_val = 1.0
                                    else: _, p_val = stats.wilcoxon(d1, d2)
                                    p_val_str = f"{p_val:.3e}"
                                except ValueError:
                                    p_val_str = "N/A"
                                    
                        print(f"{m:<16} | {mean_val:.4f} | [{ci_low:.3f}, {ci_high:.3f}] | {diff_str:<20} | {p_val_str:<10} | {map_sec:.4f}")
                        
                        # "for random, mean and spread over >=50 layouts."
                        if m == 'random':
                            n_layouts = m_data['draw_id'].nunique()
                            spread = vals.std()
                            print(f"  -> Random layouts: {n_layouts} distinct (spread std={spread:.4f})")
                            
                    # Spearman
                    print(f"\n--- Within-cell Spearman correlation ---")
                    # Using >=50 distinct layouts per cell. Random method provides this.
                    rand_data = sub[sub['method'] == 'random']
                    if not rand_data.empty:
                        # Average over sim_seeds to get expected metric per layout
                        agg_rand = rand_data.groupby('draw_id').agg({
                            'metric_val': 'mean',
                            'esp_standard': 'first',
                            'esp_thermal': 'first',
                            'esp_thermal_gate': 'first'
                        }).reset_index()
                        
                        if len(agg_rand) >= 2:
                            r_std, p_std = stats.spearmanr(agg_rand['esp_standard'], agg_rand['metric_val'])
                            r_thm, p_thm = stats.spearmanr(agg_rand['esp_thermal'], agg_rand['metric_val'])
                            r_thm_gate, p_thm_gate = stats.spearmanr(agg_rand['esp_thermal_gate'], agg_rand['metric_val'])
                            print("This checks consistency, since ESP and Aer read the same Target numbers.")
                            print(f"  esp_standard     vs {metric_type}: r={r_std:.3f} (p={p_std:.3e})")
                            print(f"  esp_thermal      vs {metric_type}: r={r_thm:.3f} (p={p_thm:.3e})")
                            print(f"  esp_thermal_gate vs {metric_type}: r={r_thm_gate:.3f} (p={p_thm_gate:.3e})")

if __name__ == '__main__':
    analyze()
