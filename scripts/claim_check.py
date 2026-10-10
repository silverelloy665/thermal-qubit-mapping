import os
import sys
import re
import pandas as pd
import numpy as np

def compute_ground_truth(csv_path="results/hardware/runtime_table.csv"):
    df = pd.read_csv(csv_path)
    gt = {}
    
    z = 1.96
    benchmarks = ['ghz', 'bv', 'qft', 'mirror']
    methods = ['random', 'qiskit_L1', 'qiskit_L3', 'qiskit_L3_best3', 'esp_no_thermal', 'esp_thermal']
    
    for b in benchmarks:
        gt[b] = {}
        for m in methods:
            sub = df[(df['benchmark'] == b) & (df['method'] == m)]
            p = float(sub['metric_val'].mean())
            n = 16384  # 8192 shots * 2 repetitions
            denom = 1 + z**2 / n
            center = (p + z**2 / (2 * n)) / denom
            diff = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
            gt[b][m] = {
                'mean': p * 100.0,
                'ci_delta': diff * 100.0,
                'ci_lower': (center - diff) * 100.0,
                'ci_upper': (center + diff) * 100.0,
            }
            
        esp_p = gt[b]['esp_no_thermal']['mean']
        rnd_p = gt[b]['random']['mean']
        l3_p = gt[b]['qiskit_L3']['mean']
        gt[b]['esp_minus_random'] = esp_p - rnd_p
        gt[b]['esp_minus_l3'] = esp_p - l3_p
        
    return gt

def audit_readme(readme_path="README.md", csv_path="results/hardware/runtime_table.csv"):
    if not os.path.exists(readme_path):
        print(f"Error: {readme_path} not found.")
        sys.exit(1)
        
    gt = compute_ground_truth(csv_path)
    
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Check for forbidden uncited percentage gains like +12%
    valid_plus_pcts = []
    for b in ['ghz', 'bv', 'qft', 'mirror']:
        diff_rnd = gt[b]['esp_minus_random']
        if diff_rnd > 0:
            valid_plus_pcts.append(f"+{diff_rnd:.2f}")
            valid_plus_pcts.append(f"+{diff_rnd:.1f}")
        diff_l3 = gt[b]['esp_minus_l3']
        if diff_l3 > 0:
            valid_plus_pcts.append(f"+{diff_l3:.2f}")
            valid_plus_pcts.append(f"+{diff_l3:.1f}")
            
    found_plus = re.findall(r'\+(\d+(?:\.\d+)?)\s*%', content)
    for p in found_plus:
        val = float(p)
        matched = False
        for vp in valid_plus_pcts:
            if abs(val - float(vp)) <= 0.05:
                matched = True
                break
        if not matched:
            print(f"FAILED: Found uncited gain claim '+{p}%' in {readme_path} not supported by ground truth.")
            return False, f"Uncited gain claim '+{p}%'"

    # Extract all numbers quoted in the format: $XX.XX\% \pm Y.YY\%$
    pattern = re.compile(r"\$(\d+\.\d+)\\\%\s*\\pm\s*(\d+\.\d+)\\\%\$")
    matches = pattern.findall(content)
    
    print(f"Found {len(matches)} quoted (mean, CI) pairs in {readme_path}.")
    if len(matches) == 0:
        print("FAILED: No (mean, CI) pairs found in README.md.")
        return False, "No pairs found"
        
    # All quoted numbers must match some benchmark/method in gt within 0.05 pt
    all_gt_pairs = []
    benchmarks = ['ghz', 'bv', 'qft', 'mirror']
    methods = ['random', 'qiskit_L1', 'qiskit_L3', 'qiskit_L3_best3', 'esp_no_thermal', 'esp_thermal']
    for b in benchmarks:
        for m in methods:
            all_gt_pairs.append((gt[b][m]['mean'], gt[b][m]['ci_delta']))
            
    for m_mean_str, m_ci_str in matches:
        m_mean = float(m_mean_str)
        m_ci = float(m_ci_str)
        matched = False
        for exp_mean, exp_ci in all_gt_pairs:
            if abs(m_mean - exp_mean) <= 0.05 and abs(m_ci - exp_ci) <= 0.05:
                matched = True
                break
        if not matched:
            print(f"FAILED: Quoted value ${m_mean}% \\pm {m_ci}%$ in README does not match any ground truth entry.")
            return False, f"Mismatch for ${m_mean}% \\pm {m_ci}%$"

    print("Claim audit passed: all numbers in README.md match runtime_table.csv within 0.05 pt.")
    return True, "All claims verified"

if __name__ == "__main__":
    success, msg = audit_readme()
    if not success:
        sys.exit(1)
    sys.exit(0)
