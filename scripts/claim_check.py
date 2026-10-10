import os
import sys
import re
import glob
import json
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

def audit_sweep_and_p1(readme_content: str, sweep_csv="results/sim/thermal_sweep.csv", p1_pattern="results/hardware/p1_ibm_marrakesh_*.json"):
    # 1. Verify zero-excess gaps vs L3 on Guadalupe
    if not os.path.exists(sweep_csv):
        return False, f"Sweep CSV {sweep_csv} not found."
    df_sw = pd.read_csv(sweep_csv)
    g0 = df_sw[(df_sw.device == 'fake_guadalupe') & (df_sw.excess_p1 == 0.0)]
    gaps = {}
    for b in ['ghz', 'bv', 'qft', 'mirror']:
        r = g0[g0.benchmark == b].iloc[0]
        # gap = mean_score_l3 - mean_score_esp_no
        gaps[b] = (r['mean_score_l3'] - r['mean_score_esp_no']) * 100.0
    
    # Assert values in README match within 0.1 pt
    expected_gaps = [f"{gaps[b]:.1f}" for b in ['ghz', 'bv', 'qft', 'mirror']] # ['8.0', '2.7', '7.5', '8.8']
    gap_pattern = r"trails L3 by (\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*and\s*(\d+\.\d+)\s*pts"
    m_gap = re.search(gap_pattern, readme_content)
    if not m_gap:
        return False, "Zero-excess gap sentence not found in README."
    quoted_gaps = list(m_gap.groups())
    for q_val, exp_val in zip(quoted_gaps, expected_gaps):
        if abs(float(q_val) - float(exp_val)) > 0.15:
            return False, f"Mismatch in quoted zero-excess gap: {q_val} vs expected {exp_val}"

    # 2. Verify L3 hot cell counts on Guadalupe
    g_hot = df_sw[(df_sw.device == 'fake_guadalupe') & (df_sw.excess_p1 > 0.0)]
    ci_gt_0 = len(g_hot[g_hot.gain_vs_l3_ci_low > 0.0])
    total_hot = len(g_hot)
    g_15 = g_hot[g_hot.excess_p1 == 0.15]
    point_gt_0_15 = len(g_15[g_15.gain_vs_l3_mean > 0.0])
    total_15 = len(g_15)

    if f"{ci_gt_0} of {total_hot} hot cells" not in readme_content:
        return False, f"Count '{ci_gt_0} of {total_hot} hot cells' not found in README."
    if f"{point_gt_0_15} of {total_15} cells at 15% excess" not in readme_content:
        return False, f"Count '{point_gt_0_15} of {total_15} cells at 15% excess' not found in README."

    # 3. Verify real hardware p1 measurements
    p1_files = glob.glob(p1_pattern)
    if not p1_files:
        return False, f"No p1 JSON file matching {p1_pattern} found."
    with open(p1_files[0], "r") as f:
        p1_data = json.load(f)
    p1_est = p1_data.get("p1_estimate", {})
    ro_err = p1_data.get("target_readout_error", {})

    hot_10 = []
    ro_fail = []
    for q_str, p in p1_est.items():
        q = int(q_str)
        ro = ro_err.get(q_str, 0.0)
        exc = p - ro
        if exc > 0.10:
            hot_10.append(q)
        if exc > 0.01 and ro > 0.20:
            ro_fail.append(q)

    count_10 = len(hot_10)
    total_q = len(p1_est)
    if f"{count_10} of {total_q} qubits" not in readme_content:
        return False, f"Quoted count '{count_10} of {total_q} qubits' not found in README."
    for q in hot_10:
        if str(q) not in readme_content:
            return False, f"Expected hot qubit {q} not mentioned in README."
    for q in ro_fail:
        if str(q) not in readme_content:
            return False, f"Expected readout-failure qubit {q} not mentioned in README."

    # 4. Verify first crossings and quoted gains for Guadalupe
    for b in ['ghz', 'bv', 'qft', 'mirror']:
        for k in [1, 2, 3]:
            sub = df_sw[(df_sw.device == 'fake_guadalupe') & (df_sw.benchmark == b) & (df_sw.hot_qubit_count == k)]
            prac = sub[(sub.paired_gain_mean >= 0.005) & (sub.paired_gain_ci_low > 0.0)]
            if not prac.empty:
                min_exc = prac.excess_p1.min()
                row = prac[prac.excess_p1 == min_exc].iloc[0]
                expected_gain = row.paired_gain_mean * 100.0
                gain_str = f"+{expected_gain:.2f}"
                if gain_str not in readme_content and f"{expected_gain:+.2f}" not in readme_content:
                    return False, f"Expected first-crossing gain {gain_str} for {b} hot={k} not found in README."

    # Check mirror 3 hot loss of effect at 0.05
    m3_05 = df_sw[(df_sw.device == 'fake_guadalupe') & (df_sw.benchmark == 'mirror') & (df_sw.hot_qubit_count == 3) & (df_sw.excess_p1 == 0.05)].iloc[0]
    m3_loss_gain = f"{m3_05.paired_gain_mean * 100.0:.2f}" # -0.12
    if m3_loss_gain not in readme_content:
        return False, f"Expected mirror (3 hot) effect loss gain {m3_loss_gain} at 0.05 not found in README."

    return True, "Sweep and p1 numbers verified successfully."

def audit_readme(readme_path="README.md", csv_path="results/hardware/runtime_table.csv", sweep_csv="results/sim/thermal_sweep.csv"):
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

    # 4. Extended sweep and p1 claims verification
    sw_ok, sw_msg = audit_sweep_and_p1(content, sweep_csv=sweep_csv)
    if not sw_ok:
        print(f"FAILED: {sw_msg}")
        return False, sw_msg

    print("Claim audit passed: all hardware and sweep numbers in README.md match ground truth within tolerance.")
    return True, "All claims verified"

if __name__ == "__main__":
    success, msg = audit_readme()
    if not success:
        sys.exit(1)
    sys.exit(0)
