import os
import sys
import yaml
import subprocess

def audit():
    errors = []
    
    # 1. 0 byte files
    for root, dirs, files in os.walk('.'):
        if '.venv' in dirs: dirs.remove('.venv')
        if '.git' in dirs: dirs.remove('.git')
        if not root.startswith(('.\\src', '.\\scripts', '.\\tests')): continue
        for f in files:
            if f == '__init__.py': continue
            path = os.path.join(root, f)
            if os.path.getsize(path) == 0:
                errors.append(f"0 byte file: {path}")
                
    # 2. Required symbols (Signature checks)
    def check_symbol(file_path, symbols):
        if not os.path.exists(file_path): return
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for s in symbols:
                if s not in content:
                    errors.append(f"Missing symbol '{s}' in {file_path}")
                    
    check_symbol('src/metrics/esp.py', ['def esp_standard', 'def esp_thermal', 'def esp_thermal_gate', 'def compute_esp_base', 'temps_mk: dict = None', 'p1: dict = None', 'thermal_term_mode'])
    check_symbol('src/noise_model/thermal.py', ['ReadoutError', 'depolarizing_error', 'reset'])
    check_symbol('src/runner_ibm.py', ['ibm_quantum_platform'])
    check_symbol('scripts/run_hardware.py', ['--execute', '--approve-seconds'])
    check_symbol('src/mappers/mappers.py', ['def mapper_qiskit_default', 'def mapper_random', 'def mapper_esp', 'def _get_layout'])
    
    # 3. tests/test_all.py
    if os.path.exists('tests/test_all.py'):
        with open('tests/test_all.py', 'r', encoding='utf-8') as f:
            if f.read().count('def test_') < 10:
                errors.append("tests/test_all.py has fewer than 10 test functions")
                
    # 4. config.yaml
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r', encoding='utf-8') as f:
            content = f.read()
            try:
                conf = yaml.safe_load(content)
                if conf.get('experiment', {}).get('budget_cap_qpu_seconds') != 600:
                    errors.append("budget_cap_qpu_seconds != 600")
            except:
                pass
                
    # 5. requirements.txt UTF-8
    try:
        with open('requirements.txt', 'r', encoding='utf-8') as f: f.read()
    except UnicodeDecodeError:
        errors.append("requirements.txt is not valid UTF-8")
        
    # 6. README.md size
    if os.path.exists('README.md') and os.path.getsize('README.md') < 2000:
        errors.append("README.md is under 2000 bytes")
        
    # 7. .env tracked
    res = subprocess.run(['git', 'ls-files', '--error-unmatch', '.env'], capture_output=True)
    if res.returncode == 0:
        errors.append(".env is tracked by git")
        
    if errors:
        print("Audit failed:")
        for e in errors: print(f" - {e}")
        sys.exit(1)
    else:
        print("Audit passed.")
        sys.exit(0)

if __name__ == '__main__':
    audit()
