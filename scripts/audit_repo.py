import os
import subprocess
import yaml

def run_audit():
    errors = []
    print("Starting Audit...")
    
    # 1. py_compile every .py file
    print("Checking syntax of all .py files...")
    py_files = []
    for root, dirs, files in os.walk('.'):
        if '.venv' in root or '.git' in root or '__pycache__' in root or '.pytest_cache' in root:
             continue
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.join(root, f))
    for f in py_files:
        res = subprocess.run(['python', '-m', 'py_compile', f], capture_output=True)
        if res.returncode != 0:
            errors.append(f"Syntax error in {f}: {res.stderr.decode()}")
            
    # 2. pytest collects >= 13 tests
    print("Checking pytest collection...")
    res = subprocess.run(['python', '-m', 'pytest', '--collect-only', '-q'], capture_output=True, text=True)
    if res.returncode != 0 and res.returncode != 5:
        errors.append("Pytest failed to run.")
    else:
        out = res.stdout
        import re
        m = re.search(r'(\d+)\s+test[s]?\s+collected', out)
        if not m:
             lines = out.splitlines()
             collected = sum(1 for line in lines if "::" in line)
        else:
             collected = int(m.group(1))
        print(f"  -> Pytest collected {collected} tests")
        if collected < 13:
             errors.append(f"Pytest collected {collected} tests, expected >= 13.")
                 
    # 3. run_hardware.py --help lists --pilot and --approve-seconds
    print("Checking run_hardware.py args...")
    res = subprocess.run(['python', 'scripts/run_hardware.py', '--help'], capture_output=True, text=True)
    flags_found = []
    if '--pilot' in res.stdout: flags_found.append('--pilot')
    else: errors.append("run_hardware.py --help missing --pilot")
    if '--approve-seconds' in res.stdout: flags_found.append('--approve-seconds')
    else: errors.append("run_hardware.py --help missing --approve-seconds")
    print(f"  -> Flags found: {', '.join(flags_found)}")
         
    # 4. config values
    print("Checking config values...")
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            exp = config.get('experiment', {})
            cap = exp.get('budget_cap_qpu_seconds')
            res_val = exp.get('reserve_qpu_seconds')
            print(f"  -> budget_cap_qpu_seconds: {cap}")
            print(f"  -> reserve_qpu_seconds: {res_val}")
            
            if cap != 350:
                errors.append(f"config budget_cap_qpu_seconds is {cap} != 350")
            if res_val != 250:
                errors.append(f"config reserve_qpu_seconds is {res_val} != 250")
            if 'thermal_term_mode' not in exp:
                errors.append("config missing thermal_term_mode")
            if 'backend_selection' not in config.get('backends', {}):
                errors.append("config missing backend_selection")
            if 'ibm_kyoto' in str(config):
                errors.append("config still contains ibm_kyoto")
            if 'hot_qubits' in str(config):
                errors.append("config still contains hot_qubits")
    except Exception as e:
        errors.append(f"Config check failed: {e}")
        
    # 5. README >= 2000 bytes
    print("Checking README size...")
    if os.path.exists('README.md'):
        size = os.path.getsize('README.md')
        print(f"  -> README.md size: {size} bytes")
        if size < 2000:
            errors.append(f"README.md is {size} < 2000 bytes")
    else:
        errors.append("README.md not found")
        
    # 6. .env untracked
    print("Checking .env untracked...")
    res = subprocess.run(['git', 'ls-files', '.env'], capture_output=True, text=True)
    if '.env' in res.stdout:
        errors.append(".env is tracked by git")
        
    # 7. no file in src/scripts/tests is 0 bytes
    print("Checking for 0 byte files in src, scripts, tests...")
    for d in ['src', 'scripts', 'tests']:
         if os.path.exists(d):
             for root, _, files in os.walk(d):
                 for f in files:
                     path = os.path.join(root, f)
                     if os.path.getsize(path) == 0:
                          if f != '__init__.py': # sometimes __init__ is 0 bytes, but prompt says "no file"
                              errors.append(f"File {path} is 0 bytes")

    if errors:
         print("\nAudit Failed with errors:")
         for e in errors:
             print(f" - {e}")
         sys.exit(1)
    else:
         print("\nAudit Passed Successfully!")
         
if __name__ == '__main__':
    import sys
    run_audit()
