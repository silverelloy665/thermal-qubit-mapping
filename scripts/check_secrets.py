import os
import sys
import re
import subprocess

def check_secrets():
    errors = []
    
    # 1. Regex checks
    token_re = re.compile(r'(?<!def )(?<!class )[A-Za-z0-9]{40,}')
    qiskit_re = re.compile(r'QISKIT_IBM_TOKEN\s*=\s*([A-Za-z0-9_-]+)')
    
    exts = ('.py', '.yaml', '.md', '.json', '.csv', '.ipynb', '.txt', '.toml')
    
    for root, dirs, files in os.walk('.'):
        if '.venv' in dirs: dirs.remove('.venv')
        if '.git' in dirs: dirs.remove('.git')
        
        for f in files:
            if f.endswith(exts):
                filepath = os.path.join(root, f)
                if 'check_secrets.py' in filepath: continue
                try:
                    with open(filepath, 'r', encoding='utf-8') as file:
                        for i, line in enumerate(file):
                            if token_re.search(line):
                                errors.append(f"Token-like string in {filepath}:{i+1}")
                            m = qiskit_re.search(line)
                            if m and len(m.group(1).strip()) > 0:
                                errors.append(f"QISKIT_IBM_TOKEN set in {filepath}:{i+1}")
                except Exception:
                    pass
                    
    # 2. Check .env is untracked
    res = subprocess.run(['git', 'ls-files', '--error-unmatch', '.env'], capture_output=True)
    if res.returncode == 0:
        errors.append(".env is tracked by git")
        
    # 3. Check .env not in history
    res2 = subprocess.run(['git', 'log', '--all', '--', '.env'], capture_output=True, text=True)
    if res2.stdout.strip():
        errors.append(".env found in git history")
        
    if errors:
        print("Secrets check failed:")
        for e in errors: print(f" - {e}")
        sys.exit(1)
    else:
        print("No secrets found. .env is safe.")
        sys.exit(0)

if __name__ == '__main__':
    check_secrets()
