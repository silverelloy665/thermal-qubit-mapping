import os
import sys

def check_secrets():
    secrets = ['QISKIT_IBM_TOKEN', 'token', 'apikey', 'api_key', 'ibm_cloud']
    found = False
    for root, dirs, files in os.walk('.'):
        if '.venv' in dirs: dirs.remove('.venv')
        if '.git' in dirs: dirs.remove('.git')
        for file in files:
            if file.endswith('.py') or file.endswith('.yaml') or file == '.env':
                filepath = os.path.join(root, file)
                if filepath.endswith('.env.example') or filepath.endswith('.env') or file == 'check_secrets.py' or file == 'run_hardware.py': continue
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for i, line in enumerate(lines):
                            line_lower = line.lower()
                            if any(s in line_lower for s in secrets) and '=' in line and len(line.split('=')[1].strip()) > 10:
                                print(f"WARNING: Potential secret found in {filepath} at line {i+1}")
                                found = True
                except:
                    pass
    if found:
        print("Secrets check failed.")
        sys.exit(1)
    else:
        print("No hardcoded secrets found.")
        sys.exit(0)

if __name__ == '__main__':
    check_secrets()
