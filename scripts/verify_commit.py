import os
import sys
import subprocess

def run_cmd(cmd):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return res.returncode, res.stdout.strip(), res.stderr.strip()

def main():
    allowed_shrink = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # a) fail if git status --short is non-empty
    rc, out, err = run_cmd("git status --short")
    if out:
        print(f"FAIL: git status --short is non-empty:\n{out}")
        sys.exit(1)
        
    # b) fail if git diff HEAD is non-empty
    rc, out, err = run_cmd("git diff HEAD")
    if out:
        print(f"FAIL: git diff HEAD is non-empty:\n{out}")
        sys.exit(1)
        
    # c) print git show --stat HEAD
    rc, out, err = run_cmd("git show --stat HEAD")
    print(out)
    
    # d) fail if any tracked text file in HEAD is >30% smaller than in HEAD~1
    rc, out, err = run_cmd("git diff --name-only HEAD~1 HEAD")
    if rc == 0 and out:
        changed_files = out.split('\n')
        for f in changed_files:
            if not f: continue
            if f in allowed_shrink: continue
            
            # Check if it's a text file (just assume .py, .md, .yaml, .txt for simplicity)
            if not any(f.endswith(ext) for ext in ['.py', '.md', '.yaml', '.txt', '.csv']):
                 continue
                 
            # get size in HEAD~1
            rc1, out1, _ = run_cmd(f"git cat-file -s HEAD~1:{f}")
            # get size in HEAD
            rc2, out2, _ = run_cmd(f"git cat-file -s HEAD:{f}")
            
            if rc1 == 0 and rc2 == 0:
                s1, s2 = int(out1), int(out2)
                if s1 > 0 and (s1 - s2) / s1 > 0.3:
                    print(f"FAIL: {f} shrank by more than 30% (from {s1} to {s2})")
                    sys.exit(1)
                    
    # e) fail if test count fell below 13
    rc, out, err = run_cmd("python -m pytest --collect-only -q")
    import re
    m = re.search(r'(\d+)\s+test[s]?\s+collected', out)
    if not m:
         lines = out.splitlines()
         collected = sum(1 for line in lines if "::" in line)
    else:
         collected = int(m.group(1))
    if collected < 13:
         print(f"FAIL: test count fell to {collected} (< 13)")
         sys.exit(1)
         
    print("VERIFY PASSED")

if __name__ == "__main__":
    main()
