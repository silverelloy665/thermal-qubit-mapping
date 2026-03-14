#!/usr/bin/env python3
"""Fix Unicode characters in source files."""

import os

# Files to fix
files = [
    'main.py',
    'src/evaluator.py',
]

for fname in files:
    fpath = os.path.join(os.path.dirname(__file__), fname)
    if os.path.exists(fpath):
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace Unicode  
        original_len = len(content)
        content = content.replace('✓', '[OK]').replace('✗', '[ERROR]')
        
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        changed = len([c for c in content if c in '✓✗'])
        print(f"Fixed {fname}: {changed} Unicode chars replaced")
    else:
        print(f"File not found: {fpath}")

print("Done!")
