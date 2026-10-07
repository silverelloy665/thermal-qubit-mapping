import os
import sys
import json
import glob
import subprocess
from datetime import datetime
from pathlib import Path

def get_git_info():
    info = {"commit": "unknown", "branch": "unknown", "dirty": False}
    try:
        res = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True, check=True)
        info["commit"] = res.stdout.strip()
    except Exception:
        pass
    try:
        res = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
        info["branch"] = res.stdout.strip()
    except Exception:
        pass
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
        info["dirty"] = bool(res.stdout.strip())
    except Exception:
        pass
    return info

def get_package_versions():
    packages = ["qiskit", "qiskit_aer", "qiskit_ibm_runtime", "numpy", "scipy", "pandas"]
    versions = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not_installed"
        except Exception as e:
            versions[pkg] = f"error: {e}"
    versions["python"] = sys.version
    return versions

def write_provenance(csv_path: str | Path) -> Path:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Generate summary of CSV
    row_count = 0
    columns = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            columns = [c.strip() for c in header.split(",") if c.strip()]
            for _ in f:
                row_count += 1
    except Exception as e:
        columns = [f"error_reading_header: {e}"]

    sidecar_data = {
        "csv_file": csv_path.name,
        "csv_path": str(csv_path.resolve()),
        "generated_at": datetime.now().isoformat(),
        "git": get_git_info(),
        "package_versions": get_package_versions(),
        "summary": {
            "row_count": row_count,
            "column_count": len(columns),
            "columns": columns,
        }
    }

    sidecar_path = csv_path.with_suffix(".provenance.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar_data, f, indent=2)

    print(f"Wrote provenance sidecar: {sidecar_path}")
    return sidecar_path

def main():
    if len(sys.argv) > 1:
        targets = [Path(p) for p in sys.argv[1:]]
    else:
        # Search for CSVs in results directory excluding invalid and stale folders
        targets = []
        for root, dirs, files in os.walk("results"):
            if "invalid" in root or "stale" in root:
                continue
            for f in files:
                if f.endswith(".csv"):
                    targets.append(Path(root) / f)

    if not targets:
        print("No target CSV files found.")
        return

    for t in targets:
        if t.exists():
            write_provenance(t)

if __name__ == "__main__":
    main()

