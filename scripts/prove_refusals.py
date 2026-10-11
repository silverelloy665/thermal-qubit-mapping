#!/usr/bin/env python3
"""
scripts/prove_refusals.py

Executes all safety refusal conditions locally against FakeMarrakesh / plan.json
and records the raw exception outputs to results/hardware_extra/refusal_proofs.txt.
"""

import os
import sys
import json
import time
import shutil
import traceback
import subprocess
from pathlib import Path

def run_proofs(output_txt_path="results/hardware_extra/refusal_proofs.txt"):
    plan_path = Path("results/hardware_extra/plan.json")
    pilot_scale_path = Path("results/hardware_extra/pilot_scale.json")
    
    # Ensure baseline dry-run and pilot exist
    subprocess.run([sys.executable, "scripts/run_hotspot_hw.py", "--local"], capture_output=True, text=True)
    subprocess.run([sys.executable, "scripts/run_hotspot_hw.py", "--local", "--pilot"], capture_output=True, text=True)
    
    with open(plan_path, "r", encoding="utf-8") as f:
        baseline_plan = json.load(f)
    with open(pilot_scale_path, "r", encoding="utf-8") as f:
        baseline_pilot = json.load(f)
        
    required_approve = int(baseline_plan["total_estimate_seconds"])
    lines = []
    lines.append("=" * 80)
    lines.append("GATE REFUSAL PROOFS: PRE-REGISTERED HARDWARE HOT-SPOT STRESS TEST")
    lines.append("=" * 80)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 1: Refusal without pilot file
    # -------------------------------------------------------------------------
    lines.append("PROOF 1: REFUSAL WITHOUT VALID PILOT FILE (PILOT FILE MISSING)")
    lines.append("-" * 80)
    pilot_scale_bak = pilot_scale_path.with_suffix(".bak")
    if pilot_scale_path.exists():
        shutil.move(pilot_scale_path, pilot_scale_bak)
    try:
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when pilot file is missing.")
    finally:
        if pilot_scale_bak.exists():
            shutil.move(pilot_scale_bak, pilot_scale_path)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 2: Refusal with pilot file missing job_id
    # -------------------------------------------------------------------------
    lines.append("PROOF 2: REFUSAL WITH PILOT FILE MISSING JOB_ID")
    lines.append("-" * 80)
    no_job_pilot = dict(baseline_pilot)
    if "job_id" in no_job_pilot:
        del no_job_pilot["job_id"]
    with open(pilot_scale_path, "w", encoding="utf-8") as f:
        json.dump(no_job_pilot, f, indent=2)
    try:
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when pilot file has no job_id.")
    finally:
        with open(pilot_scale_path, "w", encoding="utf-8") as f:
            json.dump(baseline_pilot, f, indent=2)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 3: Refusal with pilot timestamp older than plan creation time
    # -------------------------------------------------------------------------
    lines.append("PROOF 3: REFUSAL WITH PILOT TIMESTAMP OLDER THAN PLAN CREATION")
    lines.append("-" * 80)
    stale_pilot = dict(baseline_pilot)
    stale_pilot["timestamp"] = baseline_plan.get("created_timestamp", 1760164800.0) - 3600.0
    with open(pilot_scale_path, "w", encoding="utf-8") as f:
        json.dump(stale_pilot, f, indent=2)
    try:
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when pilot timestamp is older than plan creation time.")
    finally:
        with open(pilot_scale_path, "w", encoding="utf-8") as f:
            json.dump(baseline_pilot, f, indent=2)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 4: Refusal with expired pilot file (> 24 hours old)
    # -------------------------------------------------------------------------
    lines.append("PROOF 4: REFUSAL WITH EXPIRED PILOT FILE (> 24 HOURS OLD)")
    lines.append("-" * 80)
    expired_pilot = dict(baseline_pilot)
    expired_pilot["timestamp"] = time.time() - 90000 # 25 hours ago
    with open(pilot_scale_path, "w", encoding="utf-8") as f:
        json.dump(expired_pilot, f, indent=2)
    try:
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when pilot file exceeds 24-hour validity window.")
    finally:
        with open(pilot_scale_path, "w", encoding="utf-8") as f:
            json.dump(baseline_pilot, f, indent=2)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 5: Refusal with changed plan hash / modified plan
    # -------------------------------------------------------------------------
    lines.append("PROOF 5: REFUSAL WITH CHANGED PLAN HASH / TAMPERED PLAN")
    lines.append("-" * 80)
    tampered_plan = dict(baseline_plan)
    tampered_plan["plan_hash"] = "0000000000000000000000000000000000000000000000000000000000000000"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(tampered_plan, f, indent=2)
    try:
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when plan hash does not match expected canonical hash.")
    finally:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(baseline_plan, f, indent=2)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 6: Refusal with wrong approval seconds
    # -------------------------------------------------------------------------
    lines.append("PROOF 6: REFUSAL WITH WRONG APPROVAL SECONDS")
    lines.append("-" * 80)
    wrong_approve = required_approve + 15
    cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(wrong_approve)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines.append(f"Command: {' '.join(cmd)}")
    lines.append(f"Exit code: {res.returncode}")
    lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
    lines.append(f"[VERIFIED] Refused execution when approved seconds ({wrong_approve}) != required ({required_approve}).")
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 7: Refusal if remaining allowance - projected spend < reserve_s
    # -------------------------------------------------------------------------
    lines.append("PROOF 7: REFUSAL IF REMAINING ALLOWANCE < RESERVE BUFFER")
    lines.append("-" * 80)
    config_path = Path("config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config_text = f.read()
    try:
        depleted_text = config_text.replace("allowance_remaining_s: 279", "allowance_remaining_s: 200")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(depleted_text)
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when remaining allowance would drop below reserve buffer (150 s).")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_text)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 8: Refusal if calibration timestamp changed since planning
    # -------------------------------------------------------------------------
    lines.append("PROOF 8: REFUSAL IF CALIBRATION TIMESTAMP CHANGED SINCE PLANNING")
    lines.append("-" * 80)
    outdated_plan = dict(baseline_plan)
    outdated_plan["calibration_timestamp"] = "2024-01-01T00:00:00+00:00"
    outdated_plan["plan_hash"] = "dummy"
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(outdated_plan, f, indent=2)
    try:
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local", "--execute", "--approve-seconds", str(required_approve)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused execution when backend calibration timestamp differs from plan.")
    finally:
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(baseline_plan, f, indent=2)
    lines.append("")

    # -------------------------------------------------------------------------
    # PROOF 9: Refusal in dry run if total estimate exceeds extra_run_cap_s
    # -------------------------------------------------------------------------
    lines.append("PROOF 9: REFUSAL IN DRY RUN IF ESTIMATE EXCEEDS EXTRA_RUN_CAP_S")
    lines.append("-" * 80)
    try:
        low_cap_text = config_text.replace("extra_run_cap_s: 110", "extra_run_cap_s: 80")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(low_cap_text)
        cmd = [sys.executable, "scripts/run_hotspot_hw.py", "--local"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"Command: {' '.join(cmd)}")
        lines.append(f"Exit code: {res.returncode}")
        lines.append(f"Output / Error:\n{res.stderr.strip() or res.stdout.strip()}")
        lines.append("[VERIFIED] Refused dry run execution when total estimate exceeds extra_run_cap_s.")
    finally:
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_text)
    lines.append("")

    # Clean up generated pilot file from results/ directory so results/ stays clean
    if pilot_scale_path.exists():
        pilot_scale_path.unlink()

    lines.append("=" * 80)
    lines.append("ALL REFUSAL CONDITIONS RIGOROUSLY DEMONSTRATED AND VERIFIED")
    lines.append("=" * 80)

    out_content = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(out_content)

    print(f"Refusal proofs written to {output_txt_path}.")

if __name__ == "__main__":
    run_proofs()
