import json
import time
import datetime
from pathlib import Path

def parse_iso_ts(ts_str):
    try:
        return datetime.datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

def get_live_pilot_scale(pilot_path, expected_backend, default_scale=3.0/8192, shots=8192):
    p = Path(pilot_path)
    if not p.exists():
        return default_scale, 3.0
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        j_id = str(d.get("job_id", "")).strip()
        is_loc = d.get("is_local", False) or d.get("target") in ["fake", "local"]
        b_name = d.get("backend", "")
        cal_ts = d.get("calibration_timestamp")
        age_s = abs(time.time() - float(d.get("timestamp", 0.0)))
        if not j_id.startswith("pilot_local") and not is_loc and b_name == expected_backend and cal_ts and age_s <= 86400:
            scale = float(d.get("qpu_seconds_per_shot", d.get("seconds_per_shot", default_scale)))
            usage = float(d.get("pilot_usage", scale * shots))
            return scale, usage
    except Exception:
        pass
    return default_scale, 3.0

def compute_z4_cumulative_budget(config, persistent_count=None, live_pilot_marrakesh=None, live_pilot_fez=None):
    allowance = float(config.get("allowance_remaining_s", 242.0))
    reserve = float(config.get("reserve_s", 150.0))
    spendable = allowance - reserve

    is_stale = False
    checked_at_str = config.get("allowance_checked_at")
    if not checked_at_str:
        is_stale = True
    else:
        checked_at_dt = parse_iso_ts(checked_at_str)
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        age_sec = abs((now_utc - checked_at_dt).total_seconds())
        if age_sec > 86400:
            is_stale = True

    # Marrakesh pilot scale (3.0 s / PUB default until live pilot exists)
    if live_pilot_marrakesh is not None:
        sec_per_pub_m = live_pilot_marrakesh
    else:
        _, usage_m = get_live_pilot_scale("results/hardware_extra/pilot_scale.json", "ibm_marrakesh")
        sec_per_pub_m = usage_m

    # Fez pilot scale (3.0 s / PUB default until live pilot exists)
    if live_pilot_fez is not None:
        sec_per_pub_f = live_pilot_fez
    else:
        _, usage_f = get_live_pilot_scale("results/hardware_extra/pilot_scale_ibm_fez.json", "ibm_fez")
        sec_per_pub_f = usage_f

    stopping_rule_fired = (persistent_count is not None and persistent_count < 1)

    steps = [
        {"name": "Pilot Marrakesh", "backend": "ibm_marrakesh", "pubs": 1, "sec_pub": sec_per_pub_m, "est_sec": 1 * sec_per_pub_m, "is_step3": False},
        {"name": "p1v2 Marrakesh", "backend": "ibm_marrakesh", "pubs": 24, "sec_pub": sec_per_pub_m, "est_sec": 24 * sec_per_pub_m, "is_step3": False},
        {"name": "Plan Marrakesh", "backend": "ibm_marrakesh", "pubs": 12, "sec_pub": sec_per_pub_m, "est_sec": 12 * sec_per_pub_m, "is_step3": True},
        {"name": "Pilot Fez", "backend": "ibm_fez", "pubs": 1, "sec_pub": sec_per_pub_f, "est_sec": 1 * sec_per_pub_f, "is_step3": False},
        {"name": "Fez survey", "backend": "ibm_fez", "pubs": 4, "sec_pub": sec_per_pub_f, "est_sec": 4 * sec_per_pub_f, "is_step3": False},
        {"name": "Fez confirm", "backend": "ibm_fez", "pubs": 16, "sec_pub": sec_per_pub_f, "est_sec": 16 * sec_per_pub_f, "is_step3": False},
    ]

    cumul = 0.0
    table_rows = []
    for s in steps:
        if s["is_step3"] and stopping_rule_fired:
            row = {
                "step": s["name"],
                "backend": s["backend"],
                "pubs": s["pubs"],
                "est_sec": 0.0,
                "cumul_sec": cumul,
                "budget_spendable": spendable,
                "status": "DROPPED (stopping rule)",
                "note": "persistent_count < 1 (saved 36.0 s)"
            }
        else:
            cumul += s["est_sec"]
            if is_stale:
                status = "NO-GO (STALE)"
                note = "stale allowance (>24h)"
            elif cumul <= spendable:
                status = "GO"
                note = f"cumul {cumul:.1f} s <= spendable {spendable:.1f} s"
            else:
                status = "NO-GO"
                note = f"cumul {cumul:.1f} s > spendable {spendable:.1f} s"
            row = {
                "step": s["name"],
                "backend": s["backend"],
                "pubs": s["pubs"],
                "est_sec": s["est_sec"],
                "cumul_sec": cumul,
                "budget_spendable": spendable,
                "status": status,
                "note": note
            }
        table_rows.append(row)

    return {
        "allowance_remaining_s": allowance,
        "reserve_s": reserve,
        "spendable_s": spendable,
        "allowance_checked_at": checked_at_str,
        "is_allowance_stale": is_stale,
        "stopping_rule_fired": stopping_rule_fired,
        "rows": table_rows
    }

def format_z4_budget_table(budget_data):
    lines = []
    allowance = budget_data["allowance_remaining_s"]
    reserve = budget_data["reserve_s"]
    spendable = budget_data["spendable_s"]
    checked_at = budget_data.get("allowance_checked_at", "unknown")
    stale_tag = " [WARNING: >24h STALE]" if budget_data["is_allowance_stale"] else " [FRESH]"

    lines.append("=" * 115)
    lines.append("SHARED ACCOUNT CUMULATIVE BUDGET PRIORITY TABLE (CHECKPOINT Z4)")
    lines.append(f"Allowance Remaining: {allowance:.1f} s | Reserve: {reserve:.1f} s | Spendable Allowance: {spendable:.1f} s")
    lines.append(f"Allowance Checked:   {checked_at}{stale_tag}")
    lines.append("=" * 115)
    hdr = f"{'Line':<4} | {'Step':<24} | {'Backend':<14} | {'PUBs':<4} | {'Est s':<8} | {'Cumul s':<8} | {'Budget (spendable)':<18} | {'Status (GO / NO-GO)'}"
    lines.append(hdr)
    lines.append("-" * 115)

    for idx, r in enumerate(budget_data["rows"], 1):
        est_s = f"{r['est_sec']:.1f} s" if r["est_sec"] > 0 else "0.0 s"
        cum_s = f"{r['cumul_sec']:.1f} s"
        b_sp = f"{r['budget_spendable']:.1f} s"
        lines.append(
            f"{idx:<4} | {r['step']:<24} | {r['backend']:<14} | {r['pubs']:<4} | {est_s:<8} | {cum_s:<8} | {b_sp:<18} | {r['status']}"
        )
    lines.append("-" * 115)
    return "\n".join(lines)

