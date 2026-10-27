#!/usr/bin/env python3
"""Show full-decoder build status, separately from the verified hybrid model."""
import json
import datetime
import re
import sys
from zoneinfo import ZoneInfo
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "scripts"))
from auto_deploy import log_tail, process_identity

path = PROJECT / "reports/status.json"
if not path.is_file():
    raise SystemExit("Complete-decoder deployment has not started")
state = json.loads(path.read_text())
alive = state.get("process_identity") is not None and process_identity(state["pid"]) == state["process_identity"]
print("Task: Qwen3-0.6B optimized complete FPGA decoder (28 layers)")
print("Verified baseline retained:", PROJECT.parent / "full-decode")
print("Status:", "stopped (stale running status)" if state["status"] == "running" and not alive else state["status"])
print("Stage:", state["stage"])
print("Updated:", state["updated_at"])
print("Software full decoder verified:", state.get("software_full_decoder_verified", False))
print("FPGA deployment complete:", state["fpga_deployment_complete"])
print("Full decoder on FPGA verified:", state.get("full_decoder_on_fpga", False))
if "generated_ids_match_reference" in state:
    print("Output tokens match CPU reference:", state["generated_ids_match_reference"])
if state.get("decode_speedup"):
    print("Measured decode speedup:", state["decode_speedup"])
    print("Measured tokens/second:", state["tokens_per_second"])
if state.get("current_log"):
    print("Log:", state["current_log"])
    log_path = Path(state["current_log"])
    if log_path.is_file():
        print("Log updated:", datetime.datetime.fromtimestamp(log_path.stat().st_mtime,
            ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"))
    lines = log_tail(state["current_log"], 65536).replace("\r", "\n").splitlines()
    relevant = [s.strip() for s in lines if re.search(
        r"Step |synthesis in progress|Phase \d|Starting bitstream|Finished \d|full prefill|full decode|Numerical check|ERROR:|FINISHED", s)]
    if relevant:
        print("Latest progress:", relevant[-1])
runs = PROJECT / "_x/link/vivado/vpl/prj/prj.runs"
subruns = list(runs.glob("*_synth_1"))
if subruns:
    failed = [p.name for p in subruns if (p / ".vivado.error.rst").exists()]
    passed = sum((p / ".vivado.end.rst").exists() and not (p / ".vivado.error.rst").exists() for p in subruns)
    print(f"Synthesis subruns: passed={passed}, failed={len(failed)}, total={len(subruns)}")
    if failed:
        print("Failed subruns:", ", ".join(failed))
    activity = list(runs.glob("*/runme.log"))
    if activity:
        latest = max(activity, key=lambda p: p.stat().st_mtime)
        print("Most recent subrun:", latest.parent.name)
        phases = [s.strip() for s in log_tail(latest, 65536).splitlines()
                  if re.search(r"^Phase \d|^Starting |^Ending |^ERROR:|^Command:|^Running DRC|^INFO:.*DRC finished", s)]
        if phases:
            print("Subrun progress:", phases[-1])
if state.get("error"):
    print("Error:", state["error"])
