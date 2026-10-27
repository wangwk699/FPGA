#!/usr/bin/env python3
"""Print the local background deployment status without network access."""
import json
import datetime
import re
import runpy
import sys
from pathlib import Path
from zoneinfo import ZoneInfo
from auto_deploy import process_identity, log_tail

root = Path(__file__).resolve().parents[1]
if '--model' in sys.argv:
    runpy.run_path(str(root/'full-decode-stream/scripts/status.py'),run_name='__main__')
    raise SystemExit(0)
if not any(flag in sys.argv for flag in ["--hybrid", "--baseline", "--previous"]) and (root / "full-decode-stream/reports/status.json").is_file():
    runpy.run_path(str(root / "full-decode-stream/scripts/status.py"), run_name="__main__")
    raise SystemExit(0)
if "--hybrid" not in sys.argv and "--baseline" not in sys.argv and (root / "full-decode-fast/reports/status.json").is_file():
    runpy.run_path(str(root / "full-decode-fast/scripts/status.py"), run_name="__main__")
    raise SystemExit(0)
if "--hybrid" not in sys.argv and (root / "full-decode/reports/status.json").is_file():
    runpy.run_path(str(root / "full-decode/scripts/status.py"), run_name="__main__")
    raise SystemExit(0)
path = root / "reports/auto-deploy-status.json"
if not path.is_file():
    raise SystemExit("No automatic deployment worker status file exists")
state = json.loads(path.read_text())
alive = process_identity(state["pid"]) == state.get("process_identity") and state.get("process_identity") is not None
print("Status:", "stopped (stale running status)" if state["status"] == "running" and not alive else state["status"])
print("Stage:", state["stage"])
if state.get("install_action") and state["stage"] in ["installing_tools", "retrying_install"]:
    print("Installation action:", state["install_action"])
print("Updated:", state["updated_at"])
print("FPGA deployment complete:", state["fpga_deployment_complete"])
if state.get("scope"):
    print("Scope:", state["scope"])
if "full_decoder_on_fpga" in state:
    print("Full decoder on FPGA:", state["full_decoder_on_fpga"])
if "generated_ids_match_reference" in state:
    print("Output tokens match CPU reference:", state["generated_ids_match_reference"])
if state.get("download_progress"):
    print("Last download progress:", state["download_progress"])
if state.get("install_attempt"):
    print("Installation attempt:", state["install_attempt"], "/", state["install_max_attempts"])
if state.get("install_failure"):
    print("Installer diagnosis:", state["install_failure"]["summary"])
if state.get("current_log"):
    print("Log:", state["current_log"])
    log = Path(state["current_log"])
    if log.is_file():
        modified = datetime.datetime.fromtimestamp(log.stat().st_mtime, ZoneInfo("Asia/Shanghai"))
        print("Log updated:", modified.isoformat(timespec="seconds"))
        if state["stage"] == "build_kernel":
            lines = log_tail(log, limit=65536).replace("\r", "\n").splitlines()
            progress = [line.strip() for line in lines if re.search(
                r"Block-level synthesis in progress|Run vpl: Step |Run vpl: FINISHED|Phase \d+|Run run_link: Step ", line)]
            if progress:
                print("Latest build progress:", progress[-1])
            runs = root / "_x/link/vivado/vpl/prj/prj.runs"
            failures = sorted(p.parent.name for p in runs.glob("*/.vivado.error.rst"))
            synthesis = [p for p in runs.glob("*_synth_1") if p.name != "my_rm_synth_1"]
            if synthesis:
                completed = sum((p / ".vivado.end.rst").is_file() and not (p / ".vivado.error.rst").is_file()
                                for p in synthesis)
                active = sum((p / ".vivado.begin.rst").is_file()
                             and not (p / ".vivado.end.rst").is_file()
                             and not (p / ".vivado.error.rst").is_file() for p in synthesis)
                sub_failures = sum((p / ".vivado.error.rst").is_file() for p in synthesis)
                print(f"Synthesis subruns: passed={completed}, failed={sub_failures}, active={active}, total={len(synthesis)}")
                activity = [p for p in runs.glob("*/runme.log") if p.is_file()]
                if activity:
                    latest = max(activity, key=lambda p: p.stat().st_mtime)
                    time = datetime.datetime.fromtimestamp(latest.stat().st_mtime, ZoneInfo("Asia/Shanghai"))
                    print("Latest subrun activity:", time.isoformat(timespec="seconds"), latest.parent.name)
                    if latest.parent.name in ["my_rm_synth_1", "impl_1"]:
                        phases = [line.strip() for line in log_tail(latest, limit=65536).splitlines()
                                  if re.search(r"^Phase \d+|^Starting |^Ending |^ERROR:", line)]
                        if phases:
                            print("Latest implementation detail:", phases[-1])
            if failures:
                print("Failed build subruns:", len(failures))
                for name in failures:
                    print("  ", name)
        elif state["stage"] == "model_fpga_inference":
            lines = log_tail(log, limit=65536).replace("\r", "\n").splitlines()
            progress = [line.strip() for line in lines if line.startswith((
                "Local model loaded;", "Preparing fpga runtime", "Registered ", "Native decode step "))]
            if progress:
                print("Latest inference progress:", progress[-1])
if state.get("error"):
    print("Error:", state["error"])
