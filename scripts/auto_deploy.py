#!/usr/bin/env python3
"""Supervise or attach to installation, then build and validate on U55C.

Writes local status/logs only. It does not reset the board, flash firmware, install
system packages or download a model. Every command must succeed before the next.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
STATE = REPORTS / "auto-deploy-status.json"


def timestamp():
    return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def process_identity(pid):
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        # The command name can contain spaces, so parse after the closing parenthesis.
        fields = text[text.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return fields[19]  # Linux field 22, process start time.
    except (OSError, IndexError):
        return None


def download_progress(path=None):
    path = Path(path) if path else REPORTS / "vitis-install-resume.log"
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - 16384))
            text = stream.read().decode(errors="replace").replace("\r", "\n")
        lines = [line.strip() for line in text.splitlines() if "Downloading files (" in line]
        # Installer messages can be appended to a progress update without a newline.
        return re.split(r"(?:INFO|WARN|ERROR)\s+-\s+", lines[-1], maxsplit=1)[0].strip() if lines else None
    except OSError:
        return None


def log_tail(path, limit=2 * 1024 * 1024):
    try:
        path = Path(path)
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def classify_install_failure(console, detail=""):
    """Return a secret-free diagnosis, never signed download URLs or credentials."""
    text = (console + "\n" + detail).lower()
    if any(value in text for value in ["must generate an authentication token", "authentication failed",
            "authentication token has expired", "authentication token is expired", "invalid authentication token"]):
        return {"kind": "authentication", "retryable": False,
                "summary": "AMD installer authentication is missing or expired; run scripts/setup_vitis.sh auth"}
    if any(value in text for value in ["no space left on device", "not enough disk space", "insufficient disk space"]):
        return {"kind": "disk_space", "retryable": False, "summary": "Not enough disk space for Vitis installation"}
    if any(value in text for value in ["outofmemoryerror", "cannot allocate memory"]):
        return {"kind": "memory", "retryable": False, "summary": "Vitis installer ran out of memory"}
    if any(value in text for value in ["sslhandshakeexception", "ssl peer shut down", "remote host terminated the handshake"]):
        return {"kind": "download_network", "retryable": True,
                "summary": "Vitis component download failed because a TLS/SSL connection was interrupted"}
    if "checksum failed" in text:
        return {"kind": "download_checksum", "retryable": True,
                "summary": "Vitis component download failed checksum validation; existing valid cached files are retained"}
    if any(value in text for value in ["error downloading", "download failed", "sockettimeoutexception",
                                      "connection reset", "read timed out"]):
        return {"kind": "download_network", "retryable": True,
                "summary": "Vitis component download failed; retrying can reuse existing downloaded files"}
    if "canceling" in text or "cancelled" in text or "canceled" in text:
        return {"kind": "cancelled", "retryable": False, "summary": "Vitis installer was cancelled"}
    return {"kind": "installer_failure", "retryable": False,
            "summary": "Vitis installer failed; inspect the installation log"}


class Pipeline:
    def __init__(self, installer_pid):
        self.state = {"pid": os.getpid(), "process_identity": process_identity(os.getpid()),
            "started_at": timestamp(), "status": "running", "stage": "starting",
            "installer_pid": installer_pid, "fpga_deployment_complete": False,
            "fpga_executed": False, "completed_steps": [], "error": None}
        self.save()

    def save(self, **updates):
        self.state.update(updates)
        self.state["updated_at"] = timestamp()
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=2) + "\n")
        tmp.replace(STATE)

    def wait_for_installer(self, pid):
        self.save(stage="waiting_for_installer")
        identity = process_identity(pid)
        started = time.monotonic()
        while identity and process_identity(pid) == identity:
            self.save(download_progress=download_progress())
            if time.monotonic() - started > 48 * 3600:
                raise RuntimeError("Installer still running after 48 hours; inspect the installation log")
            time.sleep(30)
        console = log_tail(REPORTS / "vitis-install-resume.log")
        if "The installation failed" in console or "Error downloading files" in console:
            failure = classify_install_failure(console)
            self.save(stage="installing_tools", install_failure=failure)
            raise RuntimeError(failure["summary"])
        self.save(stage="finding_tools", download_progress=download_progress())

    def install_tools(self, max_attempts=3, retry_delay=60, command=None):
        """Own the installer process so its exit code, retries and lifetime are known."""
        command = command or [str(ROOT / "scripts/setup_vitis.sh"), "install"]
        self.save(stage="installing_tools", install_max_attempts=max_attempts)
        for attempt in range(1, max_attempts + 1):
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            path = REPORTS / f"vitis-install-{stamp}-attempt-{attempt}.log"
            started = time.time()
            self.save(stage="installing_tools", install_attempt=attempt, install_log=str(path),
                      current_log=str(path), install_failure=None, download_progress=None)
            print(f"{timestamp()} Installer attempt {attempt}/{max_attempts}", flush=True)
            with path.open("w") as output:
                os.fchmod(output.fileno(), 0o600)
                process = subprocess.Popen(command, cwd=ROOT, stdout=output,
                    stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                self.save(installer_pid=process.pid, child_pid=process.pid)
                try:
                    while True:
                        try:
                            code = process.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            self.save(download_progress=download_progress(path))
                            if time.time() - started > 48 * 3600:
                                raise RuntimeError("Vitis installer exceeded the 48-hour limit")
                except BaseException:
                    self.stop_child(process)
                    raise
            self.save(child_pid=None, installer_exit_code=code,
                      download_progress=download_progress(path))
            console = log_tail(path)
            failed_message = "The installation failed" in console or "Error downloading files" in console
            if code == 0 and not failed_message:
                self.state["completed_steps"].append("install_tools")
                self.save(stage="finding_tools")
                return
            detail = ""
            detailed_logs = list((Path.home() / ".Xilinx/xinstall").glob("xinstall-*.log"))
            if detailed_logs:
                recent = max(detailed_logs, key=lambda p: p.stat().st_mtime)
                if recent.stat().st_mtime >= started:
                    detail = log_tail(recent)
            failure = classify_install_failure(console, detail)
            self.save(install_failure=failure)
            print(f"{timestamp()} {failure['summary']}", flush=True)
            if not failure["retryable"] or attempt == max_attempts:
                raise RuntimeError(f"{failure['summary']} (attempt {attempt}/{max_attempts}); see {path}")
            self.save(stage="retrying_install", retry_delay_seconds=retry_delay * attempt)
            time.sleep(retry_delay * attempt)

    def command(self, stage, args, timeout, settings=None):
        log_path = REPORTS / f"auto-{stage}.log"
        if settings:
            args = ["/bin/bash", "-c", 'set -e; source "$1"; shift; exec "$@"',
                    "bash", str(settings), *args]
        self.save(stage=stage, current_log=str(log_path), command=args)
        print(f"{timestamp()} Starting {stage}", flush=True)
        with log_path.open("w") as output:
            process = subprocess.Popen(args, cwd=ROOT, stdout=output,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            self.save(child_pid=process.pid)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.stop_child(process)
                raise RuntimeError(f"{stage} timed out; see {log_path}")
            except BaseException:
                self.stop_child(process)
                raise
        if code:
            raise RuntimeError(f"{stage} failed with exit code {code}; see {log_path}")
        self.state["completed_steps"].append(stage)
        self.save(child_pid=None)

    @staticmethod
    def stop_child(process):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def settings_file():
    explicit = os.environ.get("QWEN_VITIS_SETTINGS")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RuntimeError(f"QWEN_VITIS_SETTINGS does not exist: {path}")
        return path
    project_config = Path(__file__).resolve().parents[1] / "deployment.json"
    config = json.loads(project_config.read_text()) if project_config.is_file() else {}
    explicit = config.get("vitis_settings")
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise RuntimeError(f"Configured vitis_settings does not exist: {path}")
        return path
    destination = Path(config.get("tool_installation", {}).get("destination", str(Path.home() / "amd")))
    candidates = [destination / "2025.1/Vitis/settings64.sh", destination / "Vitis/2025.1/settings64.sh",
                  Path("/tools/Xilinx/2025.1/Vitis/settings64.sh"), Path("/opt/Xilinx/2025.1/Vitis/settings64.sh")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Vitis 2025.1 settings64.sh is missing; set QWEN_VITIS_SETTINGS or configure vitis_settings")


def main():
    def interrupted(_signal, _frame):
        raise KeyboardInterrupt("Deployment worker interrupted")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--installer-pid", type=int)
    mode.add_argument("--install", action="store_true", help="Own and supervise the installer, including bounded network retries")
    mode.add_argument("--build-only", action="store_true", help="Resume compilation and validation using the installed Vitis tools")
    mode.add_argument("--add-hbm", action="store_true", help="Add the HBM device family to the existing installation, then resume deployment")
    mode.add_argument("--resume-link", action="store_true", help="Resume the existing VPL project from synthesis without rebuilding successful IP")
    mode.add_argument("--verify-existing", action="store_true", help="Build host code and verify the existing xclbin without FPGA recompilation")
    parser.add_argument("--install-attempts", type=int, default=3)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.installer_pid is not None and args.installer_pid <= 1:
        parser.error("Installer PID must be greater than 1")
    if not 1 <= args.install_attempts <= 5:
        parser.error("--install-attempts must be between 1 and 5")
    if args.plan:
        print(json.dumps({"installer_pid": args.installer_pid,
            "stages": ([] if args.build_only or args.resume_link or args.verify_existing else ["add_hbm_devices" if args.add_hbm else "install_tools" if args.install else "wait_for_installer"]) + ["tool_versions", "platform_info", "check_u55c_clock"] + ([] if args.verify_existing else ["build_kernel"]) + [
                       "build_host", "gemv_hardware_probe", "fpga_runtime_probe", "model_fpga_inference", "verify_result"],
            "firmware_flash": False, "model_download": False}, indent=2))
        return
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "auto-deploy.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("An automatic deployment worker is already running")
        pipeline = Pipeline(args.installer_pid)
        try:
            if args.install:
                pipeline.install_tools(max_attempts=args.install_attempts)
            elif args.add_hbm:
                pipeline.save(install_action="add_hbm_devices")
                pipeline.install_tools(max_attempts=args.install_attempts,
                    command=[str(ROOT / "scripts/setup_vitis.sh"), "add-hbm"])
            elif not (args.build_only or args.resume_link or args.verify_existing):
                pipeline.wait_for_installer(args.installer_pid)
            settings = settings_file()
            pipeline.save(settings=str(settings))
            config = json.loads((ROOT / "deployment.json").read_text())
            platform = config["platform"]
            if not Path(platform).is_file():
                raise RuntimeError(f"Development platform missing: {platform}")
            pipeline.command("vpp_version", ["v++", "--version"], 180, settings)
            pipeline.command("vivado_version", ["vivado", "-version"], 180, settings)
            config["tool_installation"]["installed"] = True
            config["tool_installation"]["status"] = "installed_cli_verified"
            config["fpga_blockers"] = ["Custom GEMV kernel build and FPGA validation are pending"]
            (ROOT / "deployment.json").write_text(json.dumps(config, indent=2) + "\n")
            pipeline.command("platform_info", ["platforminfo", "-p", platform], 180, settings)
            pipeline.command("check_u55c_clock", ["vivado", "-nolog", "-nojournal", "-mode", "batch",
                "-source", str(ROOT / "scripts/check_u55c_clock.tcl")], 180, settings)
            if "U55C_CLOCK_IP_CREATED" not in log_tail(REPORTS / "auto-check_u55c_clock.log"):
                raise RuntimeError("Vivado did not confirm successful U55C clock IP creation")
            if not args.verify_existing:
                build_target = "kernel-link-resume" if args.resume_link else "kernel"
                pipeline.command("build_kernel", ["make", build_target, f"PLATFORM={platform}"], 12 * 3600, settings)
            xclbin = ROOT / "build/hw/qwen_gemv.xclbin"
            if not xclbin.is_file():
                raise RuntimeError("Kernel build returned success without generating the expected xclbin")
            pipeline.command("build_host", ["make", "host", "runtime"], 300)
            pipeline.command("gemv_hardware_probe", [str(ROOT / "build/gemv_probe"), str(xclbin),
                                                    config["fpga_device"]], 600)
            runtime_result = REPORTS / "fpga-runtime-probe-auto.json"
            pipeline.command("fpga_runtime_probe", [config["python"],
                str(ROOT / "scripts/validate_fpga_runtime.py"), "--xclbin", str(xclbin),
                "--report", str(runtime_result)], 600)
            runtime_report = json.loads(runtime_result.read_text())
            if not runtime_report.get("fpga_executed") or not runtime_report.get("passed") or len(runtime_report.get("results", [])) != 2:
                raise RuntimeError("Device-only and tiled FPGA runtime validation failed")
            result_path = REPORTS / "hybrid-fpga-auto.json"
            pipeline.command("model_fpga_inference", [str(ROOT / "run_hybrid.sh"), "--backend", "fpga",
                "--xclbin", str(xclbin), "--max-new-tokens", "32", "--report", str(result_path)], 2 * 3600)
            result = json.loads(result_path.read_text())
            if result.get("backend") != "fpga" or not result.get("fpga_executed") or result.get("native_linear_calls", 0) < 197:
                raise RuntimeError("Inference report does not prove actual FPGA execution")
            if result.get("replaced_linears") != 197 or result.get("quantized"):
                raise RuntimeError("Inference did not preserve the requested 197 floating-point linear layers")
            reference = json.loads((ROOT / "reference/20261026-all-linears/result.json").read_text())
            ids = reference["generated_ids"][0][len(reference["input_ids"][0]):]
            identical = result["generated_ids"] == ids
            with xclbin.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            config["fpga_deployment_complete"] = True
            config["full_decoder_on_fpga"] = False
            config["fpga_blockers"] = []
            config["tool_installation"]["installed"] = True
            config["tool_installation"]["status"] = "installed_and_verified"
            config["verified_xclbin"] = {"path": str(xclbin), "sha256": digest}
            (ROOT / "deployment.json").write_text(json.dumps(config, indent=2) + "\n")
            pipeline.save(status="complete", stage="complete", fpga_executed=True,
                fpga_deployment_complete=True, scope="CPU prefill/attention + FPGA decode linear layers",
                full_decoder_on_fpga=False, generated_ids_match_reference=identical,
                answer=result["answer"], xclbin_sha256=digest, finished_at=timestamp())
            print(f"{timestamp()} Hybrid FPGA deployment completed", flush=True)
        except BaseException as error:
            pipeline.save(status="failed", error=f"{type(error).__name__}: {error}", finished_at=timestamp())
            print(f"{timestamp()} Failed: {error}", file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    main()
