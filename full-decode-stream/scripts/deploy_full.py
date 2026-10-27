#!/usr/bin/env python3
"""Build and physically verify the complete decoder, preserving hybrid artifacts."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import auto_deploy as supervisor
from model_profiles import PROFILES, canonical_model, config_path, load_config, project_path

supervisor.ROOT = PROJECT
supervisor.REPORTS = PROJECT / "reports"
supervisor.STATE = supervisor.REPORTS / "status.json"
REPORTS = supervisor.REPORTS


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def xrt_tool(*names):
    for name in names:
        path = shutil.which(name)
        if path:
            return path
        candidate = Path("/opt/xilinx/xrt/bin") / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise RuntimeError("XRT utility not found: " + " or ".join(names))


def source_digest():
    return hashlib.sha256((PROJECT / "hls/full_decode.cpp").read_bytes()
        + (PROJECT / "hls/full_decode.hpp").read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def validate_inference(report, fpga, model=None):
    expected = {"backend": "fpga" if fpga else "cpu-cpp", "quantized": False,
        "layers": 28, "matches_reference_tokens": True, "layer_compute_roundtrips_to_cpu": 0,
        "cpu_model_forward_calls": 0, "kernel_source_sha256": source_digest()}
    if model and model!='Qwen3-0.6B':
        expected['model_id']='Qwen/'+model
    if fpga:
        expected.update(fpga_executed=True, full_decoder_on_fpga=True,
            attention_device="fpga", prefill_device="fpga", kv_cache_device="hbm",
            token_selection_device="fpga")
    for key, value in expected.items():
        if report.get(key) != value:
            raise RuntimeError(f"Invalid inference evidence: {key}={report.get(key)!r}, expected {value!r}")
    checks = report.get("numerical_checks", [])
    if len(checks) != 5 or not all(check.get("passed") is True for check in checks):
        raise RuntimeError("All five layer-trace/logit comparisons must pass")
    if report["token_kernel_calls"] != len(report["input_ids"]) + len(report["generated_ids"]) - 1:
        raise RuntimeError("Token kernel call count does not cover the complete prompt and output")
    if digest(report["native_library"]) != report["native_library_sha256"]:
        raise RuntimeError("Runtime library has changed since numerical verification")


def signoff(pipeline, settings):
    runs = PROJECT / "_x/link/vivado/vpl/prj/prj.runs/impl_1"
    timing_path = runs / "hw_bb_locked_timing_summary_routed.rpt"
    timing = timing_path.read_text()
    if "All user specified timing constraints are met." not in timing:
        raise RuntimeError("Routed timing constraints have not been met")
    # Standard Vivado Design Timing Summary, ten numeric setup/hold/pulse columns.
    row = re.search(r"^\s*(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+"
                    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s+"
                    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+)\s+(\d+)\s*$", timing, re.M)
    if not row or any(int(row.group(i)) for i in (3, 7, 11)):
        raise RuntimeError("Cannot verify zero setup, hold and pulse-width failing endpoints")
    checkpoints = list(runs.glob("*_routed.dcp"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected exactly one routed checkpoint")
    bus_path = REPORTS / "bus-skew-routed.rpt"
    pipeline.command("bus_skew_signoff", ["vivado", "-nolog", "-nojournal", "-mode", "batch",
        "-source", str(ROOT / "scripts/check_routed_bus_skew.tcl"), "-tclargs",
        str(checkpoints[0]), str(bus_path)], 1800, settings)
    summary = bus_path.read_text().split("2. Bus Skew Report Per Constraint", 2)[-2]
    bus = re.findall(r"^\s*(Slow|Fast)\s+([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s*$", summary, re.M)
    if not bus or any(float(item[3]) < 0 for item in bus):
        raise RuntimeError("Routed bus skew constraints failed or summary could not be parsed")
    result = {"checked_at": supervisor.timestamp(), "timing_report": str(timing_path),
        "all_user_timing_constraints_met": True, "setup_failing_endpoints": 0,
        "hold_failing_endpoints": 0, "pulse_width_failing_endpoints": 0,
        "wns_ns": float(row.group(1)), "whs_ns": float(row.group(5)),
        "bus_skew_report": str(bus_path), "bus_constraints_checked": len(bus),
        "bus_failing_constraints": 0, "minimum_bus_slack_ns": min(float(item[3]) for item in bus)}
    save_json(REPORTS / "signoff.json", result)
    return result


def main():
    global PROJECT, REPORTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=canonical_model,choices=PROFILES,default='Qwen3-0.6B')
    parser.add_argument("--from-stage", choices=["compile", "link", "resume-link", "verify", "finalize"], default="compile")
    parser.add_argument("--baseline-report", type=Path, help="Optional real FPGA report for a matched speed comparison")
    parser.add_argument("--min-speedup", type=float, default=0.0)
    args = parser.parse_args()
    PROJECT=project_path(ROOT,args.model)
    REPORTS=PROJECT/'reports'
    supervisor.ROOT=PROJECT
    supervisor.REPORTS=REPORTS
    supervisor.STATE=REPORTS/'status.json'
    deployment_path=config_path(ROOT,args.model)
    if args.min_speedup < 0:
        parser.error("min-speedup must be nonnegative")
    if args.min_speedup and not args.baseline_report and not (REPORTS / "baseline.json").is_file():
        parser.error("min-speedup requires --baseline-report or an existing local baseline")
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / "deploy.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("A full-decoder deployment worker is already running")
        pipeline = supervisor.Pipeline(None)
        pipeline.save(scope="Complete 28-layer FPGA decoder including attention, KV, norms and MLP",
                      full_decoder_on_fpga=False, requested_full_decoder_on_fpga=True,
                      kernel_source_sha256=source_digest())
        def interrupted(_signal, _frame):
            raise KeyboardInterrupt("Full decoder deployment interrupted")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            config = load_config(ROOT,args.model)
            settings = supervisor.settings_file()
            validate_inference(json.loads((REPORTS / "full-cpp-validation.json").read_text()), False,args.model)
            pipeline.save(software_full_decoder_verified=True, settings=str(settings))
            if args.from_stage == "compile":
                pipeline.command("compile_kernel", ["make", "compile"], 3 * 3600, settings)
                hls = PROJECT / "_x/reports/qwen_full_decode/hls_reports/qwen_full_decode_csynth.rpt"
                shutil.copy2(hls, REPORTS / "hls-full-decode.rpt")
                resources = re.search(r"\|Total\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", hls.read_text())
                if not resources or int(resources.group(2)) > 2600 or int(resources.group(4)) > 360000:
                    raise RuntimeError("HLS resource estimate exceeds the conservative implementation budget")
                save_json(REPORTS / "compiled-source.json", {"kernel_source_sha256": source_digest(),
                    "xo_sha256": digest(PROJECT / "build/hw/qwen_full_decode.xo")})
            if args.from_stage not in {"verify", "finalize"}:
                compiled = json.loads((REPORTS / "compiled-source.json").read_text())
                if compiled["kernel_source_sha256"] != source_digest() or compiled["xo_sha256"] != digest(PROJECT / "build/hw/qwen_full_decode.xo"):
                    raise RuntimeError("HLS artifact differs from the verified source")
                pipeline.command("link_kernel", ["make", "resume-link" if args.from_stage == "resume-link" else "link"],
                                 16 * 3600, settings)
                save_json(REPORTS / "linked-source.json", {**compiled,
                    "xclbin_sha256": digest(PROJECT / "build/hw/qwen_full_decode.xclbin")})
            xclbin = PROJECT / "build/hw/qwen_full_decode.xclbin"
            linked = json.loads((REPORTS / "linked-source.json").read_text())
            if linked["kernel_source_sha256"] != source_digest() or linked["xclbin_sha256"] != digest(xclbin):
                raise RuntimeError("FPGA bitstream is not bound to the verified complete-decoder source")
            result_path = REPORTS / "full-fpga.json"
            if args.from_stage == "finalize":
                # Re-audit completed evidence without rerunning model computation.
                # Both bitstream and native runtime must be exactly the ones used
                # for the recorded physical model execution.
                signoff_result = json.loads((REPORTS / "signoff.json").read_text())
                if signoff_result.get("xclbin_sha256") != digest(xclbin):
                    raise RuntimeError("Signoff is not bound to the current bitstream")
            else:
                pipeline.command("build_host", ["make", "runtime"], 300)
                signoff_result = signoff(pipeline, settings)
                signoff_result["xclbin_sha256"] = digest(xclbin)
                save_json(REPORTS / "signoff.json", signoff_result)
                pipeline.command("model_fpga_inference", [str(PROJECT / "run_full.sh"), "--backend", "fpga",
                    "--verify", str(PROJECT / "reference/default"), "--max-new-tokens", "32",
                    "--report", str(result_path)], 3 * 3600)
            result = json.loads(result_path.read_text())
            validate_inference(result, True,args.model)
            if result.get("xclbin_sha256") != digest(xclbin):
                raise RuntimeError("Inference report does not match the current bitstream")
            baseline_path = args.baseline_report
            legacy_baseline = REPORTS / "baseline.json"
            if baseline_path is None and legacy_baseline.is_file():
                candidate = Path(json.loads(legacy_baseline.read_text())["inference_report"])
                if candidate.is_file():
                    baseline_path = candidate
            if args.min_speedup and baseline_path is None:
                raise RuntimeError("A speedup threshold requires an available baseline report")
            speedup = None
            performance = {"baseline_available": baseline_path is not None,
                "optimized_report": str(result_path), "quantized": False,
                "optimized_decode_seconds": result["decode_seconds"],
                "optimized_tokens_per_second": result["incremental_decode_tokens_per_second"],
                "weight_port_bits": 512, "parallel_bf16_to_fp32_multipliers": 32,
                "fp32_interleaved_accumulators": 16, "matrix_tile_rows": 16,
                "kv_cache_reused_between_q_head_pairs": True, "attention_parallel_lanes": 32,
                "full_decoder_on_fpga": True}
            if baseline_path is not None:
                base_result = json.loads(baseline_path.read_text())
                if (base_result.get("backend") != "fpga" or not base_result.get("fpga_executed")
                        or base_result.get("quantized") is not False):
                    raise RuntimeError("Baseline must be a real unquantized FPGA inference report")
                if result["input_ids"] != base_result["input_ids"] or result["generated_ids"] != base_result["generated_ids"]:
                    raise RuntimeError("Performance comparison requires identical input and output token IDs")
                speedup = base_result["decode_seconds"] / result["decode_seconds"]
                if speedup < args.min_speedup:
                    raise RuntimeError(f"Measured speedup {speedup:.3f}x is below --min-speedup")
                performance.update(baseline_report=str(baseline_path),
                    identical_input_ids=True, identical_generated_ids=True,
                    baseline_decode_seconds=base_result["decode_seconds"], decode_speedup=speedup,
                    baseline_tokens_per_second=base_result["incremental_decode_tokens_per_second"],
                    baseline_prefill_seconds=base_result["prefill_seconds"],
                    optimized_prefill_seconds=result["prefill_seconds"],
                    prefill_speedup=base_result["prefill_seconds"] / result["prefill_seconds"],
                    baseline_elapsed_seconds=base_result["elapsed_seconds"],
                    optimized_elapsed_seconds=result["elapsed_seconds"],
                    elapsed_speedup=base_result["elapsed_seconds"] / result["elapsed_seconds"],
                    baseline_seconds_per_decode_token=base_result["decode_seconds"] / (len(result["generated_ids"]) - 1),
                    optimized_seconds_per_decode_token=result["decode_seconds"] / (len(result["generated_ids"]) - 1))
            save_json(REPORTS / "PERFORMANCE.json", performance)
            pipeline.command("physical_cu", [xrt_tool("xrt-smi", "xbutil"), "examine", "-d", config["fpga_device"],
                "-r", "dynamic-regions"], 60)
            physical = supervisor.log_tail(REPORTS / "auto-physical_cu.log")
            cu = re.search(r"qwen_full_decode:qwen_full_decode_1\s*\|\s*0x[0-9a-fA-F]+\s*\|\s*(\d+)\s*\|\s*\(DONE\|IDLE\|READY\)", physical)
            if not cu or int(cu.group(1)) < result["token_kernel_calls"] or "HEALTHY" not in physical:
                raise RuntimeError("Physical compute-unit execution is not visible in XRT")
            pipeline.command("xclbin_info", [xrt_tool("xclbinutil"), "--info", "--input", str(xclbin)], 60)
            info = supervisor.log_tail(REPORTS / "auto-xclbin_info.log")
            uuid = re.search(r"UUID\s*\(xclbin\)\s*:\s*([a-fA-F0-9-]{36})", info)
            if not uuid or uuid.group(1).lower() not in physical.lower():
                raise RuntimeError("Loaded FPGA UUID does not match the built bitstream")
            completed = supervisor.timestamp()
            artifact = {"path": str(xclbin), "sha256": digest(xclbin), "uuid": uuid.group(1)}
            final = {"model_id":config['model_id'],"completed_at": completed, "full_decoder_on_fpga": True, "quantized": False,
                "model_revision": config["revision"], "model_file_sha256": config["model_file_sha256"],
                "model_hash_kind":config.get('model_hash_kind','single_checkpoint_file'),
                "model_files_sha256":config.get('source_files_sha256'),
                "fpga_device": config["fpga_device"], "xclbin": artifact,
                "native_library": {"path": result["native_library"], "sha256": result["native_library_sha256"]},
                "kernel_source_sha256": source_digest(), "inference_report": str(result_path),
                "generated_ids_match_reference": True, "token_kernel_calls": result["token_kernel_calls"],
                "layer_evaluations": result["layer_evaluations"], "cpu_model_forward_calls": 0,
                "layer_compute_roundtrips_to_cpu": 0, "answer": result["answer"],
                "elapsed_seconds": result["elapsed_seconds"], "signoff": signoff_result,
                "model_downloaded": False, "persistent_firmware_flashed": False,
                "performance": performance}
            save_json(REPORTS / "FINAL-DEPLOYMENT.json", final)
            # Re-read to avoid overwriting unrelated configuration changes during the build.
            config = json.loads(deployment_path.read_text())
            config.setdefault("full_decode_baseline", config.get("full_decode"))
            config.setdefault("full_decode_previous_optimized", config.get("full_decode_optimized"))
            config.setdefault("hybrid_deployment", {key: config.get(key) for key in [
                "scope", "verified_xclbin", "verified_native_library", "hardware_verification"]})
            config["full_decoder_on_fpga"] = True
            config["fpga_deployment_complete"] = True
            config["scope"] = "Complete FPGA prefill and decode; CPU tokenization and control only"
            config["verified_xclbin"] = artifact
            config["verified_native_library"] = final["native_library"]
            config["hardware_verification"] = {"report": str(result_path), "completed_at": completed,
                "full_decoder_on_fpga": True, "generated_ids_match_reference": True,
                "timing_constraints_met": True, "bus_skew_constraints_met": True,
                "token_kernel_calls": result["token_kernel_calls"], "layer_evaluations": result["layer_evaluations"]}
            config["full_decode"] = {"verified": True, "completed_at": completed, "xclbin": artifact,
                "report": str(REPORTS / "FINAL-DEPLOYMENT.json"), "run": str(PROJECT / "run_full.sh")}
            config["full_decode_optimized"] = {**config["full_decode"], "decode_speedup": speedup,
                "tokens_per_second": result["incremental_decode_tokens_per_second"]}
            save_json(deployment_path, config)
            pipeline.save(status="complete", stage="complete", fpga_deployment_complete=True,
                full_decoder_on_fpga=True, fpga_executed=True, generated_ids_match_reference=True,
                answer=result["answer"], xclbin=artifact, finished_at=completed,
                decode_speedup=speedup, tokens_per_second=result["incremental_decode_tokens_per_second"])
            print(f"{completed} Complete FPGA decoder verified", flush=True)
        except BaseException as error:
            pipeline.save(status="failed", error=f"{type(error).__name__}: {error}", finished_at=supervisor.timestamp())
            print(f"{supervisor.timestamp()} Failed: {error}", file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    main()
