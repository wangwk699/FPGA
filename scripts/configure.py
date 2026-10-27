#!/usr/bin/env python3
"""Create local deployment/build configuration without downloading or changing hardware."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="0000:01:00.1")
    parser.add_argument("--platform", type=Path)
    parser.add_argument("--vitis-settings", type=Path)
    parser.add_argument("--vitis-root", type=Path, default=Path.home() / "amd")
    parser.add_argument("--xrt-include", type=Path)
    parser.add_argument("--xrt-lib", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "deployment.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    from export_weights import expected_shapes, read_header
    model = args.model.expanduser().resolve()
    shapes = expected_shapes(json.loads((model / "config.json").read_text()))
    header, _ = read_header(model / "model.safetensors")
    for name, shape in shapes.items():
        if name not in header or header[name]["shape"] != shape or header[name]["dtype"] != "BF16":
            parser.error(f"Expected original BF16 Qwen3-0.6B tensor: {name}")
    for name in ["tokenizer.json", "tokenizer_config.json", "generation_config.json"]:
        if not (model / name).is_file():
            parser.error(f"Missing model file: {model / name}")
    for label in ["python", "platform", "vitis_settings", "xrt_lib"]:
        value = getattr(args, label)
        if value and not value.expanduser().is_file():
            parser.error(f"Missing {label}: {value}")
    if args.xrt_include and not (args.xrt_include.expanduser() / "xrt/xrt_device.h").is_file():
        parser.error("xrt-include must contain xrt/xrt_device.h")
    output = args.output.expanduser().resolve()
    local = output.parent / "config/local.mk"
    install_config = output.parent / "config/vitis-2025.1.local.txt"
    if not args.force and any(p.exists() for p in [output, local, install_config]):
        parser.error("Local configuration exists; choose another --output or explicitly use --force")
    config = json.loads((ROOT / "config/deployment.example.json").read_text())
    config.update(model_snapshot=str(model), python=str(args.python.expanduser().resolve()),
                  fpga_device=args.device,
                  tool_installation={"version": "2025.1", "destination": str(args.vitis_root.expanduser().resolve())})
    config.pop("platform", None)
    config.pop("vitis_settings", None)
    assignments = []
    for key, value, make_key in [("platform", args.platform, "PLATFORM"),
                                 ("xrt_include", args.xrt_include, "XRT_INCLUDE"),
                                 ("xrt_lib", args.xrt_lib, "XRT_LIB")]:
        if value:
            absolute = str(value.expanduser().resolve())
            if any(c in absolute for c in "\n\r#$ "):
                parser.error("Build dependency paths must not contain spaces, newlines, # or $")
            config[key] = absolute
            assignments.append(f"{make_key} ?= {absolute}")
    if args.vitis_settings:
        config["vitis_settings"] = str(args.vitis_settings.expanduser().resolve())
    with (model / "model.safetensors").open("rb") as stream:
        config["model_file_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    local.parent.mkdir(parents=True, exist_ok=True)
    install_template = (ROOT / "config/vitis-2025.1.example.txt").read_text()
    install_template = "\n".join("Destination=" + config["tool_installation"]["destination"]
        if line.startswith("Destination=") else line for line in install_template.splitlines()) + "\n"
    output.write_text(json.dumps(config, indent=2) + "\n")
    local.write_text("# Local machine settings; do not commit.\n" + "\n".join(assignments) + "\n")
    install_config.write_text(install_template)
    print(f"Created {output}, {local}, {install_config}; no model download or hardware action performed.")


if __name__ == "__main__":
    main()
