#!/usr/bin/env python3
"""Print environment capabilities; requires no dataset or external service."""
import argparse
import json
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Optional JSON evidence file; never overwrites an existing file")
    args = parser.parse_args()
    from rtttl_gen.training import select_device
    device = select_device(args.device)
    cuda = torch.cuda.is_available()
    result = {"python_version": platform.python_version(), "python_executable": sys.executable,
              "platform": platform.platform(), "pytorch_version": torch.__version__,
              "torch_cuda_build": torch.version.cuda, "cuda_available": cuda,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "gpu_vram_GiB": torch.cuda.get_device_properties(device).total_memory / 1024**3 if device.type == "cuda" else None,
              "cuda_device_count": torch.cuda.device_count(), "selected_device": str(device),
              "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
              "selected_amp": ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if device.type == "cuda" else "off"}
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2); output.write("\n")
if __name__ == "__main__":
    main()
