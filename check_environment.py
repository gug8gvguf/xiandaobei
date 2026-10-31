#!/usr/bin/env python3
"""检查题目包和 DCU 推理环境，不安装或修改环境。"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "progen2" / "checkpoints" / "progen2-medium" / "pytorch_model.bin"
EXPECTED_MODEL_SIZE = 3_060_333_315
EXPECTED_MODEL_SHA256 = "545F4D3E0072C0F422C90E221100D0EBE756989D0F09AF3EDE57736680ADB9E2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> int:
    print("python:", sys.version.split()[0])
    imported_packages = {}
    dependency_errors = []
    for package in ("torch", "transformers", "tokenizers", "accelerate", "numpy"):
        version = package_version(package)
        print(f"{package}:", version)
        if version == "NOT INSTALLED":
            dependency_errors.append(f"required package '{package}' is not installed")
            continue
        try:
            imported_packages[package] = importlib.import_module(package)
        except Exception as exc:
            dependency_errors.append(f"cannot import required package '{package}': {exc}")
    if dependency_errors:
        for error in dependency_errors:
            print("ERROR:", error)
        return 2
    torch = imported_packages["torch"]
    print("torch HIP:", torch.version.hip)
    print("torch.cuda available:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")

    required = [
        ROOT / "progen2" / "tokenizer.json",
        ROOT / "progen2" / "models" / "progen" / "modeling_progen.py",
        ROOT / "progen2" / "models" / "progen" / "configuration_progen.py",
        ROOT / "progen2" / "checkpoints" / "progen2-medium" / "config.json",
        MODEL,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        print("ERROR: missing files:")
        for path in missing:
            print(" -", path)
        return 2
    config = json.loads(
        (ROOT / "progen2" / "checkpoints" / "progen2-medium" / "config.json").read_text(encoding="utf-8")
    )
    print("model n_positions:", config.get("n_positions"))
    print("model file bytes:", MODEL.stat().st_size)
    if MODEL.stat().st_size != EXPECTED_MODEL_SIZE:
        print("ERROR: model file size is incorrect")
        return 2
    if "--full-hash" in sys.argv:
        actual_hash = sha256(MODEL)
        print("model SHA256:", actual_hash)
        if actual_hash != EXPECTED_MODEL_SHA256:
            print("ERROR: model SHA256 mismatch")
            return 2
    else:
        print("model SHA256: skipped (use --full-hash for the 3 GB file)")
    if not torch.cuda.is_available() or torch.version.hip is None:
        print("NOTE: package files are valid, but inference tests require a DCU/PyTorch HIP runtime")
        return 3
    print("ENVIRONMENT_CHECK_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
