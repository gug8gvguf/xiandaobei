#!/usr/bin/env python3
"""对一份作品执行四种规定负载，记录用于提交和评分的耗时与一致率。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ALLOWED = set("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission", default=str(ROOT / "submission"),
        help="包含学生 solution.py 的目录，默认为本包 submission/",
    )
    parser.add_argument("--assets", default=str(ROOT / "progen2"), help="模型资源目录")
    parser.add_argument(
        "--cases", default=str(ROOT / "public_test_data" / "public_cases.json"),
        help="统一测试输入及 FP32 参考结果文件",
    )
    parser.add_argument("--device", default="cuda:0", help="DCU 设备，默认 cuda:0")
    parser.add_argument(
        "--output", default=str(ROOT / "results" / "submission_public.json"),
        help="作品测试结果 JSON 路径",
    )
    return parser.parse_args()


def configure_runtime() -> None:
    """将测试产生的缓存放入包内，并固定 CPU 计算线程数。"""
    cache_root = ROOT / ".cache"
    cache_paths = {
        "XDG_CACHE_HOME": cache_root,
        "XDG_CONFIG_HOME": cache_root / "config",
        "TORCH_HOME": cache_root / "torch",
        "TORCH_EXTENSIONS_DIR": cache_root / "torch_extensions",
        "PYTHONPYCACHEPREFIX": cache_root / "python",
        "MIOPEN_USER_DB_PATH": cache_root / "miopen",
        "MIOPEN_CUSTOM_CACHE_DIR": cache_root / "miopen",
        "TMPDIR": cache_root / "tmp",
    }
    for name, directory in cache_paths.items():
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    (cache_root / "config" / "miopen" / "miopen-lockfiles").mkdir(
        parents=True, exist_ok=True
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    # 设置环境变量只影响之后启动的 Python；此处也固定当前进程的字节码缓存。
    sys.pycache_prefix = str(cache_root / "python")


def load_solution(submission_dir: Path, model_dir: Path, tokenizer_path: Path, device: str):
    solution_path = submission_dir / "solution.py"
    if not solution_path.is_file():
        raise FileNotFoundError(f"missing {solution_path}")
    module_name = "contest_solution_" + hashlib.sha256(
        str(solution_path).encode("utf-8")
    ).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(module_name, solution_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot create module spec for solution.py")
    module = importlib.util.module_from_spec(spec)
    if str(submission_dir) not in sys.path:
        sys.path.insert(0, str(submission_dir))
    # Match normal Python imports: decorators such as @dataclass may resolve
    # postponed annotations through sys.modules while the module is executing.
    had_previous_module = module_name in sys.modules
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        solution_class = getattr(module, "Solution", None)
        if solution_class is None:
            raise AttributeError("solution.py must define class Solution")
        return solution_class(
            model_dir=str(model_dir), tokenizer_path=str(tokenizer_path), device=device
        )
    except BaseException:
        if had_previous_module:
            sys.modules[module_name] = previous_module
        else:
            sys.modules.pop(module_name, None)
        raise


def validate_prefixes(tokenizer: Any, prefixes: Any, expected_batch: int, expected_l: int):
    if not isinstance(prefixes, list) or len(prefixes) != expected_batch:
        raise ValueError(f"expected {expected_batch} prefixes")
    if any(not isinstance(prefix, str) or not prefix for prefix in prefixes):
        raise ValueError("each prefix must be a non-empty string")
    lengths = [len(tokenizer.encode(prefix).ids) for prefix in prefixes]
    if lengths != [expected_l] * expected_batch:
        raise ValueError(f"prefix token lengths {lengths} do not equal L={expected_l}")


def validate_outputs(outputs: Any, batch: int, new_tokens: int) -> list[str]:
    if not isinstance(outputs, list) or len(outputs) != batch:
        raise ValueError(f"generate() must return list[str] with {batch} items")
    values: list[str] = []
    for index, output in enumerate(outputs):
        if not isinstance(output, str):
            raise TypeError(f"output {index} is not str")
        if len(output) != new_tokens:
            raise ValueError(f"output {index} length is {len(output)}, expected {new_tokens}")
        illegal = sorted(set(output) - ALLOWED)
        if illegal:
            raise ValueError(f"output {index} has illegal characters: {illegal}")
        values.append(output)
    return values


def synchronize(device: Any, torch: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    configure_runtime()
    # 延后导入，使 --help 在尚未安装深度学习依赖的本机也能查看。
    import torch
    from tokenizers import Tokenizer

    submission_dir = Path(args.submission).expanduser().resolve()
    assets = Path(args.assets).expanduser().resolve()
    cases_path = Path(args.cases).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available() or torch.version.hip is None:
        raise RuntimeError("作品测试须在海光 DCU 的 PyTorch HIP 环境运行，设备名仍使用 cuda:0")

    tokenizer_path = assets / "tokenizer.json"
    model_dir = assets / "checkpoints" / "progen2-medium"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    print("Loading submission (model loading is not timed):", submission_dir)
    load_started = time.perf_counter()
    solution = load_solution(submission_dir, model_dir, tokenizer_path, args.device)
    synchronize(device, torch)
    print(f"Loaded in {time.perf_counter() - load_started:.3f}s")

    rows = []
    all_exact = True
    for load in cases["loads"]:
        load_id = str(load["id"])
        batch = int(load["batch_size"])
        prefix_tokens = int(load["prefix_tokens"])
        new_tokens = int(load["new_tokens"])
        warmup_prefixes = load["warmup_prefixes"]
        prefixes = load["prefixes"]
        reference = load["reference_outputs"]
        validate_prefixes(tokenizer, warmup_prefixes, batch, prefix_tokens)
        validate_prefixes(tokenizer, prefixes, batch, prefix_tokens)
        validate_outputs(reference, batch, new_tokens)

        print(f"\n[{load_id}] warmup: B={batch}, L={prefix_tokens}, G={new_tokens}")
        if hasattr(solution, "reset"):
            solution.reset()
        warmup = solution.generate(warmup_prefixes, new_tokens)
        validate_outputs(warmup, batch, new_tokens)
        synchronize(device, torch)

        if hasattr(solution, "reset"):
            solution.reset()
        synchronize(device, torch)
        started = time.perf_counter()
        outputs = solution.generate(prefixes, new_tokens)
        synchronize(device, torch)
        elapsed = time.perf_counter() - started
        outputs = validate_outputs(outputs, batch, new_tokens)

        same = sum(
            actual_char == expected_char
            for actual, expected in zip(outputs, reference)
            for actual_char, expected_char in zip(actual, expected)
        )
        total = batch * new_tokens
        rate = same / total
        exact = same == total
        all_exact = all_exact and exact
        print(f"[{load_id}] time={elapsed:.4f}s, same={same}/{total}, consistency={rate * 100:.2f}%")
        rows.append({
            "load_id": load_id, "batch_size": batch, "prefix_tokens": prefix_tokens,
            "new_tokens": new_tokens, "time_s": elapsed, "same": same, "total": total,
            "consistency_rate": rate, "exact_match": exact,
        })

    result = {
        "purpose": "contest_test_record",
        "submission": str(submission_dir),
        "torch": torch.__version__, "hip": torch.version.hip,
        "device": torch.cuda.get_device_name(device),
        "all_public_references_exact": all_exact,
        "loads": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nSaved:", output_path)
    if all_exact:
        print("PUBLIC_INTERFACE_TEST_PASSED (100% match with the public FP32 reference)")
    else:
        print("PUBLIC_INTERFACE_TEST_PASSED (format is valid; consistency differences are shown above)")
    print("四项任务测试完成，结果已保存。")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
