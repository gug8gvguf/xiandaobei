"""torch.compile 探路 v2：用「切片写」而不是 index_copy_，绕开 xindex / Triton AMD pass 崩溃。

与 tc.py 的关键差异：
1. 直接复用 eager 的 forward（start_pos 是 Python int，KV 用切片写、RoPE 用切片取），
   全程没有 index_put/index_copy_，因此不会产生 scatter kernel；
2. 关掉 int 特化（specialize_int=False）+ dynamic=True，让 pos / total_len 变成符号量，
   避免 64 个位置各编译一次。

判据：64 步 token 与 eager 完全一致，且每步耗时明显低于 31ms。
"""

import glob
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "submission"))
from solution import Solution  # noqa: E402

ASSET = os.path.join(ROOT, "progen2")
PREFIX = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
STEPS = 64
MODES = ["default", "reduce-overhead"]


def main():
    cfg = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    model_dir = os.path.dirname(cfg)
    sub = Solution(model_dir, os.path.join(ASSET, "tokenizer.json"), device="cuda:0")
    dev = sub.device
    print("torch", torch.__version__, "hip", torch.version.hip, "dev", torch.cuda.get_device_name(dev))

    ids0 = torch.tensor([sub.tokenizer.encode(PREFIX).ids], device=dev)
    batch, plen = ids0.shape
    total = plen + STEPS
    print("batch", batch, "prefix", plen, "total", total, "dtype", sub.dtype)

    # 让 pos / 由 pos 推出的 total_len 成为符号量，避免每个位置重编译一次
    torch._dynamo.config.specialize_int = False

    shape = (batch, sub.n_head, total, sub.head_dim)

    def new_caches():
        return [
            (
                torch.zeros(shape, dtype=torch.float32, device=dev),
                torch.zeros(shape, dtype=sub.dtype, device=dev),
            )
            for _ in range(sub.n_layer)
        ]

    def step(caches, ids_, pos):
        """整步解码：模型前向 + lm_head + 贪心，全部用切片写，无 scatter。"""
        hidden = sub.model(input_ids=ids_, caches=caches, start_pos=pos)
        return sub._greedy(sub._lm_head(hidden).float())

    def prefill(caches):
        with torch.inference_mode():
            hidden = sub.model(input_ids=ids0, caches=caches, start_pos=0)
            return sub._greedy(sub._lm_head(hidden).float())

    def run_eager(caches):
        out = [prefill(caches)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(STEPS - 1):
                hidden = sub.model(input_ids=out[-1], caches=caches, start_pos=plen + s)
                out.append(sub._greedy(sub._lm_head(hidden).float()))
            torch.cuda.synchronize()
        return out, (time.perf_counter() - t0)

    toks_e, t_eager = run_eager(new_caches())
    print("eager %.2f ms/step" % (t_eager / (STEPS - 1) * 1e3))

    for mode in MODES:
        t_begin = time.perf_counter()
        try:
            fn = torch.compile(step, mode=mode, fullgraph=True, dynamic=True)
            warm = new_caches()
            tok = prefill(warm)
            with torch.inference_mode():
                fn(warm, tok, plen)
            torch.cuda.synchronize()
            t_compile = time.perf_counter() - t_begin
        except Exception as exc:  # noqa: BLE001
            print("[%s] COMPILE FAILED: %r" % (mode, exc))
            continue

        caches_c = new_caches()
        out = [prefill(caches_c)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for s in range(STEPS - 1):
                out.append(fn(caches_c, out[-1], plen + s))
            torch.cuda.synchronize()
        t_compiled = time.perf_counter() - t0

        toks_c = [int(t) for t in out]
        first_div = next((i for i, (a, b) in enumerate(zip(toks_e, toks_c)) if int(a) != int(b)), None)
        print(
            "[%s] compile %.1fs  %.2f ms/step  speedup %.3fx  first_div %s  match %s"
            % (mode, t_compile, t_compiled / (STEPS - 1) * 1e3, t_eager / t_compiled, first_div, toks_c == [int(t) for t in toks_e])
        )


if __name__ == "__main__":
    main()
