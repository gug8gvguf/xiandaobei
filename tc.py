"""torch.compile 探路：把「整步解码 + lm_head + greedy」编成一个函数，与 eager 逐 token 对比。

用法（服务器项目根目录）：
    python3 tc.py            # 默认 reduce-overhead
    PROGEN_TC_MODE=default python3 tc.py

判据：
    1. token 序列必须与 eager 完全一致（贪心解码不容错）；
    2. 每步耗时必须明显低于 eager（>1.15x 才值得继续）。
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
MODE = os.environ.get("PROGEN_TC_MODE", "reduce-overhead")


def main():
    cfg = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    model_dir = os.path.dirname(cfg)
    sub = Solution(model_dir, os.path.join(ASSET, "tokenizer.json"), device="cuda:0")
    dev = sub.device
    print("torch", torch.__version__, "hip", torch.version.hip, "dev", torch.cuda.get_device_name(dev))
    try:
        import triton
        print("triton", triton.__version__)
    except Exception as exc:  # noqa: BLE001
        print("triton missing:", exc)

    ids0 = torch.tensor([sub.tokenizer.encode(PREFIX).ids], device=dev)
    batch, plen = ids0.shape
    total = plen + STEPS
    print("batch", batch, "prefix", plen, "total", total, "dtype", sub.dtype)

    arange = torch.arange(total, dtype=torch.long, device=dev)
    zero = torch.zeros((), dtype=torch.float32, device=dev)
    neg = sub.model.transformer.h[0].attn.masked_bias
    shape = (batch, sub.n_head, total, sub.head_dim)

    def new_caches():
        return [
            (
                torch.zeros(shape, dtype=torch.float32, device=dev),
                torch.zeros(shape, dtype=sub.dtype, device=dev),
            )
            for _ in range(sub.n_layer)
        ]

    def step(caches, ids_, pos_):
        mask = torch.where(arange <= pos_, zero, neg).view(1, 1, 1, total)
        hidden = sub.model.transformer.decode(ids_, caches, pos_, mask, sub.rope)
        return sub._greedy(sub._lm_head(hidden).float())

    def prefill(caches):
        with torch.inference_mode():
            hidden = sub.model(input_ids=ids0, caches=caches, start_pos=0)
            return sub._greedy(sub._lm_head(hidden).float())

    def run_eager(caches):
        out = [prefill(caches)]
        pos = plen
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(STEPS - 1):
                hidden = sub.model(input_ids=out[-1], caches=caches, start_pos=pos)
                out.append(sub._greedy(sub._lm_head(hidden).float()))
                pos += 1
            torch.cuda.synchronize()
        return out, (time.perf_counter() - t0)

    fn = None
    compile_err = None
    t_compile = 0.0
    try:
        t_begin = time.perf_counter()
        fn = torch.compile(step, mode=MODE, fullgraph=True, dynamic=False)
        warm_caches = new_caches()
        warm_tok = prefill(warm_caches)
        warm_pos = torch.full((1,), plen, dtype=torch.long, device=dev)
        with torch.inference_mode():
            warm_tok = fn(warm_caches, warm_tok, warm_pos)
        torch.cuda.synchronize()
        t_compile = time.perf_counter() - t_begin
    except Exception as exc:  # noqa: BLE001
        compile_err = exc
        fn = None
        print("COMPILE FAILED (mode=%s): %r" % (MODE, exc))

    # 一致性：同样从空 cache 出发，跑同样 64 步
    caches_e = new_caches()
    toks_e, t_eager = run_eager(caches_e)

    if fn is not None:
        caches_c = new_caches()
        out = [prefill(caches_c)]
        pos = torch.full((1,), plen, dtype=torch.long, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(STEPS - 1):
                pos.fill_(pos.item() + 1)
                out.append(fn(caches_c, out[-1], pos))
            torch.cuda.synchronize()
        t_compiled = time.perf_counter() - t0
        toks_c = [int(t) for t in out]
        same = toks_c == [int(t) for t in toks_e]
        first_div = next(
            (i for i, (a, b) in enumerate(zip(toks_e, toks_c)) if int(a) != int(b)),
            None,
        )
        print("\nfirst divergence:", first_div)
        print("eager 64:", [int(t) for t in toks_e][:16], "...")
        print("compiled64:", toks_c[:16], "...")
        print("match:", same)
        print("compile %.1fs  eager %.3fs (%.2f ms/step)  compiled %.3fs (%.2f ms/step)  speedup %.3fx"
              % (t_compile, t_eager, t_eager / (STEPS - 1) * 1e3,
                 t_compiled, t_compiled / (STEPS - 1) * 1e3, t_eager / t_compiled))
    else:
        print("\neager %.3fs (%.2f ms/step)" % (t_eager, t_eager / (STEPS - 1) * 1e3))
        print("error:", repr(compile_err))


if __name__ == "__main__":
    main()
