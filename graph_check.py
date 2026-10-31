"""CUDA Graph 在 DCU 上的可行性验证。

只回答三个问题：
1) capture/replay 能不能跑通，重放结果是否与 eager 逐位一致；
2) KV cache 的写入位置由 GPU 张量给出（不触发 CPU 同步）时，能否在重放中改变；
3) 256 步小算子链上，重放相对 eager 的启动开销收益是多少。

任何一步报错都说明该路线不可行，直接放弃。
"""

import torch

DEV = torch.device("cuda:0")


def capture(fn, warmup=3):
    """按官方推荐流程捕获一张图：支流预热 -> 支流捕获。"""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def bench(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def main():
    print("torch", torch.__version__)
    print("cuda available", torch.cuda.is_available())
    print("device", torch.cuda.get_device_name(0))

    # ---- 1. 基础 capture / replay 正确性 ----
    x = torch.randn(4, 8, device=DEV)
    w = torch.randn(8, 8, device=DEV)
    out = torch.empty(4, 8, device=DEV)
    with torch.inference_mode():
        g1 = capture(lambda: torch.mm(x, w, out=out))
        x.copy_(torch.randn(4, 8, device=DEV))
        g1.replay()
        torch.cuda.synchronize()
        diff = (out - x @ w).abs().max().item()
        print("1) basic replay maxdiff: %.3e" % diff)

    # ---- 2. KV cache：写入位置是 GPU 张量，重放时可改 ----
    b, h, t, d = 2, 4, 16, 8
    k = torch.zeros(b, h, t, d, device=DEV)
    src = torch.randn(b, h, 1, d, device=DEV)
    pos = torch.zeros(1, dtype=torch.long, device=DEV)
    with torch.inference_mode():
        g2 = capture(lambda: k.index_copy_(2, pos, src))
        for p in (0, 5, 15):
            pos.fill_(p)
            k.zero_()
            g2.replay()
            torch.cuda.synchronize()
            ok = torch.equal(k[:, :, p, :], src[:, :, 0, :])
            print("2) kv write pos=%2d match: %s" % (p, ok))

    # ---- 3. 256 步小算子链的启动开销收益 ----
    a = torch.randn(4, 1536, device=DEV)
    wq = torch.randn(1536, 1536, device=DEV)
    acc = torch.empty_like(a)

    def chain():
        hidden = torch.mm(a, wq)
        hidden = torch.nn.functional.gelu(hidden)
        torch.add(hidden, a, out=acc)

    iters = 256
    with torch.inference_mode():
        g3 = capture(chain)
        t_eager = bench(chain, iters)
        t_graph = bench(g3.replay, iters)
    print("3) %d steps: eager %.2f ms, graph %.2f ms, speedup %.2fx"
          % (iters, t_eager, t_graph, t_eager / t_graph))

    # ---- 4. 重放自身的固定开销（只含 1 个微小 kernel）----
    tiny = torch.zeros(4, device=DEV)

    with torch.inference_mode():
        g4 = capture(lambda: tiny.add_(0.0))
        t_tiny = bench(g4.replay, iters)
    print("4) replay fixed overhead: %.2f us/次" % (t_tiny / iters * 1000.0))

    # ---- 5. 每步 27 层（81 个 kernel），接近真实解码步的算子密度 ----
    def step27():
        for _ in range(27):
            chain()

    with torch.inference_mode():
        g5 = capture(step27, warmup=2)
        t_eager27 = bench(step27, 64)
        t_graph27 = bench(g5.replay, 64)
    print("5) 27 层/步: eager %.2f ms, graph %.2f ms, speedup %.2fx"
          % (t_eager27, t_graph27, t_eager27 / t_graph27))

    print("CUDAGRAPH_CHECK_DONE")


if __name__ == "__main__":
    main()
