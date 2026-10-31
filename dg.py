"""诊断 v3：直接在「真实解码循环」里对比 eager 与 graph，并测试逐步同步是否能修复。

A 循环 = 纯 eager（start_pos 标量）
C 循环 = 完全照 solution.generate 的写法（无同步）
S 循环 = 每步 replay 前后加 torch.cuda.synchronize()
"""

import glob
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "submission"))
from solution import Solution  # noqa: E402

ASSET = os.path.join(ROOT, "progen2")
PREFIX = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
STEPS = 8


def main():
    cfg_file = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    model_dir = os.path.dirname(cfg_file)
    tok = os.path.join(ASSET, "tokenizer.json")

    sub = Solution(model_dir, tok, device="cuda:0")
    dev = sub.device
    ids = torch.tensor([sub.tokenizer.encode(PREFIX).ids], device=dev)
    prefix_len = ids.shape[1]
    total_len = prefix_len + STEPS
    shape = (1, sub.n_head, total_len, sub.head_dim)

    def fresh():
        return [
            (
                torch.zeros(shape, dtype=torch.float32, device=dev),
                torch.zeros(shape, dtype=sub.dtype, device=dev),
            )
            for _ in range(sub.n_layer)
        ]

    with torch.inference_mode():
        caches_a = fresh()
        ha = sub.model(input_ids=ids, caches=caches_a, start_pos=0)
        first = sub._greedy(sub._lm_head(ha).float())

        # ---- A：eager 循环 ----
        toks_a = [int(first)]
        nxt = first
        for s in range(STEPS):
            h = sub.model(input_ids=nxt, caches=caches_a, start_pos=prefix_len + s)
            nxt = sub._greedy(sub._lm_head(h).float())
            toks_a.append(int(nxt))

        # ---- C：graph 循环（与 generate 一致，无同步）----
        caches_c = fresh()
        hc = sub.model(input_ids=ids, caches=caches_c, start_pos=0)
        nxt_c = sub._greedy(sub._lm_head(hc).float())
        toks_c = [int(nxt_c)]
        st_c = sub._graph_for(1, total_len, caches_c, prefix_len)
        for s in range(STEPS):
            st_c["pos"].fill_(prefix_len + s)
            st_c["ids"].copy_(nxt_c)
            st_c["graph"].replay()
            nxt_c = sub._greedy(sub._lm_head(st_c["out"]).float())
            toks_c.append(int(nxt_c))

        # ---- S：graph 循环 + 每步同步 ----
        caches_s = fresh()
        hs = sub.model(input_ids=ids, caches=caches_s, start_pos=0)
        nxt_s = sub._greedy(sub._lm_head(hs).float())
        toks_s = [int(nxt_s)]
        st_s = sub._graph_for(1, total_len, caches_s, prefix_len)
        for s in range(STEPS):
            st_s["pos"].fill_(prefix_len + s)
            st_s["ids"].copy_(nxt_s)
            torch.cuda.synchronize()
            st_s["graph"].replay()
            torch.cuda.synchronize()
            nxt_s = sub._greedy(sub._lm_head(st_s["out"]).float())
            toks_s.append(int(nxt_s))

        print("A:", toks_a)
        print("C:", toks_c)
        print("S:", toks_s)
        div_c = next((i for i, (x, y) in enumerate(zip(toks_a, toks_c)) if x != y), None)
        div_s = next((i for i, (x, y) in enumerate(zip(toks_a, toks_s)) if x != y), None)
        print("first divergence: C@%s S@%s" % (div_c, div_s))
        print("match C:", toks_a == toks_c, " match S:", toks_a == toks_s)


if __name__ == "__main__":
    main()
