"""投机解码探路：n-gram（prompt lookup）草稿 + 批量验证，无需任何额外权重（符合"不得引入其他模型"的规则）。

两段测量：
A. 验证税曲线：query 长度 1/2/3/5/9 时单次前向（含 lm_head 全位置）的耗时。
   本机是下发/串行延迟瓶颈，若 qlen=5 只比 qlen=1 贵一点，投机才有意义。
B. 端到端：n-gram 草稿 + 贪心验证循环，对比纯 eager 的 token 是否逐字一致，并给出
   每轮平均接受数与每 token 耗时。

环境变量：
    SD_K     每轮草稿长度（默认 4）
    SD_NGRAM 草稿用的最长后缀 n-gram（默认 4）
    SD_STEPS 生成 token 数（默认 96）
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
STEPS = int(os.environ.get("SD_STEPS", "96"))
K = int(os.environ.get("SD_K", "4"))
NGRAM = int(os.environ.get("SD_NGRAM", "4"))
QLENS = (1, 2, 3, 5, 9)


def main():
    cfg = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    sub = Solution(os.path.dirname(cfg), os.path.join(ASSET, "tokenizer.json"), device="cuda:0")
    dev = sub.device
    print("torch", torch.__version__, "dev", torch.cuda.get_device_name(dev))
    print("steps", STEPS, "k", K, "ngram", NGRAM)

    prompt = sub.tokenizer.encode(PREFIX).ids
    ids0 = torch.tensor([prompt], device=dev)
    batch, plen = ids0.shape
    total = plen + STEPS + 8
    shape = (batch, sub.n_head, total, sub.head_dim)

    def new_caches():
        return [
            (
                torch.zeros(shape, dtype=torch.float32, device=dev),
                torch.zeros(shape, dtype=sub.dtype, device=dev),
            )
            for _ in range(sub.n_layer)
        ]

    def prefill(caches):
        hidden = sub.model(input_ids=ids0, caches=caches, start_pos=0, logits_last=True)
        return sub._greedy(sub._lm_head(hidden).float())

    def greedy(logits):
        return int(sub._greedy(logits).reshape(-1)[0])

    # ---------- A. 验证税曲线 ----------
    print("\n[A] 单次前向耗时（start_pos=%d，含全位置 lm_head）" % plen)
    base = 0.0
    for qlen in QLENS:
        caches = new_caches()
        seq = torch.randint(0, sub.model.config.vocab_size, (batch, qlen), device=dev)
        with torch.inference_mode():
            h = sub.model(input_ids=seq, caches=caches, start_pos=plen, logits_last=False)
            sub._lm_head(h)
            torch.cuda.synchronize()
            reps = 8
            t0 = time.perf_counter()
            for _ in range(reps):
                h = sub.model(input_ids=seq, caches=caches, start_pos=plen, logits_last=False)
                sub._lm_head(h)
            torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / reps * 1e3
        if qlen == QLENS[0]:
            base = per
        print("  qlen=%d  %.2f ms  (x%.2f vs qlen=1)" % (qlen, per, per / base))

    # ---------- B1. 纯 eager 参考 ----------
    caches = new_caches()
    with torch.inference_mode():
        tok = prefill(caches)
        ref = [int(tok)]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for s in range(STEPS - 1):
            h = sub.model(input_ids=tok, caches=caches, start_pos=plen + s, logits_last=True)
            tok = sub._greedy(sub._lm_head(h).float())
            ref.append(int(tok))
        torch.cuda.synchronize()
        t_eager = time.perf_counter() - t0
    print("\n[B] eager      %.2f ms/token" % (t_eager / (STEPS - 1) * 1e3))

    # ---------- B2. n-gram 投机 ----------
    def draft_from_context(ctx, k):
        """在已生成上下文里找最长后缀的上一次出现，复制其后续 token 作为草稿。"""
        n = min(NGRAM, len(ctx) - 1)
        for size in range(n, 0, -1):
            suffix = ctx[-size:]
            for j in range(len(ctx) - size - 1, -1, -1):
                if ctx[j : j + size] == suffix:
                    return list(ctx[j + size : j + size + k])
        return []

    caches = new_caches()
    rounds = accepted = 0
    with torch.inference_mode():
        tok = prefill(caches)
        ctx = list(prompt) + [int(tok)]
        out = [int(tok)]
        pos = plen
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        while len(out) < STEPS:
            k = min(K, STEPS - len(out))
            drafts = draft_from_context(ctx, k)
            seq = [int(tok)] + drafts
            seq_ids = torch.tensor([seq], dtype=torch.long, device=dev)
            hidden = sub.model(input_ids=seq_ids, caches=caches, start_pos=pos, logits_last=False)
            logits = sub._lm_head(hidden).float()
            m = 0
            for i, d in enumerate(drafts):
                if greedy(logits[:, i]) == d:
                    m += 1
                else:
                    break
            new = drafts[:m] + [greedy(logits[:, m])]
            rounds += 1
            accepted += m
            out.extend(new)
            ctx.extend(new)
            tok = torch.tensor([[new[-1]]], dtype=torch.long, device=dev)
            pos += m + 1
        torch.cuda.synchronize()
        t_spec = time.perf_counter() - t0
    out = out[:STEPS]

    print("spec(n-gram)  %.2f ms/token   speedup %.3fx" % (t_spec / (STEPS - 1) * 1e3, t_eager / t_spec))
    print("rounds %d  mean accepted %.2f  (draft cost is免费: 无额外前向)" % (rounds, accepted / rounds))
    print("first divergence:", next((i for i, (a, b) in enumerate(zip(ref, out)) if a != b), None))
    print("match:", out == ref)


if __name__ == "__main__":
    main()
