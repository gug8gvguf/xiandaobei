"""在服务器上直接对比官方实现与 submission 实现的真实模型输出。

用途：判定不一致到底来自模型内部实现，还是来自 solution.py 的生成循环。
脚本先跑官方实现记录结果并释放显存，再跑 submission 实现，逐项打印差异。
"""

import glob
import importlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(ROOT, "progen2")
SUB = os.path.join(ROOT, "submission")
DEVICE = torch.device("cuda:0")

PREFIX = "MKTAYIAKQRQISFVKSHFSRQLEERLG"
STEPS = 4


def load_impl(path, tag):
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    sys.path.insert(0, path)
    try:
        modeling = importlib.import_module("models.progen.modeling_progen")
    finally:
        sys.path.remove(path)
    print("%s modeling file: %s" % (tag, modeling.__file__))
    return modeling


def top_letters(logits, allowed, id2char):
    idx = logits.index_select(-1, allowed).argmax(-1)
    return "".join(id2char[int(allowed[i])] for i in idx)


def main():
    cfg_file = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    model_dir = os.path.dirname(cfg_file)
    tok_file = glob.glob(os.path.join(ASSET, "**", "tokenizer.json"), recursive=True)[0]
    print("model_dir:", model_dir)
    print("tokenizer:", tok_file)

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(tok_file)
    amino = "ACDEFGHIKLMNPQRSTVWY"
    allowed_ids = [tokenizer.token_to_id(c) for c in amino]
    allowed = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)
    id2char = {token_id: c for c, token_id in zip(amino, allowed_ids)}
    ids = torch.tensor([tokenizer.encode(PREFIX).ids], dtype=torch.long, device=DEVICE)
    prefix_len = ids.shape[1]
    print("prefix_len:", prefix_len)

    off_mod = load_impl(ASSET, "official")
    ref = off_mod.ProGenForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    ).eval().to(DEVICE)
    print("ref dtype:", next(ref.parameters()).dtype)

    with torch.inference_mode():
        out = ref.transformer(input_ids=ids, past_key_values=None, use_cache=True)
        ref_hidden, presents = out[0], out[1]
        ref_logits = ref.lm_head(ref_hidden[:, -1]).float()
        ref_letters = [top_letters(ref_logits, allowed, id2char)]
        token = ref_logits.index_select(-1, allowed).argmax(-1).reshape(-1, 1)
        token = allowed[token.reshape(-1)].reshape(-1, 1)
        for _ in range(STEPS - 1):
            hidden = ref.transformer.drop(ref.transformer.wte(token))
            new_presents = []
            for layer, block in enumerate(ref.transformer.h):
                block_out = block(hidden, layer_past=presents[layer], use_cache=True)
                hidden = block_out[0]
                new_presents.append(block_out[1])
            presents = new_presents
            logits = ref.lm_head(ref.transformer.ln_f(hidden)[:, -1]).float()
            ref_letters.append(top_letters(logits, allowed, id2char))
            idx = logits.index_select(-1, allowed).argmax(-1)
            token = allowed[idx].reshape(-1, 1)
        ref_last_hidden = ref_hidden[:, -1].float().cpu()

    del ref, presents
    torch.cuda.empty_cache()

    my_mod = load_impl(SUB, "submission")
    mine = my_mod.ProGenForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    ).eval().to(DEVICE)
    print("mine dtype:", next(mine.parameters()).dtype)
    rotary_dim = int(mine.config.rotary_dim)
    rope = my_mod.fixed_pos_embedding_sincos(rotary_dim, int(mine.config.n_positions), DEVICE)
    mine.transformer.rope = rope
    for block in mine.transformer.h:
        block.attn.attn_fp32 = True

    head_dim = int(mine.config.n_embd) // int(mine.config.n_head)
    total = prefix_len + STEPS
    caches = [
        (
            torch.empty(1, int(mine.config.n_head), total, head_dim, dtype=torch.float32, device=DEVICE),
            torch.empty(1, int(mine.config.n_head), total, head_dim, dtype=torch.float32, device=DEVICE),
        )
        for _ in range(int(mine.config.n_layer))
    ]

    with torch.inference_mode():
        hidden = mine(input_ids=ids, caches=caches, start_pos=0)
        my_logits = mine.lm_head(hidden).float()
        my_last_hidden = hidden.float().cpu()
        my_letters = [top_letters(my_logits, allowed, id2char)]
        token = allowed[my_logits.index_select(-1, allowed).argmax(-1)].reshape(-1, 1)
        for step in range(1, STEPS):
            hidden = mine(input_ids=token, caches=caches, start_pos=prefix_len + step - 1)
            logits = mine.lm_head(hidden).float()
            my_letters.append(top_letters(logits, allowed, id2char))
            token = allowed[logits.index_select(-1, allowed).argmax(-1)].reshape(-1, 1)

    print("\n=== RESULT ===")
    print("official  :", "".join(ref_letters))
    print("submission:", "".join(my_letters))
    print("hidden maxdiff at prefill last position: %.3e"
          % (ref_last_hidden - my_last_hidden).abs().max().item())
    print("logits maxdiff prefill: %.3e" % (ref_logits.cpu() - my_logits.cpu()).abs().max().item())


if __name__ == "__main__":
    main()
