"""修复验证：关闭 tie_word_embeddings 后，submission 与官方逐 token 是否一致。"""

import glob
import importlib
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(ROOT, "progen2")

PREFIX = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"


def load_solution(path, tag):
    for name in list(sys.modules):
        if name in ("solution", "models") or name.startswith("models."):
            del sys.modules[name]
    sys.path.insert(0, path)
    try:
        mod = importlib.import_module("solution")
    finally:
        sys.path.remove(path)
    print(tag, "solution:", mod.__file__)
    return mod.Solution


def main():
    cfg_file = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    model_dir = os.path.dirname(cfg_file)
    tok = os.path.join(ASSET, "tokenizer.json")

    official = load_solution(os.path.join(ROOT, "baseline"), "official")
    s = official(model_dir, tok, device="cuda:0")
    print("official tied:",
          s.model.lm_head.weight.data_ptr() == s.model.transformer.wte.weight.data_ptr())
    out_ref = s.generate([PREFIX], 32)[0]
    del s
    torch.cuda.empty_cache()

    mine = load_solution(os.path.join(ROOT, "submission"), "submission")
    m = mine(model_dir, tok, device="cuda:0")
    print("submission tied:",
          m.model.lm_head.weight.data_ptr() == m.model.transformer.wte.weight.data_ptr())
    out_sub = m.generate([PREFIX], 32)[0]

    print("official  :", out_ref)
    print("submission:", out_sub)
    print("match:", out_ref == out_sub)


if __name__ == "__main__":
    main()
