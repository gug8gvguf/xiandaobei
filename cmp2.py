"""检查 lm_head 的加载情况：checkpoint 里有没有 lm_head.weight、是否与 wte 共享。"""

import glob
import importlib
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(ROOT, "progen2")


def main():
    cfg_file = glob.glob(os.path.join(ASSET, "**", "config.json"), recursive=True)[0]
    model_dir = os.path.dirname(cfg_file)
    meta = json.load(open(cfg_file))
    print("tie_word_embeddings:", meta.get("tie_word_embeddings"))
    print("architectures:", meta.get("architectures"))

    bins = glob.glob(os.path.join(model_dir, "*.bin"))
    print("ckpt files:", [os.path.basename(p) for p in bins])
    sd = torch.load(bins[0], map_location="cpu")
    print("lm_head keys in ckpt:", [k for k in sd if "lm_head" in k])
    print("wte keys in ckpt:", [k for k in sd if k.endswith("wte.weight")])
    if "lm_head.weight" in sd:
        print("ckpt lm_head == ckpt wte:", torch.equal(sd["lm_head.weight"], sd["transformer.wte.weight"]))
        print("ckpt lm_head absmax:", sd["lm_head.weight"].abs().max().item())

    sub = os.path.join(ROOT, "submission")
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]
    sys.path.insert(0, sub)
    pkg = importlib.import_module("models.progen.modeling_progen")
    print("submission modeling:", pkg.__file__)
    model = pkg.ProGenForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32, low_cpu_mem_usage=True, local_files_only=True
    )
    print("mine lm_head is wte (tied):",
          model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr())
    print("mine lm_head vs wte maxdiff:",
          (model.lm_head.weight - model.transformer.wte.weight).abs().max().item())
    if "lm_head.weight" in sd:
        print("mine lm_head vs ckpt maxdiff:",
              (model.lm_head.weight - sd["lm_head.weight"]).abs().max().item())
    print("mine wte vs ckpt maxdiff:",
          (model.transformer.wte.weight - sd["transformer.wte.weight"]).abs().max().item())


if __name__ == "__main__":
    main()
