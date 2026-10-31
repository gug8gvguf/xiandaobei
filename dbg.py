"""临时诊断脚本：逐层对比官方实现与优化实现的隐藏状态差异（排查用，不属于作品）。"""

import os
import sys

os.environ["PROGEN_PRECISION"] = "fp32"

import torch

GEN_DIR = os.path.dirname(os.path.abspath(__file__))
ASSET = os.path.join(GEN_DIR, "progen2")
MODEL_DIR = os.path.join(ASSET, "checkpoints", "progen2-medium")
TOK = os.path.join(ASSET, "tokenizer.json")
DEV = "cuda:0"

from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file(TOK)
ids = tokenizer.encode("MKTAYIAKQRQISFVKSHFSRQ").ids
print("PROMPT_IDS", ids, flush=True)

# ---------------- 官方实现 ----------------
sys.path.insert(0, ASSET)
from models.progen.modeling_progen import ProGenForCausalLM as RefLM

ref = RefLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float32, local_files_only=True).eval().to(DEV)
print("REF_DTYPE", next(ref.parameters()).dtype, flush=True)

input_ids = torch.tensor([ids], device=DEV)
with torch.inference_mode():
    out = ref(input_ids=input_ids, use_cache=False, output_hidden_states=True, return_dict=True)
ref_logits = out.logits.float()
ref_hs = [h.float() for h in out.hidden_states]
print("REF_LOGITS_SHAPE", tuple(ref_logits.shape), flush=True)
print("REF_EMB_MAXDIFF_CHECK", (ref_hs[0].float() - ref.wte(input_ids).float()).abs().max().item(), flush=True)

del ref
torch.cuda.empty_cache()

# ---------------- 优化实现 ----------------
for name in [k for k in list(sys.modules) if k == "models" or k.startswith("models.")]:
    del sys.modules[name]
sys.path.remove(ASSET)
sys.path.insert(0, os.path.join(GEN_DIR, "submission"))

from solution import Solution

sol = Solution(MODEL_DIR, TOK, DEV)
model = sol.model
print("MY_DTYPE", next(model.parameters()).dtype, flush=True)

batch, length = 1, len(ids)
caches = sol._alloc_caches(batch, length)
with torch.inference_mode():
    hidden = model.transformer.wte(input_ids)
    print("EMB_MAXDIFF", (hidden.float() - ref_hs[0]).abs().max().item(), flush=True)
    bad_layer = -1
    for layer_index, block in enumerate(model.transformer.h):
        hidden = block(hidden, cache=caches[layer_index], start_pos=0, rope=sol.rope, attn_fp32=False)
        diff = (hidden.float() - ref_hs[layer_index + 1]).abs().max().item()
        if layer_index < 3 or diff > 1e-2:
            print(
                f"LAYER {layer_index} maxdiff={diff:.6f} "
                f"ref_absmax={ref_hs[layer_index + 1].abs().max().item():.4f} "
                f"my_absmax={hidden.abs().max().item():.4f}",
                flush=True,
            )
        if diff > 1e-2 and bad_layer < 0:
            bad_layer = layer_index
    final_hidden = model.transformer.ln_f(hidden)
    my_logits = model.lm_head(final_hidden).float()

print("LOGITS_MAXDIFF", (my_logits - ref_logits).abs().max().item(), flush=True)
print("REF_TOP1", ref_logits[0, -1].argmax().item(), "MY_TOP1", my_logits[0, -1].argmax().item(), flush=True)
print("FIRST_BAD_LAYER", bad_layer, flush=True)
