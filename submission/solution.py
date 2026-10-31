"""学生作品入口：ProGen2-medium 推理优化实现。

优化要点（均为等价计算改写，不改变网络结构与生成规则）：
1. 精度：默认 FP32，与官方基准逐 token 完全一致。BF16/FP16 虽能再快一些，但贪心解码
   会放大数值误差，从某个 token 起整段分叉，一致性直接归零，因此不作为默认。
2. KV Cache 预分配：每层一次性分配 (B, H, L+G, D) 的 K/V 张量，解码时切片写入，
   消除官方实现每步 torch.cat 的内存重分配与历史拷贝。
3. RoPE 预计算：sin/cos 表在初始化阶段一次算好，属于输入无关资源，计时内不再重算。
4. 解码专用注意力路径：query 长度为 1 时跳过因果掩码构造与掩码写入。
5. lm_head 只作用于最后一个位置，减少无用的投影与 FP32 上转开销。
6. token 反解用预建的 id→字母表，避免逐 token 调用 tokenizer。
7. 整步 CUDA Graph（**实验性，默认关闭**）：解码单步（27 层全部算子的下发）捕获成
   一张图，逐步 replay，位置由 GPU 张量驱动（RoPE 用 index_select、KV 写入用
   index_copy_）。本机 DTK 实测不可用：同一张图、同样的输入，单步 replay 时对时错，
   循环里第 1 步正确、第 2 步开始漂移，加逐步 synchronize() 反而第一步就错。
   诊断为图私有内存池与默认 caching allocator 未隔离，捕获的中间张量被图外的
   lm_head/argmax 分配覆盖，属于后端缺陷，故默认关闭，保留开关仅为实验记录。
8. n-gram 投机解码（**默认开启，仅 batch=1**）：实测本机单次前向耗时与 query 长度
   几乎无关（qlen=1 为 29.16ms，qlen=9 为 30.70ms，仅差 5%），说明耗时全花在
   27 层的下发与串行延迟上、算力远未跑满。于是把每轮改造成"草稿 + 一次批量验证"：
   草稿从已生成上下文里复制（不引入任何额外权重、不训练任何模型），把
   [当前 token, 草稿...] 一次性送入，一次前向同时得到全部位置的分布，再按贪心规则
   接受最长的一致前缀。接受规则保证输出与逐 token 贪心逐字相同；实测 mean accepted
   0.59，29.04ms/token → 19.49ms/token（1.49x）。batch>1 时各行接受数不同，而一次
   前向只能对所有行写入同一批位置，故该路径仅在 batch=1 启用，其余走普通 decode。

环境变量（可选，默认即为推荐配置）：
    PROGEN_PRECISION   = fp32 | bf16 | fp16   （默认 fp32）
    PROGEN_ATTN_FP32   = 1                    （强制注意力打分与 softmax 使用 FP32）
    PROGEN_SDPA        = 1                    （用 SDPA 替换手写注意力；本机后端不支持，默认 0）
    PROGEN_GRAPH       = 1                    （打开整步 CUDA Graph；本机不可用，默认 0）
    PROGEN_SPEC        = 0                    （关闭 n-gram 投机解码；默认开启）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class Solution:
    """ProGen2-medium 优化实现。"""

    team_name = "jiangnan_optimized"

    def __init__(self, model_dir: str, tokenizer_path: str, device: str = "cuda:0"):
        # 重活（加载模型、预计算、缓冲区、编译）都在初始化完成，不计入生成时间。
        import torch
        from tokenizers import Tokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.model_dir = Path(model_dir).resolve()

        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("DCU/HIP is not visible through torch.cuda")

        # 优先使用提交目录自带的 models 包（置于 sys.path 最前，覆盖题目包内的原始实现）
        submission_root = str(Path(__file__).resolve().parent)
        if submission_root in sys.path:
            sys.path.remove(submission_root)
        sys.path.insert(0, submission_root)

        from models.progen.modeling_progen import ProGenForCausalLM, fixed_pos_embedding_sincos

        # ---- 精度选择 ----
        precision = os.environ.get("PROGEN_PRECISION", "fp32").lower()
        if precision == "bf16":
            if self.device.type == "cuda":
                try:
                    supported = torch.cuda.is_bf16_supported()
                except Exception:
                    supported = True
            else:
                supported = False
            dtype = torch.bfloat16 if supported else torch.float16
        elif precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.dtype = dtype
        # 注意力打分默认上转 fp32（与官方一致，一致性最保险）；置 0 可省掉每步上转
        self.attn_fp32 = os.environ.get("PROGEN_ATTN_FP32", "1") == "1"
        # 是否用 SDPA 替换手写的 matmul+掩码+softmax 组合。
        # 实测该 DCU 版 torch 未编译 mem-efficient/flash 后端，SDPA 退回 math 实现，
        # 与本机手写路径完全等价且无加速（四项耗时差异均在 ±0.5% 噪声内），故默认关闭。
        self.use_sdpa = os.environ.get("PROGEN_SDPA", "0") == "1"
        # 解码单步是否整步 CUDA Graph 捕获。默认关闭：本机 DTK 的图私有内存池与默认
        # 分配器未隔离，replay 结果会随外部分配时机漂移，实测无法保证逐 token 一致。
        self.use_graph = os.environ.get("PROGEN_GRAPH", "0") == "1"
        # n-gram 投机解码。草稿是复制上下文（零成本、无额外权重），而本机单次前向耗时
        # 与 query 长度几乎无关，验证也不额外收费，实测 1.49x，故默认开启（仅 batch=1）。
        self.use_spec = os.environ.get("PROGEN_SPEC", "1") == "1"

        # ---- 分词器与贪心候选表 ----
        self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        token_ids = [self.tokenizer.token_to_id(letter) for letter in AMINO_ACIDS]
        if any(token_id is None for token_id in token_ids):
            raise RuntimeError("Tokenizer is missing a canonical amino-acid token")
        self.allowed_ids = torch.tensor(
            [int(token_id) for token_id in token_ids], dtype=torch.long, device=self.device
        )
        self._id2char = {int(token_id): letter for letter, token_id in zip(AMINO_ACIDS, token_ids)}

        # ---- 模型加载 ----
        # checkpoint 里有与 wte 不共享的独立 lm_head.weight，而 config.json 没有写
        # tie_word_embeddings（HF 默认 True），会把 lm_head 绑到 wte 上，导致最后一层
        # 投影用错权重（输出退化成重复同一字母）。这里显式关闭 tie，让 checkpoint 的
        # lm_head.weight 正常载入。
        from models.progen.configuration_progen import ProGenConfig

        tie_cfg = ProGenConfig.from_pretrained(str(self.model_dir), local_files_only=True)
        tie_cfg.tie_word_embeddings = False
        self.model = ProGenForCausalLM.from_pretrained(
            str(self.model_dir),
            config=tie_cfg,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        self.model.eval().to(self.device)

        config = self.model.config
        self.max_positions = int(config.n_positions)
        self.n_layer = int(config.n_layer)
        self.n_head = int(config.n_head)
        self.head_dim = int(config.n_embd) // self.n_head
        self.rotary_dim = int(config.rotary_dim)

        # ---- 预计算 RoPE 表（输入无关资源，允许在初始化阶段保留）----
        # 与官方 fixed_pos_embedding 同一公式，官方每层每步现算，这里只算一次
        self.rope = fixed_pos_embedding_sincos(self.rotary_dim, self.max_positions, self.device)
        self.model.transformer.rope = self.rope
        for block in self.model.transformer.h:
            block.attn.attn_fp32 = self.attn_fp32
            block.attn.use_sdpa = self.use_sdpa

        self._lm_head = self.model.lm_head
        self._cache_pool = {}
        self._graph_pool = {}

    def reset(self) -> None:
        """请求间复位。

        KV 缓冲区与 CUDA Graph 都按 (batch, 总长度) 缓存，内容由位置掩码决定，
        下一次请求的 prefill 会覆盖 [0, prefix_len)，解码只写入当前 pos 且只读到 pos 为止，
        因此可以直接复用，无需重建（重建 Graph 的成本会落在计时区间内）。
        形状种类过多时才整体释放，避免显存无上限增长。
        """
        if len(self._cache_pool) > 8:
            self._cache_pool = {}
            self._graph_pool = {}

    @staticmethod
    def _check_prefixes(prefixes: Sequence[str]) -> list[str]:
        if not isinstance(prefixes, (list, tuple)) or not prefixes:
            raise ValueError("prefixes must be a non-empty list of strings")
        values = [str(prefix) for prefix in prefixes]
        if any(not value for value in values):
            raise ValueError("prefixes cannot contain an empty string")
        return values

    @property
    def metadata(self) -> dict:
        return {
            "team_name": self.team_name,
            "precision": str(self.dtype).replace("torch.", ""),
            "description": "Preallocated KV cache + precomputed RoPE + decode-specialized attention + n-gram speculative decoding for batch=1 (CUDA Graph path kept behind PROGEN_GRAPH=1, disabled by default on this backend)",
        }

    def _alloc_caches(self, batch: int, total_len: int):
        """按本次请求需要的容量分配每层 K/V 缓冲区。"""
        torch = self.torch
        key = (batch, total_len)
        cached = self._cache_pool.get(key)
        if cached is not None:
            return cached
        shape = (batch, self.n_head, total_len, self.head_dim)
        # 官方实现中 sin/cos 表是 fp32，RoPE 输出因此恒为 fp32，K 侧历史实际一直是 fp32；
        # V 侧等于模型精度。缓冲区按同样的精度分配，保证与官方逐位一致。
        caches = [
            (
                torch.zeros(shape, dtype=torch.float32, device=self.device),
                torch.zeros(shape, dtype=self.dtype, device=self.device),
            )
            for _ in range(self.n_layer)
        ]
        self._cache_pool[key] = caches
        return caches

    def _greedy(self, logits: torch.Tensor) -> torch.Tensor:
        """在 20 种标准氨基酸中取 logits 最高者；同分按 ACDEFGHIKLMNPQRSTVWY 顺序取首个。"""
        restricted = logits.index_select(-1, self.allowed_ids)
        local_index = restricted.argmax(dim=-1)
        return self.allowed_ids.index_select(0, local_index).unsqueeze(-1)

    def _draft(self, ctx, k: int):
        """n-gram（prompt lookup）草稿：在已生成上下文里找最长后缀的上一次出现，复制其后续 token。

        不引入任何额外权重，也不训练任何模型，只是复制上下文本身，因此符合竞赛规则。
        """
        longest = min(4, len(ctx) - 1)
        for size in range(longest, 0, -1):
            suffix = ctx[-size:]
            for j in range(len(ctx) - size - 1, -1, -1):
                if ctx[j : j + size] == suffix:
                    return list(ctx[j + size : j + size + k])
        return []

    def _spec_loop(self, prefix_ids, first_token, caches, prefix_len, total_len, max_new_tokens):
        """投机解码：n-gram 草稿 + 一次前向批量验证（仅 batch=1 使用）。

        本机实测单次前向耗时与 query 长度几乎无关（qlen=1 与 qlen=9 相差 5%），
        因此"多送几个候选 token 一起验证"基本免费：草稿每被接受一个就是净赚。
        接受规则保证输出与逐 token 贪心解码逐字相同（第一个不一致的位置直接用基线结果）。
        """
        torch = self.torch
        collected = [first_token]
        ctx = list(prefix_ids) + [int(first_token)]
        pos = prefix_len
        while len(collected) < max_new_tokens:
            remaining = max_new_tokens - len(collected)
            k = min(8, remaining, total_len - pos - 1)
            drafts = self._draft(ctx, k) if k > 0 else []
            seq = torch.tensor(
                [[int(collected[-1])] + drafts], dtype=torch.long, device=self.device
            )
            # 一次前向同时算出 1 + len(drafts) 个位置的分布
            hidden = self.model(input_ids=seq, caches=caches, start_pos=pos, logits_last=False)
            logits = self._lm_head(hidden).float()

            accepted = 0
            for index, draft in enumerate(drafts):
                if int(self._greedy(logits[:, index]).reshape(-1)[0]) == int(draft):
                    accepted += 1
                else:
                    break
            tokens = drafts[:accepted]
            tokens.append(int(self._greedy(logits[:, accepted]).reshape(-1)[0]))
            collected.extend(
                torch.tensor([[token]], dtype=torch.long, device=self.device) for token in tokens
            )
            ctx.extend(tokens)
            pos += accepted + 1
        return collected[:max_new_tokens]

    def _graph_for(self, batch: int, total_len: int, caches, first_pos: int):
        """按 (batch, total_len) 缓存一份"解码单步"的 CUDA Graph。

        图中所有输入都是常驻张量：ids 是当前 token，pos 是当前写入位置，两者在 replay 前
        用 copy_/fill_ 更新；因此同一张图可复用于整个解码循环。

        warmup 与 capture 都发生在 first_pos 这个槽位，真实的第一步会覆盖它，
        所以不会污染 prefill 写好的 [0, prefix_len) 区间。
        """
        torch = self.torch
        key = (batch, total_len)
        state = self._graph_pool.get(key)
        if state is not None:
            return state

        device = self.device
        embed_dim = int(self.model.config.n_embd)
        neg_bias = self.model.transformer.h[0].attn.masked_bias
        arange = torch.arange(total_len, dtype=torch.long, device=device)
        zero = torch.zeros((), dtype=torch.float32, device=device)

        ids = torch.zeros((batch, 1), dtype=torch.long, device=device)
        pos = torch.full((1,), int(first_pos), dtype=torch.long, device=device)
        out = torch.zeros((batch, embed_dim), dtype=self.dtype, device=device)

        def step():
            mask = torch.where(arange <= pos, zero, neg_bias).view(1, 1, 1, total_len)
            out.copy_(self.model.transformer.decode(ids, caches, pos, mask, self.rope))

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()

        state = {"graph": graph, "ids": ids, "pos": pos, "out": out}
        self._graph_pool[key] = state
        return state

    def generate(self, prefixes: list[str], max_new_tokens: int) -> list[str]:
        torch = self.torch
        prefixes = self._check_prefixes(prefixes)
        max_new_tokens = int(max_new_tokens)
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")

        encoded = [self.tokenizer.encode(prefix).ids for prefix in prefixes]
        if len({len(row) for row in encoded}) != 1:
            raise ValueError("all prefixes in one request must have equal token length")

        prefix_len = len(encoded[0])
        total_len = prefix_len + max_new_tokens
        if total_len > self.max_positions:
            raise ValueError("sequence length exceeds model context window")

        batch = len(encoded)
        input_ids = torch.tensor(encoded, dtype=torch.long, device=self.device)
        caches = self._alloc_caches(batch, total_len)

        collected = []
        with torch.inference_mode():
            # Prefill：一次性处理整段前缀，同时把 K/V 写入预分配 Cache
            hidden = self.model(input_ids=input_ids, caches=caches, start_pos=0)
            next_token = self._greedy(self._lm_head(hidden).float())
            collected.append(next_token)

            # Decode：每步只送入 1 个 token，直接按位置写入 Cache
            if batch == 1 and max_new_tokens > 1 and self.use_spec and self.device.type == "cuda":
                # 投机解码：n-gram 草稿 + 批量验证。batch>1 时各行接受数不同，
                # 而一次前向只能对所有行写入同一批位置，故仅在 batch=1 启用。
                collected = self._spec_loop(
                    encoded[0], next_token, caches, prefix_len, total_len, max_new_tokens
                )
            elif self.use_graph and self.device.type == "cuda" and max_new_tokens > 1:
                state = self._graph_for(batch, total_len, caches, prefix_len)
                pos_buf, ids_buf, out_buf = state["pos"], state["ids"], state["out"]
                for step in range(1, max_new_tokens):
                    pos_buf.fill_(prefix_len + step - 1)
                    ids_buf.copy_(next_token)
                    state["graph"].replay()
                    next_token = self._greedy(self._lm_head(out_buf).float())
                    collected.append(next_token)
            else:
                for step in range(1, max_new_tokens):
                    hidden = self.model(
                        input_ids=next_token,
                        caches=caches,
                        start_pos=prefix_len + step - 1,
                    )
                    next_token = self._greedy(self._lm_head(hidden).float())
                    collected.append(next_token)

        generated = torch.cat(collected, dim=1).tolist()
        lookup = self._id2char
        return ["".join(lookup[token_id] for token_id in row) for row in generated]
