# coding=utf-8
"""ProGen2 推理优化版模型实现。

本文件是官方 modeling_progen.py 的**逐行转写**，只做三处等价的数据搬运改写：

1. KV Cache：官方每步 `torch.cat((past_key, key), dim=-2)` 拼接历史；
   这里改为初始化时一次性预分配 (B, H, L+G, D) 缓冲区，按位置切片写入、切片读取。
   —— 写入的就是官方 cat 之后的内容，位置由显式 start_pos 指定（不用 shape 推断）。
2. RoPE：官方每层每步用 fixed_pos_embedding 现算 sin/cos；这里改为初始化时按同一公式
   预计算整张表（输入无关资源），apply_rotary_pos_emb 的切片方式与官方完全一致。
3. 解码阶段 query 长度为 1 时，官方因果掩码必为全 True（where 是空操作），直接跳过。

除此之外，以下细节均与官方保持逐字一致（这些是最容易写错的地方）：
- qkv 的切分顺序是 query, value, key（不是 q,k,v）
- mp_num = 8 的两步 reshape（_split_heads）
- rotary 在 permute(0,2,1,3) 之前完成合成
- _attn 中 query/key 上转 fp32 后打分、softmax 后再转回 value.dtype
- 残差加法顺序 attn_output + feed_forward + residual
- bias / masked_bias 缓冲区同名同形注册
"""

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_progen import ProGenConfig

logger = logging.get_logger(__name__)


def fixed_pos_embedding_sincos(dim, seq_len, device):
    """输入无关的 sin/cos 表，公式与官方 fixed_pos_embedding 完全一致。"""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    sinusoid_inp = torch.einsum("i , j -> i j", positions, inv_freq)
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), axis=-1)
    return x.flatten(-2)


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = map(lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


def apply_rotary_pos_emb_at(x, sincos, pos):
    """单 token 版 RoPE：位置用张量下标选取，使整个计算与 Python 标量无关，可被 CUDA Graph 捕获。

    x 形状 (B, H, 1, D)；pos 是形状 (1,) 的整型张量。取到的 sin/cos 数值与
    apply_rotary_pos_emb(x, sincos, offset=int(pos)) 完全相同。
    """
    sin, cos = map(lambda t: t.index_select(0, pos)[None, :, None, :].repeat_interleave(2, 3), sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


class ProGenAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_attention_heads")

        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = None
        if config.rotary_dim is not None:
            self.rotary_dim = config.rotary_dim

        self.softmax = nn.Softmax(dim=-1)
        # 注意力打分是否上转 fp32；True 与官方完全一致，False 可省掉每步上转开销
        self.attn_fp32 = True
        # 是否用 SDPA 融合 matmul + 缩放 + 因果掩码 + softmax + matmul
        self.use_sdpa = True

    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        query_length, key_length = query.size(-2), key.size(-2)

        # SDPA 路径：一次调用完成 matmul + 缩放 + 因果掩码 + softmax + matmul。
        # 缩放因子按官方的“除以 scale_attn”写成 1/scale_attn；解码阶段 query 长度为 1
        # 时因果掩码必然全可见，用 is_causal=False（与官方跳过掩码的行为一致）。
        if self.use_sdpa and attention_mask is None and head_mask is None:
            if self.attn_fp32:
                query = query.to(torch.float32)
                key = key.to(torch.float32)
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=query_length > 1,
                scale=float(1.0 / self.scale_attn),
            )
            return attn_output, None

        if self.attn_fp32:
            # 与官方一致：打分保持在 fp32，避免溢出
            query = query.to(torch.float32)
            key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / self.scale_attn

        # 解码阶段 query_length == 1 时，causal_mask 全为 True，where 是空操作，直接跳过
        if query_length > 1:
            causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
            attn_weights = torch.where(causal_mask, attn_weights, self.masked_bias.to(attn_weights.dtype))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = self.softmax(attn_weights)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        return attn_output, attn_weights

    def forward_decode(self, hidden_states, cache, pos, mask, rope):
        """解码单步（query 长度恒为 1）的等价实现，专门用于 CUDA Graph 捕获。

        与 forward 的唯一区别是"位置"的处理方式：start_pos 的 Python 切片被换成
        张量下标（RoPE 用 index_select，KV 写入用 index_copy_），注意力改为直接读
        整块预分配 Cache，再用 -1e9 掩码屏蔽 pos 之后的位置。

        -1e9 与官方 causal mask 用的 masked_bias 是同一个值，softmax 在 fp32 下
        exp(-1e9 - max) 严格下溢为 0，因此等价于"只对有效长度做 softmax"。
        """
        qkv = self.qkv_proj(hidden_states)
        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        k_rot = key[:, :, :, : self.rotary_dim]
        k_pass = key[:, :, :, self.rotary_dim :]
        q_rot = query[:, :, :, : self.rotary_dim]
        q_pass = query[:, :, :, self.rotary_dim :]

        k_rot = apply_rotary_pos_emb_at(k_rot, rope, pos)
        q_rot = apply_rotary_pos_emb_at(q_rot, rope, pos)

        key = torch.cat([k_rot, k_pass], dim=-1)
        query = torch.cat([q_rot, q_pass], dim=-1)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        k_cache, v_cache = cache
        k_cache.index_copy_(2, pos, key)
        v_cache.index_copy_(2, pos, value)

        attn_output, _ = self._attn(query, k_cache, v_cache, attention_mask=mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states,
        cache=None,
        start_pos=0,
        rope=None,
        attention_mask=None,
        head_mask=None,
    ):
        qkv = self.qkv_proj(hidden_states)
        mp_num = 8
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        local_dim = self.head_dim * self.num_attention_heads // mp_num
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        value = value.permute(0, 2, 1, 3)

        seq_len = key.shape[1]
        offset = start_pos
        total_len = seq_len + offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]
            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            sincos = rope
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            key = apply_rotary_pos_emb(key, rope, offset=offset)
            query = apply_rotary_pos_emb(query, rope, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        # 官方此处是 key = torch.cat((past_key, key), dim=-2)；
        # 这里写入预分配缓冲区，再取 [0, total_len) 的切片视图，内容完全相同
        k_cache, v_cache = cache
        k_cache[:, :, offset:total_len, :] = key
        v_cache[:, :, offset:total_len, :] = value
        key = k_cache[:, :, :total_len, :]
        value = v_cache[:, :, :total_len, :]

        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return attn_output


class ProGenMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.n_embd
        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        self.fc_out = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ProGenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = ProGenAttention(config)
        self.mlp = ProGenMLP(inner_dim, config)

    def forward(self, hidden_states, cache=None, start_pos=0, rope=None):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn(hidden_states, cache=cache, start_pos=start_pos, rope=rope)
        feed_forward_hidden_states = self.mlp(hidden_states)
        return attn_output + feed_forward_hidden_states + residual

    def decode(self, hidden_states, cache, pos, mask, rope):
        """解码单步版本，运算顺序与 forward 逐行一致。"""
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output = self.attn.forward_decode(hidden_states, cache, pos, mask, rope)
        feed_forward_hidden_states = self.mlp(hidden_states)
        return attn_output + feed_forward_hidden_states + residual


class ProGenPreTrainedModel(PreTrainedModel):
    config_class = ProGenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ProGenModel(ProGenPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([ProGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)
        # 输入无关的 RoPE 表，由 Solution 在初始化阶段填入
        self.rope = None
        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(self, input_ids=None, caches=None, start_pos=0, logits_last=True, **kwargs):
        hidden_states = self.wte(input_ids)
        hidden_states = self.drop(hidden_states)

        for layer_index, block in enumerate(self.h):
            hidden_states = block(hidden_states, cache=caches[layer_index], start_pos=start_pos, rope=self.rope)

        hidden_states = self.ln_f(hidden_states)
        if logits_last:
            hidden_states = hidden_states[:, -1]
        return hidden_states

    def decode(self, input_ids, caches, pos, mask, rope):
        """解码单步：输入长度恒为 1，状态全部通过张量传递，便于 CUDA Graph 捕获。"""
        hidden_states = self.wte(input_ids)
        for layer_index, block in enumerate(self.h):
            hidden_states = block.decode(hidden_states, cache=caches[layer_index], pos=pos, mask=mask, rope=rope)
        hidden_states = self.ln_f(hidden_states)
        return hidden_states[:, -1]


class ProGenForCausalLM(ProGenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"h\.\d+\.attn\.bias",
        r"lm_head\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = ProGenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        caches=None,
        start_pos: int = 0,
        logits_last: bool = True,
        use_cache=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        """caches 为每层 (k_cache, v_cache) 预分配张量；返回指定位置的隐藏状态。"""
        if caches is None:
            caches = past_key_values
        return self.transformer(input_ids=input_ids, caches=caches, start_pos=start_pos, logits_last=logits_last)
