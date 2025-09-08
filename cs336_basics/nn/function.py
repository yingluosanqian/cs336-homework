import torch
import torch.nn as nn
import math
from torch import Tensor, LongTensor, BoolTensor
from einops import rearrange, einsum
from jaxtyping import Float, Int, Bool


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    max_x = torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(x - max_x)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x


def scaled_dot_product_attention(
    query: Float[Tensor, "... seq d_qk"],
    key: Float[Tensor, "... seq d_qk"],
    value: Float[Tensor, "... seq d_v"],
    *,
    attn_mask: Bool[Tensor, "... seq_q seq_kv"] | None = None,
) -> Float[Tensor, "... seq d_v"]:
    scale = math.sqrt(query.shape[-1])
    P = einsum(
        query, key, "... seq_q d_qk, ... seq_kv d_qk -> ... seq_q seq_kv") / scale
    P = P.masked_fill(~attn_mask, float(
        '-inf')) if attn_mask is not None else P
    S = softmax(P, dim=-1)
    return einsum(S, value, "... seq_q seq_kv, ... seq_kv d_v -> ... seq_q d_v")


def cross_entropy_loss(
    logits: Float[Tensor, "... vocab_size"],
    target: Int[Tensor, "..."],
) -> Float[Tensor, "..."]:
    # -log(softmax(logits)[target])
    # => -log( exp(logits[target] - max) / sum_j(exp(logits[j] - max)) )
    # => -log( exp(logits[target] - max) ) + log( sum_j(exp(logits[j] - max)) )
    # => -logits[target] + max + log( sum_j(exp(logits[j] - max)) )
    # => -o + max + log_sum_exp
    max: Float[Tensor, "... 1"] = torch.max(
        logits, dim=-1, keepdim=True).values
    log_sum_exp: Float[Tensor, "..."] = torch.log(
        torch.sum(torch.exp(logits - max), dim=-1))
    o: Float[Tensor, "..."] = logits.gather(
        dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    return (-o + max.squeeze(-1) + log_sum_exp).mean()
