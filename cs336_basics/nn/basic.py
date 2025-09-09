import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from einops import rearrange, einsum
from jaxtyping import Float, Int

from .initialize import init_linear_weights, init_embedding_weights, init_rmsnorm_weights
from .function import scaled_dot_product_attention, softmax


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        '''
        Args:
            in_features (int): final dimension of the input
            out_features (int): final dimension of the output
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None): = None Data type of the parameters
        '''
        super().__init__()
        self.weights: Float[Tensor, "d_out d_in"] = nn.Parameter(
            init_linear_weights((out_features, in_features),
                                in_features, out_features, device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        '''
        Args:
            num_embeddings (int): Size of the vocabulary
            embedding_dim (int): Dimension of the embedding vectors, i.e., dmodel
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None = None): Data type of the parameters
        '''
        super().__init__()
        self.weights: Float[Tensor, "vocab_size d_model"] = nn.Parameter(
            init_embedding_weights((num_embeddings, embedding_dim),
                                   device=device, dtype=dtype)
        )

    def forward(self, token_ids: Int[LongTensor, "... seq"]) -> Float[Tensor, "... seq d_model"]:
        return self.weights[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        '''
        Args:
            d_model (int): Hidden dimension of the model
            eps (float): = 1e-5 Epsilon value for numerical stability
            device (torch.device | None = None): Device to store the parameters on
            dtype (torch.dtype | None): = None Data type of the parameters
        '''
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights: Float[Tensor, "d_model"] = nn.Parameter(
            init_rmsnorm_weights((d_model,), device=device, dtype=dtype)
        )

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        # Cast to FP32 first
        in_type = x.dtype
        x = x.to(torch.float32)

        rs = torch.sum(x**2, dim=-1, keepdim=True)
        rms = torch.sqrt(rs / self.d_model + self.eps)

        result = x / rms * self.weights
        return result.to(in_type)


class SiLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.silu = SiLu()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        '''
        Args:
            theta (float): Θ value for the RoPE
            d_k (int): dimension of query and key vectors
            max_seq_len (int): Maximum sequence length that will be inputted
            device (torch.device | None = None): Device to store the buffer on
        '''
        super().__init__()
        self.cos_value: Float[Tensor, "max_seq_len half_d_k"]
        self.sin_value: Float[Tensor, "max_seq_len half_d_k"]
        self.d_k = d_k
        if d_k % 2 > 0:
            raise ValueError("Rotary positional embedding requires even d_k")
        pos_id = torch.arange(max_seq_len, device=device).float()
        dim_id = torch.arange(0, d_k, 2, device=device).float()
        theta_exp = torch.pow(theta, dim_id / d_k)

        pos_id = rearrange(pos_id, "max_seq_len -> max_seq_len 1")
        dim_id = rearrange(dim_id, "half_d_k -> 1 half_d_k")
        angles = pos_id / theta_exp
        cos_value: Float[Tensor, "max_seq_len half_d_k"] = torch.cos(angles)
        sin_value: Float[Tensor, "max_seq_len half_d_k"] = torch.sin(angles)
        # Save cos_value and sin_value
        self.register_buffer("cos_value", cos_value, persistent=False)
        self.register_buffer("sin_value", sin_value, persistent=False)

    def forward(self, x: Float[Tensor, "... seq d_k"], token_positions: Int[Tensor, "... seq"]) -> Float[Tensor, "... seq d_k"]:
        d_k = x.shape[-1]
        if d_k != self.d_k:
            raise ValueError(
                f"Input d_k ({d_k}) does not match layer d_k ({self.d_k})")
        cos_value: Float[Tensor,
                         "... seq half_d_k"] = self.cos_value[token_positions]
        sin_value: Float[Tensor,
                         "... seq half_d_k"] = self.sin_value[token_positions]

        x = rearrange(
            x, "... seq (half_d_k pair) -> ... seq half_d_k pair", pair=2)
        x0: Float[Tensor, "... seq half_d_k"] = x[..., 0]
        x1: Float[Tensor, "... seq half_d_k"] = x[..., 1]

        rotated_x0: Float[Tensor,
                          "... seq half_d_k"] = cos_value * x0 - sin_value * x1
        rotated_x1: Float[Tensor,
                          "... seq half_d_k"] = sin_value * x0 + cos_value * x1

        rope_x = rearrange([rotated_x0, rotated_x1],
                           "pair ... seq half_d_k -> ... seq (half_d_k pair)", pair=2)
        return rope_x


class CausalMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 rope: RotaryPositionalEmbedding | None = None,
                 dtype=None, device=None) -> None:
        '''
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
        '''
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope

        # (num_heads, d_model) for each of Q, K, V
        self.qkv_proj = nn.Parameter(
            torch.cat([
                init_linear_weights((d_model, d_model), d_model,
                                    d_model, device=device, dtype=dtype),
                init_linear_weights((d_model, d_model), d_model,
                                    d_model, device=device, dtype=dtype),
                init_linear_weights((d_model, d_model), d_model,
                                    d_model, device=device, dtype=dtype),
            ], dim=0)
        )
        self.o_proj = Linear(num_heads * self.head_dim,
                             d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Float[Tensor, "... seq"] | None,
    ) -> Float[Tensor, "... seq d_model"]:
        # Q, K, V
        QKV = einsum(x, self.qkv_proj,
                     "... seq model, model_3 model -> ... seq model_3")
        Q, K, V = (
            rearrange(x, "... seq (num_heads d_k) -> ... num_heads seq d_k",
                      num_heads=self.num_heads)
            for x in QKV.split(self.d_model, dim=-1)
        )
        # RoPE
        if token_positions is not None:
            if self.rope is None:
                raise ValueError(
                    "Rotary positional embedding (RoPE) is not set in this layer.")
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        # Causal Masked Attention
        causal_mask = torch.tril(torch.ones(
            x.shape[-2], x.shape[-2], dtype=torch.bool, device=x.device))
        attn = scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask)
        attn = rearrange(
            attn, "... num_heads seq d_v -> ... seq (num_heads d_v)")
        return self.o_proj(attn)


class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 rope: RotaryPositionalEmbedding, device=None, dtype=None) -> None:
        '''
        Args:
            d_model (int): Dimensionality of the Transformer block inputs.
            num_heads (int): Number of heads to use in multi-head self-attention.
            d_ff (int): Dimensionality of the position-wise feed-forward inner layer.
        '''

        super().__init__()

        self.rmsnorm_1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.mha = CausalMultiheadAttention(
            d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.rmsnorm_2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.swiglu = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq d_model"],
        token_positions: Int[Tensor, "... seq"] | None = None,
    ) -> Float[Tensor, "... seq d_model"]:
        # Sublayer 1: Multi-head Self-Attention
        attn_output = self.mha(self.rmsnorm_1(x),
                               token_positions=token_positions)
        x = x + attn_output
        # Sublayer 2: Position-wise Feed-Forward Network
        ffn_output = self.swiglu(self.rmsnorm_2(x))
        x = x + ffn_output
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
    ):
        '''
        Args:
            vocab_size (int): The size of the vocabulary, necessary for
                determining the dimensionality of the token embedding matrix.
            context_length (int): The maximum context length, necessary
                for determining the dimensionality of the position embedding matrix.
            num_layers (int): The number of Transformer blocks to use.
            d_model (int): The dimensionality of the Transformer block inputs.
            num_heads (int): The number of heads to use in multi-head self-attention.
            d_ff (int): The dimensionality of the position-wise feed-forward inner layer.
            rope_theta (float): Θ value for the RoPE
        '''
        super().__init__()

        rope = RotaryPositionalEmbedding(
            theta=rope_theta, d_k=d_model // num_heads, max_seq_len=context_length, device=device)
        self.token_positions: Int[Tensor, "1 context_length"] = rearrange(
            torch.arange(context_length, device=device), "context_length -> 1 context_length")

        self.token_embedding = Embedding(vocab_size, d_model,
                                         device=device, dtype=dtype)
        self.transformer_blocks = nn.ModuleList([
            PreNormTransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope=rope,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        self.rms_norm = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: Int[LongTensor, "batch seq"]) -> Float[Tensor, "batch seq vocab_size"]:
        '''
        Args:
            input_ids (Int[LongTensor, "batch seq"]): Input token IDs.
            token_positions (Int[LongTensor, "batch seq"] | None = None): Positional indices for each token.
                If using RoPE, this should be provided. If using absolute positional embeddings,
                this can be None.
        Returns:
            Float[Tensor, "batch seq vocab_size"]: The output logits for each token in the vocabulary.
        '''
        token_positions: Int[Tensor,
                             "1 seq"] = self.token_positions[:, :input_ids.shape[-1]]

        x: Float[Tensor, "batch seq d_model"] = self.token_embedding(input_ids)
        for block in self.transformer_blocks:
            x = block(x, token_positions=token_positions)
        x: Float[Tensor, "batch seq d_model"] = self.rms_norm(x)
        x: Float[Tensor, "batch seq vocab_size"] = self.lm_head(x)
        return x
