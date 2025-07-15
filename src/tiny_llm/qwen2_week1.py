import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen2MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.E = hidden_size  # hidden size
        self.H = num_heads  # number of attention heads
        self.D = self.E // self.H  # dimension of each attention head
        self.K = num_kv_heads  # number of key-value heads
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.rope = RoPE(
            dims=self.D,
            seq_len=self.max_seq_len,
            base=self.theta,
            traditional=False,
        )

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        q = linear(x, self.wq, self.bq)
        k = linear(x, self.wk, self.bk)
        v = linear(x, self.wv, self.bv)
        L = q.shape[-3]  # Length of the query sequence
        q = self.rope(q, offset=slice(offset, offset + L))
        k = self.rope(k, offset=slice(offset, offset + L))
        x = scaled_dot_product_attention_grouped(
            query=q,
            key=k,
            value=v,
            mask=mask,
        )
        x = linear(x, self.wo)
        return x


class Qwen2MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: mx.array) -> mx.array:
        linear_1 = linear(x, self.w_gate)
        silu_1 = silu(linear_1)
        linear_2 = linear(x, self.w_up)
        gated = silu_1 * linear_2
        return linear(gated, self.w_down)


class Qwen2TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        bq: mx.array,
        bk: mx.array,
        bv: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        offset: int,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen2ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
        offset: int,
    ) -> mx.array:
        pass
