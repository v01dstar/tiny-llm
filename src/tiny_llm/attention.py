import mlx.core as mx
from .basics import softmax, linear

def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # query, key, value: (B, H, L, D)  and (B, H, S, D) for K, V
    L = query.shape[-2]
    S = key.shape[-2]
    D = query.shape[-1]
    scale_factor = 1.0 / mx.sqrt(D) if scale is None else scale
    attention_bias = mx.zeros(query.shape[:-2] + (L, S), dtype=query.dtype)
    if mask is not None:
        attention_bias = mask
    # B, H, L, D @ B, H, S, D -> B, H, L, S
    attention_scores = query @ key.swapaxes(-1, -2) * scale_factor
    attention_scores = attention_scores + attention_bias
    attention_scores = softmax(attention_scores, axis=-1)
    # B, H, L, S @ B, H, S, D -> B, H, L, D
    output = attention_scores @ value
    return output


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.E = hidden_size # hidden size
        self.H = num_heads # number of attention heads
        self.D = self.E // self.H # dimension of each attention head

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # query, key, value: (B, L, E)  and (B, S, E) for K, V
        # Q: (B, L, E) @ (E, H*D) -> (B, L, H*D)
        shape = query.shape[:-1] + (self.H, self.D)
        Q = linear(query, self.wq).reshape(shape).swapaxes(-2, -3)
        K = linear(key, self.wk).reshape(shape).swapaxes(-2, -3)
        V = linear(value, self.wv).reshape(shape).swapaxes(-2, -3)
        # Reshape to (B, L, H, D) and swap axes to (B, H, L, D)

        # Attention scores: (B, H, L, D) @ (B, H, D, S) -> (B, H, L, S)
        attention_scores = scaled_dot_product_attention_simple(Q, K, V, mask=mask)

        # Reshape back to (B, L, H*D)
        attention_scores = attention_scores.swapaxes(-2, -3)
        attention_scores = attention_scores.reshape(query.shape[:-1] + (self.E,))
        attention_scores = linear(attention_scores, self.wo)
        return attention_scores


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def flash_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
) -> mx.array:
    pass
