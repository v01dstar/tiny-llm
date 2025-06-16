import mlx.core as mx
import math


def softmax(x: mx.array, axis: int) -> mx.array:
    # TODO: manual implementation
    return mx.softmax(x, axis=axis)


# x is N.. x I
# w is O x I
# bias is O
# I may not be 1 dimensional

def linear(
    x: mx.array,
    w: mx.array,
    bias: mx.array | None = None,
) -> mx.array:
    return x @ w.T + (bias if bias is not None else 0)


def silu(x: mx.array) -> mx.array:
    pass
