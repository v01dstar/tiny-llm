import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # Cast to float32 for precision during computation
        x_f32 = x.astype(mx.float32)
        weight_f32 = self.weight.astype(mx.float32)
        
        # Compute RMS norm
        variance = mx.mean(mx.square(x_f32), axis=-1, keepdims=True)
        norm = mx.sqrt(variance + self.eps)
        normalized = x_f32 / norm
        result = normalized * weight_f32
        
        # Cast back to original dtype
        return result.astype(x.dtype)
