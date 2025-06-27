import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional
        # Initialize the positional encoding matrix here if needed
        self.rope_init()

    def rope_init(self) :
        theta = 1.0 / self.base ** (mx.arange(0, self.dims // 2, dtype=mx.float32)*2 / self.dims)
        seq_idx = mx.arange(self.seq_len, dtype=theta.dtype)
        idx_theta = mx.outer(seq_idx, theta)
        self.rope = mx.stack(
            [mx.cos(idx_theta), mx.sin(idx_theta)],
            axis=-1
)


    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        if self.traditional:
            return self.traditional_rope(x, offset)
        else:
            return self.non_traditional_rope(x, offset)

    def traditional_rope(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        if offset is None:
            # x shape is (B, L, H, D)
            x_shaped =  x.reshape(
                x.shape[:-1] +  (x.shape[-1] // 2, 2)
            )
            rope_shaped = self.rope[:x.shape[1]]
            rope_shaped = rope_shaped[None, :, None, :, :] 

            return mx.stack(
                [x_shaped[..., 0] * rope_shaped[..., 0] - x_shaped[..., 1] * rope_shaped[..., 1],
                x_shaped[..., 0] * rope_shaped[..., 1] + x_shaped[..., 1] * rope_shaped[..., 0]],
                axis=-1,
            ).reshape(x.shape)
        else:
            x_shaped = x.reshape(
                x.shape[:-1] + (x.shape[-1] // 2, 2)
            )
            rope_shaped = self.rope[offset]
            rope_shaped = rope_shaped[None, :, None, :, :] 
            return mx.stack(
                [x_shaped[..., 0] * rope_shaped[..., 0] - x_shaped[..., 1] * rope_shaped[..., 1],
                x_shaped[..., 0] * rope_shaped[..., 1] + x_shaped[..., 1] * rope_shaped[..., 0]],
                axis=-1,
            ).reshape(x.shape)

    def non_traditional_rope(self, x: mx.array, offset: list[slice] | slice | None = None) -> mx.array:
        if offset is None:
            # x shape is (B, L, H, D)
            first_half = x[..., : self.dims // 2]
            second_half = x[..., self.dims // 2 :]
            rope_shaped = self.rope[: x.shape[1]]
            rope_shaped = rope_shaped[None, :, None, :, :]
            return mx.concat(
                [
                    first_half * rope_shaped[..., 0] - second_half * rope_shaped[..., 1],
                    first_half * rope_shaped[..., 1] + second_half * rope_shaped[..., 0],
                ],
                axis=-1,
            )
        else:
            first_half = x[..., : self.dims // 2]
            second_half = x[..., self.dims // 2 :]
            rope_slice = self.rope[offset]
            rope_shaped = rope_slice[None, :, None, :, :]
            return mx.concat(
                [
                    first_half * rope_shaped[..., 0] - second_half * rope_shaped[..., 1],
                    first_half * rope_shaped[..., 1] + second_half * rope_shaped[..., 0],
                ],
                axis=-1,
            )

