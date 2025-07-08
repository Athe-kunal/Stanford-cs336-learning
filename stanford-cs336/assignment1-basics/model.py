import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: str | None = None, dtype=None
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        self.w = nn.Parameter(init.trunc_normal_(w, std=1, a=-3, b=3))

    def forward(self, x: torch.Tensor):
        return torch.einsum("o i, ... i -> ... o", [self.w, x])


class EmbeddingLayer(nn.Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int, device=None, dtype=None
    ):
        super().__init__()
        self.num_embeddings = num_embeddings  # vocab_size
        self.embedding_dim = embedding_dim  # d_model
        self.device = device
        self.dtype = dtype

        self.w = nn.Parameter(
            torch.empty(
                num_embeddings, embedding_dim, device=self.device, dtype=self.dtype
            )
        )

    def forward(self, token_ids: torch.Tensor):
        return self.w[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.gain = nn.Parameter(
            init.trunc_normal_(torch.empty(d_model, dtype=dtype, device=device))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.square(x).mean(dim=-1, keepdim=True) + self.eps)
        normed = x / rms
        out = torch.einsum("d, ... d -> ... d", [self.gain, normed])
        return out.to(x_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_ff: int, d_model: int, device: str | None = None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.device = device
        self.dtype = dtype

        self.w1_weight = nn.Parameter(
            init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype),
                std=1,
                a=-3,
                b=3,
            )
        )
        self.w3_weight = nn.Parameter(
            init.trunc_normal_(
                torch.empty(d_ff, d_model, device=device, dtype=dtype),
                std=1,
                a=-3,
                b=3,
            )
        )
        self.w2_weight = nn.Parameter(
            init.trunc_normal_(
                torch.empty(d_model, d_ff, device=device, dtype=dtype),
                std=1,
                a=-3,
                b=3,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_x = torch.einsum("fd,... d -> ... f", [self.w1_weight, x])
        w3_x = torch.einsum("fd, ... d -> ... f", [self.w3_weight, x])
        w1_w3_x_silu = (w1_x * (1 / (1 + torch.exp(-w1_x)))) * w3_x
        w2_out = torch.einsum("df,...f->...d", [self.w2_weight, w1_w3_x_silu])
        return w2_out


def run_silu(in_features: torch.Tensor) -> torch.Tensor:
    return in_features * (1 / (1 + torch.exp(-in_features)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device: str = None
    ) -> None:
        super().__init__()
        self.theta = theta
        assert d_k % 2 == 0, f"d_k should be even, but got {d_k}"
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._build_cache()

    def reset_parameters(self):
        self._build_cache()

    def _build_cache(self):
        inv_freq = 1 / (
            self.theta ** (torch.arange(0, self.d_k, 2, device=self.device) / self.d_k)
        )
        self.register_buffer(
            "inv_freq", inv_freq, persistent=False
        )  # don't want to be part of the module state_dict
        seq_idx = torch.arange(self.max_seq_len)
        m_theta = seq_idx.unsqueeze(-1) * inv_freq
        m_theta_cos = torch.repeat_interleave(torch.cos(m_theta), repeats=2, dim=-1)
        m_theta_sin = torch.repeat_interleave(torch.sin(m_theta), repeats=2, dim=-1)
        self.register_buffer("m_theta_cos", m_theta_cos, persistent=False)
        self.register_buffer("m_theta_sin", m_theta_sin, persistent=False)

    def forward(self, x: torch.Tensor, input_token_positions: torch.Tensor):
        rope_cache_cos = self.m_theta_cos[input_token_positions]
        rope_cache_sin = self.m_theta_sin[input_token_positions]
        x_alt = torch.stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(
            start_dim=-2, end_dim=-1
        )
        out = (x * rope_cache_cos + x_alt * rope_cache_sin).to(x.dtype)
        return out
