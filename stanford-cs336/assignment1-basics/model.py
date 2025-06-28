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
