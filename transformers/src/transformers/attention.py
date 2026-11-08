from torch import nn
import torch.nn.functional as F
import torch


# A single head of attention
class Head(nn.Module):
    def __init__(self, n_head: int, n_embed: int, block_size: int):
        super().__init__()
        self.query = nn.Linear(n_embed, n_head, bias=False)
        self.key = nn.Linear(n_embed, n_head, bias=False)
        self.value = nn.Linear(n_embed, n_head, bias=False)

        self.n_embed = 0.5

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-1, -2)
        wei = wei / self.n_embed ** 0.5
        wei = wei.masked_fill(self.mask == 0, float("-inf"))
        wei = F.softmax(wei, dim=1)

        return wei @ v

    # (T, D)
    # (T, D)
