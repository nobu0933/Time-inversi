"""Residual board encoder + complete history GRUs + policy/value heads."""
from dataclasses import asdict, dataclass
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from .encoding import CHANNELS, PADDING, VOCABULARY, encode


@dataclass
class NetworkConfig:
    channels: int = 64
    blocks: int = 3
    history_hidden: int = 64

    def __post_init__(self):
        if self.channels < 4 or self.blocks < 1 or self.history_hidden < 1:
            raise ValueError('channels >= 4, blocks >= 1, history_hidden >= 1 required')


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.GroupNorm(1, channels), nn.ReLU(),
                                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                                    nn.GroupNorm(1, channels))

    def forward(self, x):
        return torch.relu(x + self.layers(x))


def batch_inputs(states, device):
    encoded = [encode(state) for state in states]
    boards = torch.as_tensor(np.stack([x[0] for x in encoded]), device=device)
    histories = []
    for direction in (0, 1):
        seq = [torch.tensor(x[1][direction], dtype=torch.long) for x in encoded]
        histories.append((pad_sequence(seq, batch_first=True, padding_value=PADDING).to(device),
                          torch.tensor([len(x) for x in seq], dtype=torch.long)))
    masks = torch.as_tensor(np.stack([x[2] for x in encoded]), device=device)
    return boards, histories, masks


class PolicyValueNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or NetworkConfig()
        c, h = self.config.channels, self.config.history_hidden
        self.board = nn.Sequential(nn.Conv2d(CHANNELS, c, 3, padding=1, bias=False),
                                   nn.GroupNorm(1, c), nn.ReLU(),
                                   *(ResidualBlock(c) for _ in range(self.config.blocks)),
                                   nn.Flatten(), nn.Linear(c * 36, 128), nn.ReLU())
        self.embedding = nn.Embedding(VOCABULARY, 32, padding_idx=PADDING)
        self.history = nn.GRU(32, h, batch_first=True)
        self.policy = nn.Linear(128 + 2 * h, 37)
        self.value = nn.Sequential(nn.Linear(128 + 2 * h, 64), nn.ReLU(), nn.Linear(64, 1), nn.Tanh())

    def forward(self, boards, histories):
        features = [self.board(boards)]
        for tokens, lengths in histories:
            packed = pack_padded_sequence(self.embedding(tokens), lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.history(packed)
            features.append(hidden[-1])
        combined = torch.cat(features, dim=1)
        return self.policy(combined), self.value(combined).squeeze(-1)

    @torch.inference_mode()
    def predict(self, state):
        self.eval()
        boards, histories, legal = batch_inputs([state], next(self.parameters()).device)
        logits, value = self(boards, histories)
        if not legal.any():
            raise ValueError('Do not predict on a terminal state')
        policy = torch.softmax(logits.masked_fill(~legal, -1e9), dim=-1)
        return policy[0].cpu().numpy(), float(value.item())

    def config_dict(self):
        return asdict(self.config)
