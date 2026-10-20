"""6x6 pretrained Othello network plus exact endgame labels.

The network architecture and parameter names follow suragnair/alpha-zero-general
(MIT; https://github.com/suragnair/alpha-zero-general). A pretrained weight
file and its upstream license are bundled in python_ai/pretrained/.
"""
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .ordinary_othello import Position, legal, play, terminal
from .ordinary_solver import SearchLimit, solve


class OthelloNNet(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3)
        self.conv4 = nn.Conv2d(channels, channels, 3)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)
        self.fc1 = nn.Linear(channels * 4, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 37)
        self.fc4 = nn.Linear(512, 1)

    def forward(self, board):
        x = board.reshape(-1, 1, 6, 6)
        for convolution, batch_norm in ((self.conv1, self.bn1),
                                        (self.conv2, self.bn2),
                                        (self.conv3, self.bn3),
                                        (self.conv4, self.bn4)):
            x = F.relu(batch_norm(convolution(x)))
        x = x.flatten(1)
        x = F.relu(self.fc_bn1(self.fc1(x)))
        x = F.relu(self.fc_bn2(self.fc2(x)))
        return self.fc3(x), torch.tanh(self.fc4(x)).flatten()


def load_pretrained(path, device='cpu'):
    """Load a state-dict-only upstream checkpoint without arbitrary unpickling."""
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    weights = checkpoint.get('state_dict') if isinstance(checkpoint, dict) else None
    if not isinstance(weights, dict) or 'conv1.weight' not in weights:
        raise ValueError('Not an alpha-zero-general Othello PyTorch checkpoint')
    channels = int(weights['conv1.weight'].shape[0])
    model = OthelloNNet(channels)
    model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


class Teacher:
    def __init__(self, model, *, exact_empty=10, max_nodes=300_000,
                 exact_seconds=5.0, approximate_depth=3,
                 approximate_seconds=2.0, temperature=0.2):
        if approximate_depth < 1 or approximate_seconds <= 0 or temperature <= 0:
            raise ValueError('Invalid teacher settings')
        self.model = model.eval()
        self.exact_empty = exact_empty
        self.max_nodes = max_nodes
        self.exact_seconds = exact_seconds
        self.approximate_depth = approximate_depth
        self.approximate_seconds = approximate_seconds
        self.temperature = temperature

    @torch.inference_mode()
    def network(self, position):
        device = next(self.model.parameters()).device
        board = torch.as_tensor(np.asarray(position.board, dtype=np.float32) * position.player,
                                device=device).reshape(1, 6, 6)
        logits, value = self.model(board)
        return logits[0].cpu().numpy(), float(value.item())

    def _leaf(self, position, depth, deadline):
        if perf_counter() >= deadline:
            raise SearchLimit
        if terminal(position):
            difference = position.board.count(position.player) - position.board.count(-position.player)
            return float(np.sign(difference))
        if depth == 0:
            return self.network(position)[1]
        return max(-self._leaf(play(position, action), depth - 1, deadline)
                   for action in legal(position))

    def label(self, position):
        """Return (37-policy, value, exact_flag); all values are player-relative."""
        if terminal(position):
            raise ValueError('Do not label a terminal position')
        answer = solve(position, max_empty=self.exact_empty,
                       max_nodes=self.max_nodes, seconds=self.exact_seconds)
        policy = np.zeros(37, dtype=np.float32)
        if answer is not None:
            for action in answer.best_actions:
                policy[action] = 1 / len(answer.best_actions)
            return policy, float(np.sign(answer.value)), True
        actions = legal(position)
        logits, raw_value = self.network(position)
        deadline = perf_counter() + self.approximate_seconds
        combined = logits[list(actions)].astype(np.float64)
        value = raw_value
        for depth in range(1, self.approximate_depth + 1):
            try:
                scores = np.array([-self._leaf(play(position, a), depth - 1, deadline)
                                   for a in actions], dtype=np.float64)
            except SearchLimit:
                # Keep the last *fully completed* depth, never a partial root.
                break
            # Use pretrained policy to break value ties; never choose an illegal move.
            combined = scores / self.temperature + 0.15 * logits[list(actions)]
            value = float(np.clip(scores.max(), -1, 1))
        probabilities = np.exp(combined - combined.max())
        probabilities /= probabilities.sum()
        policy[list(actions)] = probabilities.astype(np.float32)
        return policy, value, False

    def label_with_source(self, position, state=None):
        policy, value, exact = self.label(position)
        return policy, value, 'exact' if exact else 'approximate'
