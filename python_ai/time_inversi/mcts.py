"""PUCT search. Keys contain full rule state, never just stone colors."""
from dataclasses import dataclass
import numpy as np
from .rules import ACTION_SPACE, legal_actions, outcome, step


@dataclass
class Node:
    prior: np.ndarray
    legal: tuple
    visits: np.ndarray
    total_value: np.ndarray


class MCTS:
    def __init__(self, evaluator, rng=None, c_puct=1.5, max_depth=512):
        self.evaluator = evaluator
        self.rng = rng if rng is not None else np.random.default_rng()
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.nodes = {}

    def _expand(self, state):
        policy, value = self.evaluator.predict(state)
        policy = np.asarray(policy, dtype=np.float64)
        if policy.shape != (ACTION_SPACE,) or not np.all(np.isfinite(policy)) or not np.isfinite(value):
            raise ValueError('Non-finite or incorrectly shaped network output')
        legal = legal_actions(state)
        prior = np.zeros(ACTION_SPACE)
        prior[list(legal)] = np.maximum(policy[list(legal)], 0)
        if prior.sum() <= 0:
            prior[list(legal)] = 1
        prior /= prior.sum()
        self.nodes[state.key()] = Node(prior, legal, np.zeros(ACTION_SPACE, dtype=np.int64), np.zeros(ACTION_SPACE))
        return float(np.clip(value, -1, 1))

    def _visit(self, state, root_key, root_prior, depth=0):
        if state.ended:
            return outcome(state, state.player)
        if depth >= self.max_depth:
            raise RuntimeError('Search exceeded max_depth; no synthetic draw is assigned')
        key = state.key()
        if key not in self.nodes:
            return self._expand(state)
        node = self.nodes[key]
        prior = root_prior if key == root_key else node.prior
        q = np.divide(node.total_value, node.visits, out=np.zeros(ACTION_SPACE), where=node.visits > 0)
        scores = q + self.c_puct * prior * np.sqrt(1 + node.visits.sum()) / (1 + node.visits)
        legal = np.array(node.legal)
        best = legal[np.isclose(scores[legal], scores[legal].max(), rtol=0, atol=1e-12)]
        action = int(self.rng.choice(best))
        child = step(state, action)
        value = self._visit(child, root_key, root_prior, depth + 1)
        if child.player != state.player:
            value = -value
        node.visits[action] += 1
        node.total_value[action] += value
        return value

    def search(self, state, simulations=64, noise=False, alpha=0.3, noise_fraction=0.25):
        if state.ended or simulations < 1:
            raise ValueError('Search requires a nonterminal state and positive simulations')
        if alpha <= 0 or not 0 <= noise_fraction <= 1:
            raise ValueError('Invalid root noise settings')
        key = state.key()
        if key not in self.nodes:
            self._expand(state)
        node = self.nodes[key]
        prior = node.prior.copy()
        if noise:
            legal = list(node.legal)
            prior[legal] = (1 - noise_fraction) * prior[legal] + noise_fraction * self.rng.dirichlet([alpha] * len(legal))
        for _ in range(simulations):
            self._visit(state, key, prior)
        return (node.visits / node.visits.sum()).astype(np.float32)


class UniformEvaluator:
    """For rule/search tests and cheap baseline searches, not a trained network."""
    def predict(self, state):
        return np.ones(ACTION_SPACE) / ACTION_SPACE, 0.0
