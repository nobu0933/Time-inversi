from dataclasses import dataclass
import numpy as np
from .rules import State, create_game, outcome, step
from .mcts import MCTS


@dataclass
class Sample:
    state: State
    policy: np.ndarray
    value: float

    def to_dict(self):
        return dict(state=self.state.to_dict(), policy=self.policy.tolist(), value=float(self.value))

    @classmethod
    def from_dict(cls, data):
        return cls(State.from_dict(data['state']), np.asarray(data['policy'], dtype=np.float32), data['value'])


def self_play(evaluator, rng, simulations=64, temperature_moves=12, max_moves=512):
    state = create_game()
    search = MCTS(evaluator, rng, max_depth=max_moves)
    trajectory = []
    for turn in range(max_moves):
        policy = search.search(state, simulations, noise=True)
        trajectory.append((state, policy))
        if turn < temperature_moves:
            probabilities = policy.astype(np.float64)
            action = int(rng.choice(37, p=probabilities / probabilities.sum()))
        else:
            action = int(rng.choice(np.flatnonzero(policy == policy.max())))
        state = step(state, action)
        if state.ended:
            return [Sample(s, p, float(outcome(state, s.player))) for s, p in trajectory], state
    raise RuntimeError(f'Self-play exceeded {max_moves} turns; the unfinished game is not trained as a draw')
