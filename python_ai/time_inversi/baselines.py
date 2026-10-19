"""Simple evaluation opponents; greedy always scores the acting player's color."""
from .rules import legal_actions, step, outcome


def random_action(state, rng):
    actions = legal_actions(state)
    if not actions:
        raise ValueError('Terminal state')
    return int(rng.choice(actions))


def greedy_action(state, rng, epsilon=0.0):
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon must be in [0, 1]')
    actions = legal_actions(state)
    if not actions:
        raise ValueError('Terminal state')
    if rng.random() < epsilon:
        return random_action(state, rng)
    scores = []
    for action in actions:
        after = step(state, action)  # Includes disappearance and corner inversion.
        difference = after.board.count(state.player) - after.board.count(-state.player)
        scores.append(100 * outcome(after, state.player) if after.ended else difference)
    best = max(scores)
    return int(rng.choice([a for a, score in zip(actions, scores) if score == best]))
