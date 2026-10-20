"""Exact ordinary-Othello labels for arbitrary 6x6 positions, with a hard budget."""
from dataclasses import dataclass
from time import perf_counter
from .ordinary_othello import Position, legal, play, stone_difference, terminal
from .rules import PASS


class SearchLimit(Exception):
    pass


@dataclass(frozen=True)
class Solution:
    # Final stone difference from the input player's perspective for each action.
    action_values: dict[int, int]
    nodes: int

    @property
    def value(self):
        return max(self.action_values.values())

    @property
    def best_actions(self):
        return tuple(a for a, v in self.action_values.items() if v == self.value)


def solve(position, *, max_empty=10, max_nodes=300_000, seconds=5.0):
    """Return an exact answer or None; never relabel an interrupted search as exact."""
    if not isinstance(position, Position):
        raise TypeError('Expected Position')
    if max_empty < 0 or max_nodes < 1 or seconds <= 0:
        raise ValueError('Invalid search budget')
    if position.board.count(0) > max_empty:
        return None
    deadline = perf_counter() + seconds
    nodes = 0
    order = (100, -20, 10, 10, -20, 100,
             -20, -40, 0, 0, -40, -20,
             10, 0, 2, 2, 0, 10,
             10, 0, 2, 2, 0, 10,
             -20, -40, 0, 0, -40, -20,
             100, -20, 10, 10, -20, 100)

    def exact(board, player, passed, alpha, beta):
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or perf_counter() >= deadline:
            raise SearchLimit
        state = Position(board, player, passed)
        if terminal(state):
            return stone_difference(state)
        best = -37
        for action in sorted(legal(state), key=lambda a: order[a] if a != PASS else 0,
                             reverse=True):
            child = play(state, action)
            value = -exact(child.board, child.player, child.passed, -beta, -alpha)
            best = max(best, value)
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best

    try:
        if terminal(position):
            return Solution({PASS: stone_difference(position)}, 1)
        values = {}
        for action in sorted(legal(position), key=lambda a: order[a] if a != PASS else 0,
                             reverse=True):
            child = play(position, action)
            values[action] = -exact(child.board, child.player, child.passed, -37, 37)
        return Solution(values, nodes)
    except SearchLimit:
        return None
