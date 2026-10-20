"""Read the 6x6-reversi-oracle opening tree for standard-prefix positions.

The two binary data files are obtained from the upstream Oracle separately.
Tree edges follow that project's fixed black-move order; node data contains
the chosen white reply and the solved white stone advantage for that line.
"""
from dataclasses import dataclass
from pathlib import Path

from .ordinary_othello import Position, legal, play
from .rules import PASS, create_game

FIRST_MOVES = (8, 13, 22, 27)  # C2, B3, E4, D5


def transform(action, first):
    """Canonical E4 coordinates and each first-move orientation are involutions."""
    if action == PASS:
        return PASS
    x, y = action % 6, action // 6
    if first == 8:  # C2: anti-diagonal
        x, y = 5 - y, 5 - x
    elif first == 13:  # B3: half-turn
        x, y = 5 - x, 5 - y
    elif first == 27:  # D5: diagonal
        x, y = y, x
    elif first != 22:
        raise ValueError('Unsupported first move')
    return y * 6 + x


@dataclass(frozen=True)
class BookResult:
    action_values: dict[int, int]

    @property
    def best_actions(self):
        best = max(self.action_values.values())
        return tuple(action for action, value in self.action_values.items() if value == best)

    @property
    def value(self):
        return max(self.action_values.values())


class OracleOpening:
    def __init__(self, tree_path, data_path):
        bits = Path(tree_path).read_bytes()
        self.data = Path(data_path).read_bytes()
        if len(self.data) % 2:
            raise ValueError('Oracle opening data has an incomplete node')
        node_count = len(self.data) // 2
        if node_count < 1 or len(bits) * 8 < node_count * 2 + 2:
            raise ValueError('Oracle opening tree is incomplete')

        def bit(index):
            return (bits[index // 8] >> (index % 8)) & 1

        if bit(0) != 1 or bit(1) != 0:
            raise ValueError('Oracle opening tree has an invalid root')
        self.first_child = [0] * (node_count + 1)
        self.child_counts = bytearray(node_count + 1)
        position, next_node = 2, 2
        for node in range(1, node_count + 1):
            self.first_child[node] = next_node
            count = 0
            while bit(position):
                count += 1
                position += 1
            position += 1
            if count > 36:
                raise ValueError('Oracle opening tree has too many children')
            self.child_counts[node] = count
            next_node += count
        if next_node != node_count + 1:
            raise ValueError('Oracle opening tree node count disagrees with data')

    def _data(self, node):
        return self.data[2 * (node - 1)], self.data[2 * (node - 1) + 1]

    @staticmethod
    def _ordered_legal(position):
        actions = set(legal(position))
        if actions == {PASS}:
            return (PASS,)
        # lookup.rs uses simple_next_board!, which scans the board row-major.
        return tuple(action for action in range(36) if action in actions)

    def lookup(self, state, position):
        """Return a book choice only for an unchanged ordinary opening prefix."""
        if (state is None or state.ended or state.direction != 1 or
                state.reverse or state.used_corners):
            return None
        history = tuple(PASS if action is None else action for action in state.forward)
        actual = Position(create_game().board, 1)
        try:
            for action in history:
                actual = play(actual, action)
        except ValueError:
            return None
        if actual.board != position.board or actual.player != position.player:
            return None
        if not history:
            return BookResult({action: -4 for action in FIRST_MOVES})
        first = history[0]
        if first not in FIRST_MOVES:
            return None
        remaining = [transform(action, first) for action in history[1:]]
        canonical = play(Position(create_game().board, 1), 22)
        node = 1
        while True:
            white_reply, white_value = self._data(node)
            if not remaining:
                reply = transform(white_reply, first)
                return (BookResult({reply: white_value})
                        if reply in legal(position) else None)
            if self.child_counts[node] == 0:
                return None
            if remaining.pop(0) != white_reply:
                return None
            canonical = play(canonical, white_reply)
            black_actions = self._ordered_legal(canonical)
            if self.child_counts[node] != len(black_actions):
                return None
            if not remaining:
                values = {}
                for index, action in enumerate(black_actions):
                    child = self.first_child[node] + index
                    values[transform(action, first)] = -self._data(child)[1]
                return BookResult(values)
            action = remaining.pop(0)
            if action not in black_actions:
                return None
            node = self.first_child[node] + black_actions.index(action)
            canonical = play(canonical, action)
