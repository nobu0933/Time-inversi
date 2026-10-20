"""Use the 6x6-reversi-oracle's published coefficients as an ordinary-Othello teacher.

The coefficient file is obtained separately from the upstream project. This
module independently evaluates its four 8-cell edge patterns, legal mobility,
and potential mobility on any 6x6 board, reachable or otherwise.
"""
import argparse
import hashlib
import math
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

from .ordinary_othello import Position, legal, terminal
from .ordinary_teacher import Teacher
from .oracle_opening import OracleOpening
from .rules import PASS

SOURCE_URL = ('https://raw.githubusercontent.com/mame/6x6-reversi-oracle/'
              '3c1469173ed47a4dd604a872e5ffe76532071988/core/data/weight.rs')
SOURCE_SHA256 = '6fe8fb524772986986a0c9ddd8275664ecf4cc81a8915ea7b9bb00bb703f3f6a'
PATTERNS = ((7, 0, 1, 2, 3, 4, 5, 10),
            (10, 5, 11, 17, 23, 29, 35, 28),
            (28, 35, 34, 33, 32, 31, 30, 25),
            (25, 30, 24, 18, 12, 6, 0, 7))


def pattern_ids():
    """Number reversal-equivalent ternary patterns in upstream enumeration order."""
    indices = [0] * (3 ** 8)
    current = 0
    for pattern in range(len(indices)):
        digits = []
        value = pattern
        for _ in range(8):
            digits.append(value % 3)
            value //= 3
        reverse = sum(digit * (3 ** (7 - i)) for i, digit in enumerate(digits))
        if pattern <= reverse:
            indices[pattern] = indices[reverse] = current
            current += 1
    if current != 3321:
        raise AssertionError('Unexpected oracle pattern count')
    return tuple(indices)


PATTERN_IDS = pattern_ids()
NUMBER = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?'
WEIGHT_BLOCK = re.compile(r'Weight\s*\{\s*edge2x:\s*\[([^\]]+)\],\s*'
                          r'mob:\s*(' + NUMBER + r'),\s*pmob_black:\s*(' + NUMBER +
                          r'),\s*pmob_white:\s*(' + NUMBER + r')', re.S)


class OracleEvaluator:
    def __init__(self, stages):
        if len(stages) != 2 or any(len(stage[0]) != 3321 for stage in stages):
            raise ValueError('Oracle must contain two stages of 3321 pattern weights')
        self.stages = stages

    @classmethod
    def from_file(cls, path):
        source = Path(path).read_text(encoding='utf-8')
        blocks = WEIGHT_BLOCK.findall(source)
        if len(blocks) != 2:
            raise ValueError('Expected two oracle weight stages in weight.rs')
        stages = []
        for pattern_text, mobility, potential_own, potential_other in blocks:
            values = tuple(float(value) for value in re.findall(NUMBER, pattern_text))
            stages.append((values, float(mobility), float(potential_own),
                           float(potential_other)))
        if not all(math.isfinite(value) for stage in stages for value in (*stage[0], *stage[1:])):
            raise ValueError('Non-finite oracle coefficient')
        return cls(tuple(stages))

    def raw_score(self, position):
        """Upstream stage/pattern score, relative to the player to move."""
        if not isinstance(position, Position):
            raise TypeError('Expected ordinary Othello Position')
        board = position.board
        own = position.player
        weights, mob_weight, own_weight, other_weight = self.stages[
            0 if board.count(0) < 19 else 1]
        score = 0.0
        for cells in PATTERNS:
            pattern = 0
            place = 1
            for cell in cells:
                pattern += (1 if board[cell] == own else 2 if board[cell] == -own else 0) * place
                place *= 3
            score += weights[PATTERN_IDS[pattern]]
        score += mob_weight * sum(action != PASS for action in legal(position))
        near_empty = set()
        for cell, stone in enumerate(board):
            if stone:
                continue
            x, y = cell % 6, cell // 6
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < 6 and 0 <= y + dy < 6:
                    near_empty.add((y + dy) * 6 + x + dx)
        score += own_weight * sum(board[cell] == own for cell in near_empty)
        score += other_weight * sum(board[cell] == -own for cell in near_empty)
        return score

    def value(self, position):
        # The upstream evaluator reports score*16 as an ordering value. The
        # teacher needs a bounded target, and only score ordering is relevant.
        return math.tanh(self.raw_score(position) / 8.0)


class OracleTeacher(Teacher):
    def __init__(self, evaluator, *, exact_empty=10, max_nodes=300_000,
                 exact_seconds=5.0, approximate_depth=3, approximate_seconds=2.0,
                 temperature=0.2, opening_book=None):
        if approximate_depth < 1 or approximate_seconds <= 0 or temperature <= 0:
            raise ValueError('Invalid teacher settings')
        self.evaluator = evaluator
        self.exact_empty = exact_empty
        self.max_nodes = max_nodes
        self.exact_seconds = exact_seconds
        self.approximate_depth = approximate_depth
        self.approximate_seconds = approximate_seconds
        self.temperature = temperature
        self.opening_book = opening_book

    def network(self, position):
        return np.zeros(37, dtype=np.float32), self.evaluator.value(position)

    def label_with_source(self, position, state=None):
        result = self.opening_book.lookup(state, position) if self.opening_book else None
        if result is not None:
            policy = np.zeros(37, dtype=np.float32)
            for action in result.best_actions:
                policy[action] = 1 / len(result.best_actions)
            return policy, float(np.sign(result.value)), 'opening_book'
        policy, value, exact = Teacher.label(self, position)
        return policy, value, 'exact' if exact else 'approximate'


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fetch the oracle coefficient file for local use')
    parser.add_argument('--download', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.download.exists():
        parser.error(f'Refusing to overwrite {args.download}')
    with urlopen(SOURCE_URL, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('Downloaded oracle coefficients do not match the pinned source')
    args.download.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.download.with_suffix('.tmp')
    temporary.write_bytes(data)
    try:
        OracleEvaluator.from_file(temporary)
        temporary.replace(args.download)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f'Oracle weights saved to {args.download}')


if __name__ == '__main__':
    main()
