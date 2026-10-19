"""Immutable port of src/game/rules.js. No ML dependencies are required."""
from dataclasses import dataclass, replace

SIZE = 6
BLACK, WHITE = 1, -1
PASS = 36
ACTION_SPACE = 37
CORNERS = (0, 5, 30, 35)
DIRECTIONS = tuple((x, y) for y in (-1, 0, 1) for x in (-1, 0, 1) if x or y)


@dataclass(frozen=True)
class State:
    board: tuple[int, ...]
    origins: tuple[int, ...]
    player: int = BLACK
    direction: int = 1
    forward: tuple[int | None, ...] = ()
    reverse: tuple[int | None, ...] = ()
    used_corners: tuple[int, ...] = ()
    ended: bool = False
    clock_origin: int = 0
    clock_anchor: int = 0

    def history(self, direction):
        return self.forward if direction == 1 else self.reverse

    def key(self):
        # Clock positions are visual only. Full histories, including passes, matter.
        return (self.board, self.origins, self.player, self.direction, self.forward,
                self.reverse, tuple(sorted(self.used_corners)), self.ended)

    def to_dict(self):
        return dict(board=list(self.board), origins=list(self.origins), player=self.player,
                    direction=self.direction, history={'1': list(self.forward), '-1': list(self.reverse)},
                    usedCorners=list(self.used_corners), ended=self.ended,
                    clockOrigin=self.clock_origin, clockAnchor=self.clock_anchor)

    @classmethod
    def from_dict(cls, data):
        return cls(tuple(data['board']), tuple(data['origins']), data['player'], data['direction'],
                   tuple(data['history']['1']), tuple(data['history']['-1']),
                   tuple(data['usedCorners']), data['ended'],
                   data.get('clockOrigin', 0), data.get('clockAnchor', 0))


@dataclass(frozen=True)
class Transition:
    state: State
    removed: int | None = None
    removed_color: int = 0
    index: int | None = None
    captured: tuple[int, ...] = ()
    passed: bool = False
    placed_direction: int | None = None
    before_inversion: State | None = None
    corner_inversion: 'Transition | None' = None

    def to_dict(self):
        result = dict(state=self.state.to_dict(), removed=self.removed, removedColor=self.removed_color)
        if self.placed_direction is not None:
            result.update(index=self.index, captured=list(self.captured), passed=self.passed,
                          placedDirection=self.placed_direction,
                          beforeInversion=self.before_inversion.to_dict() if self.before_inversion else None,
                          cornerInversion=self.corner_inversion.to_dict() if self.corner_inversion else None)
        return result


def create_game():
    board = [0] * 36
    board[14] = board[21] = WHITE
    board[15] = board[20] = BLACK
    return State(tuple(board), (0,) * 36)


def flips(board, index, player):
    if type(index) is not int or not 0 <= index < 36 or board[index]:
        return ()
    result = []
    for dx, dy in DIRECTIONS:
        x, y = index % SIZE + dx, index // SIZE + dy
        ray = []
        while 0 <= x < SIZE and 0 <= y < SIZE and board[y * SIZE + x] == -player:
            ray.append(y * SIZE + x)
            x, y = x + dx, y + dy
        if ray and 0 <= x < SIZE and 0 <= y < SIZE and board[y * SIZE + x] == player:
            result.extend(ray)
    return tuple(result)


def moves(board, player):
    return tuple(i for i in range(36) if flips(board, i, player))


def legal_actions(state):
    if state.ended:
        return ()
    return moves(state.board, state.player) or (PASS,)


def _rewind(state):
    opposite = state.history(-state.direction)
    removed = opposite[-1] if opposite else None
    state = replace(state, **{'reverse' if state.direction == 1 else 'forward': opposite[:-1]})
    color = state.board[removed] if removed is not None else 0
    if removed is not None:
        board, origins = list(state.board), list(state.origins)
        board[removed] = origins[removed] = 0
        state = replace(state, board=tuple(board), origins=tuple(origins))
    black, white = counts(state)
    stalled = not moves(state.board, BLACK) and not moves(state.board, WHITE)
    ended = black == 0 or white == 0 or black + white == 36 or (stalled and -state.direction not in state.origins)
    return Transition(replace(state, ended=ended), removed, color)


def invert(state):
    """Parity API for JS; NOT an action in the current corner-only training rules."""
    if state.ended:
        return None
    turn = _rewind(replace(state, direction=-state.direction))
    if turn.state.direction == 1:
        anchor = min(state.clock_anchor, turn.state.clock_origin + (turn.removed is not None))
        turn = replace(turn, state=replace(turn.state, clock_anchor=anchor, clock_origin=anchor))
    return turn


def advance(state, index=None):
    if state.ended:
        return None
    captured = flips(state.board, index, state.player) if index is not None else ()
    if (index is None and moves(state.board, state.player)) or (index is not None and not captured):
        return None
    board, origins = list(state.board), list(state.origins)
    if index is not None:
        board[index], origins[index] = state.player, state.direction
        for i in captured:
            board[i] = state.player
    origin = state.clock_origin
    if state.direction == -1 and index is not None and all(i is None for i in state.forward):
        origin -= 1
    corner = index in CORNERS and index not in state.used_corners
    placed = replace(state, board=tuple(board), origins=tuple(origins), player=-state.player,
                     clock_origin=origin,
                     used_corners=state.used_corners + (index,) if corner else state.used_corners,
                     **{'forward' if state.direction == 1 else 'reverse': state.history(state.direction) + (index,)})
    turn = _rewind(placed)
    inversion = invert(replace(turn.state, ended=False)) if corner else None
    return replace(turn, state=inversion.state if inversion else turn.state,
                   index=index, captured=captured, passed=index is None, placed_direction=state.direction,
                   before_inversion=turn.state if corner else None, corner_inversion=inversion)


def step(state, action):
    if type(action) is not int or not 0 <= action < ACTION_SPACE:
        raise ValueError(f'Invalid action: {action!r}')
    turn = advance(state, None if action == PASS else action)
    if turn is None:
        raise ValueError(f'Illegal action: {action}')
    return turn.state


def counts(state):
    return state.board.count(BLACK), state.board.count(WHITE)


def outcome(state, player):
    if not state.ended:
        raise ValueError('Outcome is only defined at the end of a game')
    black, white = counts(state)
    return ((black > white) - (black < white)) * player
