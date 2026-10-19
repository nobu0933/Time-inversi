"""Lossless state inputs: board planes and untruncated ordered turn sequences."""
import numpy as np
from .rules import PASS, legal_actions

CHANNELS = 8
PADDING = 37
EMPTY = 38
VOCABULARY = 39


def encode(state):
    board = np.asarray(state.board).reshape(6, 6)
    origins = np.asarray(state.origins).reshape(6, 6)
    corners = np.zeros(36, dtype=np.float32)
    corners[list(state.used_corners)] = 1
    legal = np.zeros(37, dtype=np.bool_)
    legal[list(legal_actions(state))] = True
    planes = np.stack([board == state.player, board == -state.player,
                       origins == 1, origins == -1, corners.reshape(6, 6),
                       legal[:36].reshape(6, 6), np.full((6, 6), state.direction == 1),
                       np.full((6, 6), state.player == 1)]).astype(np.float32)
    histories = tuple(tuple(PASS if i is None else i for i in history) or (EMPTY,)
                      for history in (state.forward, state.reverse))
    return planes, histories, legal
