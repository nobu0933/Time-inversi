import unittest
from dataclasses import replace

from time_inversi.corner_study import balance_rollout_prefix, corner_options
from time_inversi.rules import BLACK, WHITE, create_game


class CornerStudyTests(unittest.TestCase):
    def test_rollout_prefix_balances_colors_without_losing_states(self):
        initial = create_game()
        states = [replace(initial, player=color, forward=(index,))
                  for index, color in enumerate((BLACK, BLACK, BLACK, WHITE, WHITE, WHITE))]
        result = balance_rollout_prefix(states, 4)
        self.assertEqual([state.player for state in result[:4]], [BLACK, BLACK, WHITE, WHITE])
        self.assertEqual({state.key() for state in result}, {state.key() for state in states})

    def test_a_spent_corner_is_not_a_first_time_trigger(self):
        # Create a legal corner after the trigger has already been consumed.
        state = create_game()
        board = list(state.board)
        board[1], board[2] = WHITE, BLACK
        state = replace(state, board=tuple(board))
        trigger, other = corner_options(state)
        self.assertIn(0, trigger)
        trigger, other = corner_options(replace(state, used_corners=(0,)))
        self.assertNotIn(0, trigger)
        self.assertIn(0, other)


if __name__ == '__main__':
    unittest.main()
