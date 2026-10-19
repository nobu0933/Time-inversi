from dataclasses import replace
import unittest
import numpy as np
from time_inversi.rules import *
from time_inversi.encoding import encode, EMPTY, PASS as PASS_TOKEN
from time_inversi.baselines import greedy_action
from time_inversi.mcts import MCTS, UniformEvaluator
from time_inversi.selfplay import self_play
from time_inversi.evaluate import evaluate


class SearchTests(unittest.TestCase):
    def test_encoding_retains_every_pass_in_order(self):
        state = replace(create_game(), forward=(None,) * 300 + (8, None), reverse=(None, 13))
        planes, histories, mask = encode(state)
        self.assertEqual(planes.shape, (8, 6, 6))
        self.assertEqual(len(histories[0]), 302)
        self.assertEqual(histories[0][-2:], (8, PASS_TOKEN))
        self.assertEqual(histories[1], (PASS_TOKEN, 13))
        self.assertEqual(encode(create_game())[1], ((EMPTY,), (EMPTY,)))
        self.assertEqual(tuple(np.flatnonzero(mask)), legal_actions(state))

    def test_greedy_uses_acting_color_even_for_black(self):
        rng = np.random.default_rng(8)
        tested = set()
        state = create_game()
        for _ in range(100):
            if state.ended:
                break
            scores = {}
            for action in legal_actions(state):
                child = step(state, action)
                scores[action] = (100 * outcome(child, state.player) if child.ended else
                                  child.board.count(state.player) - child.board.count(-state.player))
            if len(set(scores.values())) > 1:
                chosen = greedy_action(state, rng)
                self.assertEqual(scores[chosen], max(scores.values()))
                tested.add(state.player)
            state = step(state, int(rng.choice(legal_actions(state))))
        self.assertEqual(tested, {BLACK, WHITE})

    def test_terminal_value_sign_and_legal_mask(self):
        board = [BLACK] * 36
        board[0], board[1] = 0, WHITE
        state = replace(create_game(), board=tuple(board), used_corners=CORNERS)
        search = MCTS(UniformEvaluator(), np.random.default_rng(1))
        policy = search.search(state, 8)
        self.assertEqual(policy[0], 1)
        node = search.nodes[state.key()]
        self.assertEqual(node.total_value[0] / node.visits[0], 1)

    def test_root_noise_does_not_accumulate_into_prior(self):
        state = create_game()
        search = MCTS(UniformEvaluator(), np.random.default_rng(1))
        for _ in range(2):
            p = search.search(state, 3, noise=True)
            self.assertAlmostEqual(float(p.sum()), 1)
            self.assertTrue(all(p[a] == 0 for a in range(37) if a not in legal_actions(state)))
            np.testing.assert_allclose(search.nodes[state.key()].prior[list(legal_actions(state))], .25)

    def test_selfplay_labels_and_balanced_evaluation(self):
        samples, final = self_play(UniformEvaluator(), np.random.default_rng(2), simulations=2)
        for sample in samples:
            self.assertEqual(sample.value, outcome(final, sample.state.player))
            self.assertAlmostEqual(float(sample.policy.sum()), 1)
        stats = evaluate(UniformEvaluator(), games=2, simulations=2)
        self.assertEqual(sum(stats['by_color']['black'].values()), 1)
        self.assertEqual(sum(stats['by_color']['white'].values()), 1)
        with self.assertRaises(ValueError):
            evaluate(UniformEvaluator(), games=3)


if __name__ == '__main__':
    unittest.main()
