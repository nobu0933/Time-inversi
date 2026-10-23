from pathlib import Path
from dataclasses import replace
from functools import lru_cache
import tempfile
import unittest
import numpy as np
import torch
from time_inversi.ordinary_othello import Position, legal, play, terminal
from time_inversi.ordinary_solver import SearchLimit, solve
from time_inversi.ordinary_teacher import OthelloNNet, Teacher, load_pretrained
from time_inversi.oracle_teacher import OracleEvaluator, OracleTeacher, PATTERN_IDS
from time_inversi.oracle_opening import OracleOpening
from time_inversi.pretrain import ordinary_state, candidate_states, build_samples, symmetry_sample
from time_inversi.rules import PASS, create_game, legal_actions, step
from time_inversi.selfplay import Sample


class OrdinaryTeacherTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_standard_rules_do_not_invert_at_corner(self):
        rng = np.random.default_rng(6)
        candidates = list(candidate_states(rng, 2, 0))
        self.assertTrue(candidates)
        self.assertTrue(all(source == 'ordinary' and state.direction == 1
                            for state, _, source in candidates))
        self.assertTrue(any(state.used_corners for state, _, _ in candidates))
        for state, position, _ in candidates:
            self.assertEqual(set(legal_actions(state)), set(legal(position)))

    def test_exact_from_unreachable_board_and_pass(self):
        board = [1] * 36
        board[1], board[2] = -1, 0
        black = Position(tuple(board), 1)
        self.assertEqual(legal(black), (2,))
        solution = solve(black, max_empty=1)
        self.assertEqual(solution.best_actions, (2,))
        self.assertEqual(solution.value, 36)
        white = Position(tuple(board), -1)
        self.assertEqual(legal(white), (PASS,))
        self.assertFalse(terminal(white))
        self.assertEqual(solve(white, max_empty=1).action_values[PASS], -36)
        self.assertTrue(terminal(play(white, PASS)) is False)
        self.assertIsNone(solve(white, max_empty=0))

    def test_pretrained_format_and_exact_label(self):
        model = OthelloNNet(8).eval()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'teacher.pth.tar'
            torch.save({'state_dict': model.state_dict()}, path)
            restored = load_pretrained(path)
            self.assertEqual(restored.conv1.out_channels, 8)
            start = Position(create_game().board, 1)
            logits, value = Teacher(restored).network(start)
            self.assertEqual(logits.shape, (37,))
            self.assertTrue(np.isfinite(logits).all())
            self.assertTrue(-1 <= value <= 1)
            approximate = Teacher(restored, exact_empty=0).label(start)
            self.assertFalse(approximate[2])
            self.assertAlmostEqual(float(approximate[0].sum()), 1, places=5)
            self.assertEqual(set(np.flatnonzero(approximate[0])), set(legal(start)))
            timed_out = Teacher(restored, exact_empty=0,
                                approximate_seconds=1e-10).label(start)
            self.assertFalse(timed_out[2])
            self.assertAlmostEqual(float(timed_out[0].sum()), 1, places=5)

    def test_distillation_labels_obey_time_inversi_legality(self):
        fake = Teacher(OthelloNNet(8).eval(), exact_empty=1)
        samples, counts = build_samples(fake, np.random.default_rng(3), 1, 1, 4)
        self.assertEqual(len(samples), 4)
        self.assertEqual(sum(sum(row.values()) for row in counts.values()), 4)
        for sample in samples:
            self.assertAlmostEqual(float(sample.policy.sum()), 1, places=5)
            self.assertTrue(set(np.flatnonzero(sample.policy)) <= set(legal_actions(sample.state)))

    def test_ordinary_teacher_sample_share(self):
        class UniformTeacher:
            exact_empty = 0

            def label_with_source(self, position, state):
                policy = np.zeros(37, dtype=np.float32)
                actions = legal(position)
                policy[list(actions)] = 1 / len(actions)
                return policy, 0.0, 'approximate'

        teacher = UniformTeacher()
        samples, counts = build_samples(teacher, np.random.default_rng(92),
                                        2, 2, 10, ordinary_share=0.8)
        self.assertEqual(len(samples), 10)
        self.assertEqual(sum(counts['ordinary'].values()), 8)
        self.assertEqual(sum(counts['time_inversi'].values()), 2)
        with self.assertRaises(ValueError):
            build_samples(teacher, np.random.default_rng(92), 2, 2, 10,
                          ordinary_share=1.1)

    def test_symmetries_preserve_actions_and_histories(self):
        rng = np.random.default_rng(47)
        states = [state for state, _, _ in candidate_states(rng, 1, 2)][::8]
        self.assertTrue(any(state.reverse for state in states))
        for state in states:
            policy = np.zeros(37, dtype=np.float32)
            for action in legal_actions(state):
                policy[action] = 1 / len(legal_actions(state))
            sample = Sample(state, policy, 0.25)
            for reflected in (False, True):
                for rotation in range(4):
                    transformed = symmetry_sample(sample, rotation, reflected)
                    self.assertEqual(set(np.flatnonzero(transformed.policy)),
                                     set(legal_actions(transformed.state)))
                    self.assertEqual(transformed.state.board.count(0), state.board.count(0))
                    self.assertEqual(transformed.state.origins.count(1), state.origins.count(1))
                    self.assertEqual(transformed.state.direction, state.direction)
                    self.assertEqual(transformed.value, sample.value)
                    for action in legal_actions(state):
                        one_hot = np.zeros(37, dtype=np.float32)
                        one_hot[action] = 1
                        translated = symmetry_sample(Sample(state, one_hot, 0),
                                                     rotation, reflected)
                        new_action = int(np.argmax(translated.policy))
                        actual = step(transformed.state, new_action)
                        expected = symmetry_sample(
                            Sample(step(state, action), one_hot, 0),
                            rotation, reflected).state
                        self.assertEqual(actual.key(), expected.key())

    def test_exact_search_against_independent_minimax_on_arbitrary_boards(self):
        rng = np.random.default_rng(903)

        @lru_cache(None)
        def reference(position):
            if terminal(position):
                return position.board.count(position.player) - position.board.count(-position.player)
            return max(-reference(play(position, action)) for action in legal(position))

        for empty in (3, 7):
            checked = 0
            for _ in range(30):
                board = rng.choice([-1, 1], size=36).tolist()
                for index in rng.choice(36, size=empty, replace=False):
                    board[int(index)] = 0
                position = Position(tuple(board), int(rng.choice([-1, 1])))
                if terminal(position):
                    continue
                expected = {action: -reference(play(position, action)) for action in legal(position)}
                result = solve(position, max_empty=empty, max_nodes=50000, seconds=5)
                self.assertIsNotNone(result)
                self.assertEqual(result.action_values, expected)
                checked += 1
            self.assertGreaterEqual(checked, 20)

    def test_real_bundled_teacher_checkpoint(self):
        path = Path(__file__).resolve().parents[1] / 'pretrained' / '6x6_153checkpoints_best.pth.tar'
        if not path.is_file():
            self.skipTest('Optional legacy teacher checkpoint is not included')
        model = load_pretrained(path)
        position = Position(create_game().board, 1)
        policy, value, exact = Teacher(model, exact_empty=0).label(position)
        self.assertFalse(exact)
        self.assertAlmostEqual(float(policy.sum()), 1, places=5)
        self.assertEqual(set(np.flatnonzero(policy)), set(legal(position)))
        self.assertTrue(np.isfinite(value))

    def test_oracle_coefficients_on_arbitrary_and_reflected_boards(self):
        path = Path(__file__).resolve().parents[1] / 'pretrained' / 'oracle-weight.rs'
        self.assertTrue(path.is_file())
        evaluator = OracleEvaluator.from_file(path)
        self.assertEqual(len(set(PATTERN_IDS)), 3321)
        teacher = OracleTeacher(evaluator, exact_empty=0, approximate_depth=2)
        rng = np.random.default_rng(219)
        for empty in (4, 18, 20):
            board = rng.choice([-1, 1], size=36).tolist()
            for index in rng.choice(36, size=empty, replace=False):
                board[int(index)] = 0
            position = Position(tuple(board), 1)
            reflected = Position(tuple(board[row * 6 + 5 - col]
                                       for row in range(6) for col in range(6)), 1)
            self.assertAlmostEqual(evaluator.raw_score(position),
                                   evaluator.raw_score(reflected), places=8)
            if not terminal(position):
                policy, value, exact = teacher.label(position)
                self.assertFalse(exact)
                self.assertAlmostEqual(float(policy.sum()), 1, places=5)
                self.assertEqual(set(np.flatnonzero(policy)), set(legal(position)))
                self.assertTrue(np.isfinite(value))
        samples, counts = build_samples(teacher, np.random.default_rng(8), 1, 1, 6)
        self.assertEqual(len(samples), 6)
        self.assertEqual(sum(sum(row.values()) for row in counts.values()), 6)
        for sample in samples:
            self.assertTrue(set(np.flatnonzero(sample.policy)) <=
                            set(legal_actions(sample.state)))

    def test_oracle_opening_tree_and_teacher_fallback(self):
        root = Path(__file__).resolve().parents[1] / 'pretrained'
        book = OracleOpening(root / 'oracle-opening-tree.dat',
                             root / 'oracle-opening-data.dat')
        teacher = OracleTeacher(OracleEvaluator.from_file(root / 'oracle-weight.rs'),
                                exact_empty=0, approximate_depth=1,
                                opening_book=book)
        start = Position(create_game().board, 1)
        for first, reply in ((8, 19), (13, 9), (22, 26), (27, 16)):
            state = ordinary_state(play(start, first), [first])
            result = book.lookup(state, Position(state.board, state.player))
            self.assertEqual(result.best_actions, (reply,))
            policy, _, source = teacher.label_with_source(
                Position(state.board, state.player), state)
            self.assertEqual(source, 'opening_book')
            self.assertEqual(int(np.argmax(policy)), reply)
        self.assertEqual(set(book.lookup(ordinary_state(start, []), start).best_actions),
                         set(legal(start)))
        canonical = play(play(start, 22), 26)
        self.assertEqual(book.lookup(ordinary_state(canonical, [22, 26]),
                                     canonical).action_values,
                         {7: -14, 13: -6, 19: -4, 25: -18, 31: -10})
        # Oracle only certifies an unchanged ordinary-game prefix.
        inverted = replace(step(create_game(), 8), direction=-1)
        self.assertIsNone(book.lookup(inverted, Position(inverted.board, inverted.player)))
        altered = ordinary_state(start, [8])
        self.assertIsNone(book.lookup(altered, start))
        position = play(start, 22)
        outside = play(position, 16)
        self.assertIsNone(book.lookup(ordinary_state(outside, [22, 16]), outside))
        self.assertEqual(teacher.label_with_source(outside,
                         ordinary_state(outside, [22, 16]))[2], 'approximate')

    def test_timeout_keeps_last_complete_approximate_depth(self):
        torch.manual_seed(17)
        model = OthelloNNet(8).eval()
        position = Position(create_game().board, 1)
        shallow = Teacher(model, exact_empty=0, approximate_depth=1).label(position)
        deeper = Teacher(model, exact_empty=0, approximate_depth=3)
        leaf = deeper._leaf

        def fail_at_depth_two(state, depth, deadline):
            if depth >= 1:
                raise SearchLimit
            return leaf(state, depth, deadline)

        deeper._leaf = fail_at_depth_two
        fallback = deeper.label(position)
        np.testing.assert_allclose(fallback[0], shallow[0], atol=1e-7)
        self.assertAlmostEqual(fallback[1], shallow[1])
        self.assertFalse(fallback[2])


if __name__ == '__main__':
    unittest.main()
