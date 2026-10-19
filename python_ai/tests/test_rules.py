import json
from pathlib import Path
import random
import subprocess
import shutil
import unittest
from dataclasses import replace
from time_inversi.rules import *


class RulesParityTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Optional JS parity test requires Node.js; Python training does not')
    def test_full_transition_parity_with_javascript(self):
        rng = random.Random(17)
        cases, expected = [], []
        steps = 0
        for manual in (False, True):
            for _ in range(30):
                state = create_game()
                initial = state.to_dict()
                commands, records = [], []
                for _ in range(260):
                    if state.ended:
                        break
                    if manual and rng.random() < .2:
                        command = 'invert'
                        turn = invert(state)
                    else:
                        action = rng.choice(legal_actions(state))
                        command = None if action == PASS else action
                        turn = advance(state, command)
                    commands.append(command)
                    records.append(turn.to_dict())
                    state = turn.state
                    steps += 1
                cases.append(dict(initial=initial, commands=commands))
                expected.append(records)
        board = [1] * 36
        board[0] = board[30] = 0
        board[1] = board[31] = -1
        fixtures = [(replace(create_game(), board=tuple(board), player=-1, used_corners=CORNERS), [None, 0, 'invert', None]),
                    (replace(create_game(), forward=(8, None), board=tuple(1 if i == 8 else x for i,x in enumerate(create_game().board)),
                             origins=tuple(1 if i == 8 else 0 for i in range(36))), ['invert']),
                    (create_game(), [None, -1, 36, 0, 8, 8])]
        for initial, commands in fixtures:
            state, records = initial, []
            for command in commands:
                turn = invert(state) if command == 'invert' else advance(state, command)
                records.append(turn.to_dict() if turn else None)
                if turn:
                    state = turn.state
            cases.append(dict(initial=initial.to_dict(), commands=commands))
            expected.append(records)
        process = subprocess.run(['node', str(Path(__file__).with_name('reference.cjs'))],
                                 input=json.dumps(cases), capture_output=True, text=True, check=True)
        actual = json.loads(process.stdout)
        for i, (wanted, got) in enumerate(zip(expected, actual)):
            with self.subTest(case=i):
                self.assertEqual(wanted, got)
        self.assertGreater(steps, 1000)

    def test_search_key_keeps_passes_but_ignores_visual_clock(self):
        state = create_game()
        self.assertEqual(state.key(), replace(state, clock_origin=-50, clock_anchor=-50).key())
        self.assertNotEqual(state.key(), replace(state, forward=(None,)).key())
        self.assertNotEqual(state.key(), replace(state, used_corners=(0,)).key())

    def test_immutable_input_and_forced_corner(self):
        board = [0] * 36
        board[2] = board[14] = BLACK
        board[1] = board[7] = WHITE
        state = replace(create_game(), board=tuple(board))
        before = state.to_dict()
        turn = advance(state, 0)
        self.assertEqual(state.to_dict(), before)
        self.assertEqual(turn.state.direction, -1)
        self.assertEqual(turn.state.used_corners, (0,))
        self.assertEqual(turn.corner_inversion.removed, 0)
        self.assertEqual(turn.state.board[0], 0)


if __name__ == '__main__':
    unittest.main()
