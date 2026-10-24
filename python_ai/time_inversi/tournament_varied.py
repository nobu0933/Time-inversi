"""Round-robin match of four checkpoints on paired, unseen random openings."""

import argparse
from itertools import combinations
import json
from pathlib import Path
import time

import numpy as np
import torch

from .checkpoint import device_for, load
from .compare_varied import atomic_json, excluded_states, file_hash, starts, state_from
from .mixed_finetune import play
from .rules import BLACK


ITERATIONS = (257, 267, 292, 307)
PAIRS = tuple(combinations(ITERATIONS, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous-run', type=Path, required=True)
    parser.add_argument('--previous-match', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--openings', type=int, default=30)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--seed', type=int, default=72024)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if min(args.openings, args.simulations, args.threads) < 1:
        parser.error('Openings, simulations and threads must be positive')
    torch.set_num_threads(args.threads)
    checkpoints = {i: args.run / f'iteration-{i}.pt' for i in ITERATIONS}
    exclusions = (args.run / 'metrics.jsonl', args.run / 'evaluation-307.json',
                  args.previous_run / 'metrics.jsonl',
                  args.previous_run / 'evaluation-207.json', args.previous_match)
    config = dict(checkpoints={str(i): file_hash(p) for i, p in checkpoints.items()},
                  exclusions={str(p.resolve()): file_hash(p) for p in exclusions},
                  openings=args.openings, simulations=args.simulations, seed=args.seed,
                  pairs=[list(pair) for pair in PAIRS])
    excluded = set()
    excluded.update(excluded_states(exclusions[0], exclusions[1]))
    excluded.update(excluded_states(exclusions[2], exclusions[3]))
    previous = json.loads(args.previous_match.read_text(encoding='utf-8'))
    excluded.update(state_from(actions).key() for actions in previous['opening_actions'])
    selected = starts(args.seed, args.openings, excluded)
    action_lists = [actions for _, actions in selected]
    if args.output.exists():
        data = json.loads(args.output.read_text(encoding='utf-8'))
        if data['config'] != config or data['opening_actions'] != action_lists:
            raise ValueError('Saved tournament has different inputs or openings')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data = dict(config=config, opening_actions=action_lists, results=[])
        atomic_json(args.output, data)
    rows = data['results']
    planned = args.openings * len(PAIRS) * 2
    if len(rows) > planned:
        raise ValueError('Saved result count exceeds planned games')
    models = {i: load(path, device_for(args.device))[0] for i, path in checkpoints.items()}
    for game_index in range(len(rows), planned):
        opening_index, remainder = divmod(game_index, len(PAIRS) * 2)
        pair_index, color_index = divmod(remainder, 2)
        candidate_id, opponent_id = PAIRS[pair_index]
        color = BLACK if color_index == 0 else -BLACK
        state, actions = selected[opening_index]
        rng = np.random.default_rng(args.seed + 100000 + game_index)
        began = time.perf_counter()
        _, row = play(models[candidate_id], models[opponent_id], rng,
                      color, state, actions, args.simulations, 0, 512, False)
        row.update(game=game_index + 1, opening_index=opening_index,
                   candidate=candidate_id, opponent=opponent_id,
                   seconds=time.perf_counter() - began)
        rows.append(row)
        atomic_json(args.output, data)
        print(f'{game_index + 1}/{planned}: {candidate_id} vs {opponent_id} '
              f'({"black" if color == BLACK else "white"}) result {row["result"]}',
              flush=True)


if __name__ == '__main__':
    main()
