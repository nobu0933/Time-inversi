"""Read-only study of first-time corner choices and paired continuations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load
from .mcts import MCTS
from .rules import CORNERS, PASS, State, counts, legal_actions, outcome, step


def corner_options(state):
    legal = legal_actions(state)
    trigger = tuple(action for action in legal if action in CORNERS and action not in state.used_corners)
    other = tuple(action for action in legal if action not in trigger and action != PASS)
    return trigger, other


def choose_states(replay, phases, count, seed):
    rng = np.random.default_rng(seed)
    pools = defaultdict(list)
    seen = set()
    for row in replay['samples']:
        state = State.from_dict(row['state'])
        phase = len(state.used_corners)
        if phase not in phases or state.key() in seen:
            continue
        trigger, other = corner_options(state)
        if trigger and other:
            seen.add(state.key())
            pools[phase].append(state)
    selected = {}
    for phase in phases:
        if len(pools[phase]) < count:
            raise ValueError(f'Only {len(pools[phase])} eligible states at phase {phase}')
        indices = rng.choice(len(pools[phase]), count, replace=False)
        selected[phase] = [pools[phase][int(i)] for i in indices]
    return selected, {phase: len(pools[phase]) for phase in phases}


def balance_rollout_prefix(states, count):
    """Put equal numbers of black and white positions first for paired rollouts."""
    if count == 0:
        return states
    if count % 2:
        raise ValueError('Paired rollout count must be even for color balance')
    black = [state for state in states if state.player == 1]
    white = [state for state in states if state.player == -1]
    if min(len(black), len(white)) < count // 2:
        raise ValueError('Not enough positions of both colors for paired rollouts')
    prefix = black[:count // 2] + white[:count // 2]
    selected = {state.key() for state in prefix}
    return prefix + [state for state in states if state.key() not in selected]


def candidate_choice(model, state, simulations, seed):
    trigger, other = corner_options(state)
    policy = MCTS(model, np.random.default_rng(seed)).search(state, simulations, noise=False)
    # A legal move can have zero visits when the search budget is small.
    prior, _ = model.predict(state)
    ranked = sorted(legal_actions(state), key=lambda a: (policy[a], prior[a], -a), reverse=True)
    corner = max(trigger, key=lambda a: (policy[a], prior[a], -a))
    alternative = max(other, key=lambda a: (policy[a], prior[a], -a))
    return dict(chosen=int(ranked[0]), corner=int(corner), alternative=int(alternative),
                corner_mass=float(sum(policy[a] for a in trigger)),
                corner_visits=int(round(policy[corner] * simulations)),
                alternative_visits=int(round(policy[alternative] * simulations)))


def continuation(state, forced_action, candidate, opponent, simulations, seed, max_moves):
    color = state.player
    agents = {color: MCTS(candidate, np.random.default_rng(seed), max_depth=max_moves),
              -color: MCTS(opponent, np.random.default_rng(seed + 1), max_depth=max_moves)}
    state = step(state, forced_action)
    for turn in range(max_moves):
        if state.ended:
            black, white = counts(state)
            return dict(result=int(outcome(state, color)), black=black, white=white, turns=turn + 1)
        policy = agents[state.player].search(state, simulations, noise=False)
        # Reproducible tie-breaking, independent of how the paired branch diverges.
        action = int(np.flatnonzero(policy == policy.max())[0])
        state = step(state, action)
    raise RuntimeError(f'Continuation exceeded {max_moves} moves')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pro', required=True, type=Path, help='Full current PRO checkpoint')
    parser.add_argument('--mid', required=True, type=Path, help='Full MID checkpoint')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--phases', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--survey-per-phase', type=int, default=20)
    parser.add_argument('--rollout-per-phase', type=int, default=2)
    parser.add_argument('--root-simulations', type=int, default=64)
    parser.add_argument('--rollout-simulations', type=int, default=16)
    parser.add_argument('--max-moves', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    if (not args.phases or any(phase not in range(4) for phase in args.phases)
            or len(set(args.phases)) != len(args.phases)):
        parser.error('--phases must contain distinct values from 0 through 3')
    if (min(args.survey_per_phase, args.root_simulations, args.rollout_simulations,
            args.max_moves, args.threads) < 1 or not 0 <= args.rollout_per_phase <= args.survey_per_phase):
        parser.error('Invalid sample count or search budget')
    torch.set_num_threads(args.threads)
    pro, pro_data = load(args.pro, 'cpu')
    mid, mid_data = load(args.mid, 'cpu')
    if pro_data['network_config'] != mid_data['network_config']:
        parser.error('The checkpoints use different network architectures')
    selected, pool_sizes = choose_states(pro_data['replay'], args.phases,
                                         args.survey_per_phase, args.seed)
    for phase in args.phases:
        selected[phase] = balance_rollout_prefix(selected[phase], args.rollout_per_phase)
    report = dict(config=dict(phases=args.phases, survey_per_phase=args.survey_per_phase,
                              rollout_per_phase=args.rollout_per_phase,
                              root_simulations=args.root_simulations,
                              rollout_simulations=args.rollout_simulations,
                              seed=args.seed, pro_iteration=pro_data['iteration'],
                              mid_iteration=mid_data['iteration']),
                  eligible_states=pool_sizes, rows=[])
    for phase in args.phases:
        for index, state in enumerate(selected[phase]):
            root_seed = args.seed + phase * 10000 + index * 100
            choice = candidate_choice(pro, state, args.root_simulations, root_seed)
            row = dict(phase=phase, sample=index, direction=state.direction,
                       color=state.player, state=state.to_dict(), choice=choice)
            if index < args.rollout_per_phase:
                row['continuations'] = {}
                for name, opponent in [('MID', mid), ('PRO100', pro)]:
                    pair_seed = root_seed + (10 if name == 'MID' else 20)
                    row['continuations'][name] = {
                        'corner': continuation(state, choice['corner'], pro, opponent,
                                               args.rollout_simulations, pair_seed, args.max_moves),
                        'alternative': continuation(state, choice['alternative'], pro, opponent,
                                                    args.rollout_simulations, pair_seed, args.max_moves),
                    }
            report['rows'].append(row)
            print(f'Phase {phase}, position {index + 1}/{len(selected[phase])}: '
                  f'PRO {"corner" if choice["chosen"] in CORNERS and choice["chosen"] not in state.used_corners else "other"}',
                  flush=True)
    for phase in args.phases:
        rows = [row for row in report['rows'] if row['phase'] == phase]
        report.setdefault('summary', {})[str(phase)] = dict(
            surveyed=len(rows), corner_chosen=sum(row['choice']['chosen'] == row['choice']['corner'] for row in rows),
            mean_corner_mass=float(np.mean([row['choice']['corner_mass'] for row in rows])))
        for name in ('MID', 'PRO100'):
            pairs = [row['continuations'][name] for row in rows if 'continuations' in row]
            if pairs:
                report['summary'][str(phase)][name] = dict(
                    pairs=len(pairs), corner_score=sum((p['corner']['result'] + 1) / 2 for p in pairs),
                    alternative_score=sum((p['alternative']['result'] + 1) / 2 for p in pairs),
                    corner_better=sum(p['corner']['result'] > p['alternative']['result'] for p in pairs),
                    alternative_better=sum(p['corner']['result'] < p['alternative']['result'] for p in pairs),
                    tied=sum(p['corner']['result'] == p['alternative']['result'] for p in pairs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['summary'], ensure_ascii=False, indent=2), flush=True)
    print(f'Saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
