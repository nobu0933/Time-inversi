"""Compare the unsolved pretrained teacher with exact ordinary-Othello moves."""
import argparse
import json
from pathlib import Path
import numpy as np
from .ordinary_solver import solve
from .ordinary_teacher import Teacher, load_pretrained
from .oracle_teacher import OracleEvaluator, OracleTeacher
from .oracle_opening import OracleOpening
from .pretrain import candidate_states


def measure(teacher, positions, *, max_empty=10, max_nodes=1_000_000, seconds=10.0):
    result = {}
    for state, position, group in positions:
        if position.board.count(0) > max_empty:
            continue
        exact = solve(position, max_empty=max_empty, max_nodes=max_nodes, seconds=seconds)
        if exact is None:
            continue
        policy, _, source = teacher.label_with_source(position, state)
        if source == 'exact':
            raise ValueError('Benchmark teacher must have exact search disabled')
        chosen = int(np.argmax(policy))
        row = result.setdefault(group, dict(positions=0, optimal=0, total_regret=0,
                                            missed_wins=0, opening_book=0))
        row['positions'] += 1
        row['opening_book'] += int(source == 'opening_book')
        row['optimal'] += int(chosen in exact.best_actions)
        row['total_regret'] += exact.value - exact.action_values[chosen]
        row['missed_wins'] += int(exact.value > 0 and exact.action_values[chosen] <= 0)
    for row in result.values():
        row['optimal_rate'] = row['optimal'] / row['positions']
        row['mean_stone_regret'] = row['total_regret'] / row['positions']
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', required=True)
    p.add_argument('--teacher-type', choices=('auto', 'oracle', 'neural'), default='auto')
    p.add_argument('--ordinary-games', type=int, default=10)
    p.add_argument('--variant-games', type=int, default=10)
    p.add_argument('--max-empty', type=int, default=10)
    p.add_argument('--max-nodes', type=int, default=1_000_000)
    p.add_argument('--seconds', type=float, default=10.0)
    p.add_argument('--approximate-depth', type=int, default=3)
    p.add_argument('--approximate-seconds', type=float, default=2.0)
    p.add_argument('--seed', type=int, default=51235)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=('cpu', 'cuda', 'auto'), default='auto')
    args = p.parse_args(argv)
    import torch
    from .checkpoint import device_for
    torch.set_num_threads(args.threads)
    teacher_type = ('oracle' if args.teacher.endswith('.rs') else 'neural') if args.teacher_type == 'auto' else args.teacher_type
    settings = dict(exact_empty=0, approximate_depth=args.approximate_depth,
                    approximate_seconds=args.approximate_seconds)
    if teacher_type == 'oracle':
        folder = Path(args.teacher).resolve().parent
        book = OracleOpening(folder / 'oracle-opening-tree.dat',
                             folder / 'oracle-opening-data.dat')
        teacher = OracleTeacher(OracleEvaluator.from_file(args.teacher),
                                opening_book=book, **settings)
    else:
        teacher = Teacher(load_pretrained(args.teacher, device_for(args.device)), **settings)
    candidates = list(candidate_states(np.random.default_rng(args.seed),
                                       args.ordinary_games, args.variant_games))
    metrics = measure(teacher, candidates, max_empty=args.max_empty,
                      max_nodes=args.max_nodes, seconds=args.seconds)
    if not any(group['positions'] for group in metrics.values()):
        raise RuntimeError('No positions could be solved within the budget')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
