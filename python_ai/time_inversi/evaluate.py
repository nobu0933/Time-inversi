"""Balanced evaluation: each opponent plays equally often as black and white."""
import argparse
import json
import numpy as np
from .baselines import greedy_action, random_action
from .mcts import MCTS
from .rules import BLACK, WHITE, counts, create_game, outcome, step


def evaluate(evaluator, games=20, simulations=64, opponent='greedy', seed=1234, max_moves=512):
    if games < 2 or games % 2:
        raise ValueError('Evaluation games must be a positive even number (at least 2)')
    if opponent not in ('random', 'greedy'):
        raise ValueError('Unknown opponent')
    rng = np.random.default_rng(seed)
    results = []
    for number in range(games):
        color = BLACK if number % 2 == 0 else WHITE
        state = create_game()
        search = MCTS(evaluator, rng, max_depth=max_moves)
        for turn in range(max_moves):
            if state.player == color:
                policy = search.search(state, simulations, noise=False)
                action = int(rng.choice(np.flatnonzero(policy == policy.max())))
            else:
                action = (random_action if opponent == 'random' else greedy_action)(state, rng)
            state = step(state, action)
            if state.ended:
                results.append(dict(color=color, result=outcome(state, color),
                                    black=counts(state)[0], white=counts(state)[1], turns=turn + 1))
                break
        else:
            raise RuntimeError('Evaluation exceeded max_moves; not counted as a draw')
    wins = sum(x['result'] == 1 for x in results)
    draws = sum(x['result'] == 0 for x in results)
    return dict(opponent=opponent, games=games, wins=wins, draws=draws, losses=games-wins-draws,
                win_rate=wins/games, score_rate=(wins + draws/2)/games,
                by_color={name: {label: sum(x['color'] == color and x['result'] == value for x in results)
                                 for label, value in [('wins', 1), ('draws', 0), ('losses', -1)]}
                          for name, color in [('black', BLACK), ('white', WHITE)]}, results=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--games', type=int, default=20)
    parser.add_argument('--simulations', type=int, default=64)
    parser.add_argument('--opponent', choices=['random', 'greedy'], default='greedy')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max-moves', type=int, default=512)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    import torch
    from .checkpoint import load, device_for
    torch.set_num_threads(2)
    model, _ = load(args.checkpoint, device_for(args.device))
    print(json.dumps(evaluate(model, args.games, args.simulations, args.opponent, args.seed, args.max_moves), indent=2))


if __name__ == '__main__':
    main()
