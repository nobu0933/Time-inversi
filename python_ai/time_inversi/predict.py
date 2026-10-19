"""Choose a legal action from a browser-format state JSON using a checkpoint."""
import argparse
import json
from pathlib import Path
import numpy as np
from .checkpoint import device_for, load
from .mcts import MCTS
from .rules import PASS, State


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--simulations', type=int, default=64)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    state = State.from_dict(json.loads(args.state.read_text(encoding='utf-8-sig')))
    model, _ = load(args.checkpoint, device_for(args.device))
    rng = np.random.default_rng(args.seed)
    policy = MCTS(model, rng).search(state, args.simulations)
    action = int(rng.choice(np.flatnonzero(policy == policy.max())))
    print(json.dumps(dict(action=action, index=None if action == PASS else action, policy=policy.tolist())))


if __name__ == '__main__':
    main()
