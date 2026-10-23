"""Fine-tune PRO with self-play plus games against frozen MID and PRO-100."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .buffer import ReplayBuffer
from .checkpoint import device_for, load, save
from .compare_models import compare
from .mcts import MCTS
from .rules import BLACK, PASS, create_game, legal_actions, outcome, step
from .selfplay import Sample, self_play
from .train import train_batch


def sample_policy(rng, policy):
    """Normalize MCTS visit fractions in float64 before NumPy's strict sum check."""
    probabilities = np.asarray(policy, dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(probabilities), p=probabilities))


def evaluation_due(iteration, base_iteration, interval):
    return interval > 0 and (iteration - base_iteration) % interval == 0


def frozen_game(candidate, opponent, rng, candidate_color, simulations=64,
                temperature_moves=12, max_moves=512):
    """Return samples from candidate turns only; the frozen model is not imitated."""
    if candidate_color not in (-1, 1):
        raise ValueError('Candidate color must be black or white')
    state = create_game()
    agents = {candidate_color: MCTS(candidate, rng, max_depth=max_moves),
              -candidate_color: MCTS(opponent, rng, max_depth=max_moves)}
    trajectory = []
    passes = reverse_turns = 0
    for turn in range(max_moves):
        ours = state.player == candidate_color
        policy = agents[state.player].search(state, simulations, noise=ours)
        if ours:
            trajectory.append((state, policy))
        if ours and turn < temperature_moves:
            action = sample_policy(rng, policy)
        else:
            action = int(rng.choice(np.flatnonzero(policy == policy.max())))
        if action not in legal_actions(state):
            raise RuntimeError('Search chose an illegal action')
        passes += action == PASS
        reverse_turns += state.direction == -1
        state = step(state, action)
        if state.ended:
            samples = [Sample(s, p, float(outcome(state, s.player))) for s, p in trajectory]
            return samples, dict(turns=turn + 1, samples=len(samples),
                                 result=outcome(state, candidate_color),
                                 color=candidate_color, passes=int(passes),
                                 reverse_turns=int(reverse_turns),
                                 corner_inversions=len(state.used_corners))
    raise RuntimeError(f'Frozen-opponent game exceeded {max_moves} turns')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, help='Full PRO-100 checkpoint; kept frozen as the anchor')
    p.add_argument('--mid', type=Path, help='Full MID checkpoint; kept frozen')
    p.add_argument('--resume', type=Path, help='Resume this fine-tuning run from latest.pt')
    p.add_argument('--output', type=Path, help='New run directory; never overwrite PRO-100')
    p.add_argument('--iterations', type=int, default=20, help='Additional iterations')
    p.add_argument('--self-play-games', type=int, default=6)
    p.add_argument('--mid-games', type=int, default=2)
    p.add_argument('--anchor-games', type=int, default=2)
    p.add_argument('--simulations', type=int, default=64)
    p.add_argument('--temperature-moves', type=int, default=12)
    p.add_argument('--max-moves', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--updates', type=int, default=100)
    p.add_argument('--learning-rate', type=float, default=0.0002)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-games', type=int, default=20)
    p.add_argument('--seed', type=int, default=20261022)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return p


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.resume:
        if args.base or args.mid:
            p.error('--resume uses its saved frozen opponents; omit --base and --mid')
        resume = args.resume.resolve()
        if not resume.is_file() or resume.name != 'latest.pt':
            p.error('--resume must point to this run\'s latest.pt')
        output = resume.parent
        if args.output and args.output.resolve() != output:
            p.error('--output must be the resume checkpoint directory')
    else:
        if not args.base or not args.mid or not args.output:
            p.error('A new run requires --base, --mid and --output')
        output = args.output.resolve()
        if output.joinpath('latest.pt').exists() or output.joinpath('pro100.pt').exists() or output.joinpath('mid.pt').exists():
            p.error('Output already contains training data; use --resume or a new directory')
        if not args.base.is_file() or not args.mid.is_file():
            p.error('Both full checkpoints must exist')
        if output == args.base.resolve().parent or output == args.mid.resolve().parent:
            p.error('Output must differ from both input checkpoint directories')
    if args.iterations < 1 or args.threads < 1:
        p.error('Iterations and threads must be positive')
    device = device_for(args.device)
    torch.set_num_threads(args.threads)

    if args.resume:
        candidate, data = load(resume, device)
        config = data['training_config']
        if config.get('mode') != 'frozen-opponent-finetune-v1':
            p.error('--resume requires a fine-tuning checkpoint')
        for name in ('self_play_games', 'mid_games', 'anchor_games', 'simulations',
                     'temperature_moves', 'max_moves', 'batch_size', 'updates',
                     'learning_rate', 'weight_decay', 'eval_every', 'eval_games', 'seed'):
            supplied = getattr(args, name)
            default = p.get_default(name)
            if supplied != default and supplied != config[name]:
                p.error(f'{name} cannot change on resume; saved value is {config[name]}')
            setattr(args, name, config[name])
        if data['iteration'] < config['base_iteration']:
            p.error('Fine-tuning checkpoint iteration precedes its base')
        replay = ReplayBuffer.from_dict(data['replay'])
        optimizer = torch.optim.AdamW(candidate.parameters())
        optimizer.load_state_dict(data['optimizer'])
        rng = np.random.default_rng()
        rng.bit_generator.state = data['numpy_rng']
        torch.set_rng_state(data['torch_rng'])
        if device.type == 'cuda' and len(data['cuda_rng']) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(data['cuda_rng'])
        start = data['iteration']
    else:
        candidate, base_data = load(args.base, device)
        _, mid_data = load(args.mid, device)
        if base_data['iteration'] != 100:
            p.error(f'--base must be the 100-iteration PRO checkpoint, got iteration {base_data["iteration"]}')
        if base_data['network_config'] != mid_data['network_config']:
            p.error('PRO-100 and MID network architectures must match')
        output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.base, output / 'pro100.pt')
        shutil.copy2(args.mid, output / 'mid.pt')
        config = dict(mode='frozen-opponent-finetune-v1', base_iteration=base_data['iteration'],
                      pro100_sha256=_digest(output / 'pro100.pt'),
                      mid_sha256=_digest(output / 'mid.pt'),
                      **{name: getattr(args, name) for name in
                         ('self_play_games', 'mid_games', 'anchor_games', 'simulations',
                          'temperature_moves', 'max_moves', 'batch_size', 'updates',
                          'learning_rate', 'weight_decay', 'eval_every', 'eval_games', 'seed')})
        replay = ReplayBuffer.from_dict(base_data['replay'])
        optimizer = torch.optim.AdamW(candidate.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        start = base_data['iteration']

    if args.self_play_games < 0 or args.mid_games < 0 or args.anchor_games < 0 or not sum(
            (args.self_play_games, args.mid_games, args.anchor_games)):
        p.error('At least one nonnegative game count is required')
    if args.mid_games % 2 or args.anchor_games % 2:
        p.error('Frozen-opponent game counts must be even for color balance')
    if min(args.simulations, args.max_moves, args.batch_size, args.updates) < 1:
        p.error('Simulation, move, batch and update counts must be positive')
    if args.batch_size > replay.samples.maxlen:
        p.error('Batch size exceeds replay capacity')
    if args.eval_every < 0 or args.eval_every and (args.eval_games < 2 or args.eval_games % 2):
        p.error('Evaluation interval/games are invalid')
    if not (np.isfinite(args.learning_rate) and args.learning_rate > 0 and
            np.isfinite(args.weight_decay) and args.weight_decay >= 0):
        p.error('Learning rate or weight decay is invalid')

    if _digest(output / 'pro100.pt') != config['pro100_sha256'] or _digest(output / 'mid.pt') != config['mid_sha256']:
        p.error('A frozen reference checkpoint changed')
    anchor, _ = load(output / 'pro100.pt', device)
    mid, _ = load(output / 'mid.pt', device)
    candidate.eval()
    if not args.resume:
        # Keep an iteration-100 recovery point before the first expensive game.
        save(output / 'latest.pt', candidate, optimizer, replay, rng, start, config)
    print(f'Device: {device}; output: {output}; starting iteration: {start}', flush=True)
    for iteration in range(start + 1, start + args.iterations + 1):
        began = time.perf_counter()
        schedule = [('self', None)] * args.self_play_games
        for name, count in (('mid', args.mid_games), ('pro100', args.anchor_games)):
            schedule.extend((name, BLACK if game % 2 == 0 else -BLACK) for game in range(count))
        rng.shuffle(schedule)
        records = []
        for game, (name, color) in enumerate(schedule, 1):
            if name == 'self':
                samples, final = self_play(candidate, rng, args.simulations,
                                           args.temperature_moves, args.max_moves)
                record = dict(type=name, turns=len(samples), samples=len(samples),
                              passes=int(sum(s.policy[PASS] > .999 for s in samples)),
                              reverse_turns=sum(s.state.direction == -1 for s in samples),
                              corner_inversions=len(final.used_corners))
            else:
                samples, record = frozen_game(candidate, mid if name == 'mid' else anchor,
                                              rng, color, args.simulations,
                                              args.temperature_moves, args.max_moves)
                record['type'] = name
            replay.add(samples)
            records.append(record)
            print(f'Iteration {iteration}, game {game}/{len(schedule)} ({name}): '
                  f'{record["turns"]} turns, {len(samples)} training samples', flush=True)
        losses = []
        if len(replay) >= args.batch_size:
            for update in range(args.updates):
                losses.append(train_batch(candidate, optimizer, replay.sample(args.batch_size, rng)))
                if (update + 1) % 25 == 0 or update + 1 == args.updates:
                    print(f'Iteration {iteration}, weight updates: {update + 1}/{args.updates}', flush=True)
        metrics = dict(iteration=iteration, buffer=len(replay), games=records,
                       updates=len(losses), training_samples=sum(r['samples'] for r in records))
        if losses:
            metrics.update({name: float(np.mean([row[name] for row in losses]))
                            for name in losses[0]})
        print(f'Iteration {iteration}: saving checkpoint', flush=True)
        save(output / 'latest.pt', candidate, optimizer, replay, rng, iteration, config)
        due = evaluation_due(iteration, config['base_iteration'], args.eval_every)
        if due:
            print(f'Iteration {iteration}: evaluating against MID and PRO-100', flush=True)
            metrics['evaluation'] = {
                'mid': compare(candidate, mid, args.eval_games, args.simulations,
                               args.seed + iteration * 1000, args.max_moves),
                'pro100': compare(candidate, anchor, args.eval_games, args.simulations,
                                  args.seed + iteration * 1000 + 500, args.max_moves),
            }
        if due:
            shutil.copy2(output / 'latest.pt', output / f'iteration-{iteration}.pt')
        metrics['seconds'] = time.perf_counter() - began
        with (output / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(metrics) + '\n')
        print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
