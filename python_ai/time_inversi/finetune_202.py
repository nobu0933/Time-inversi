"""Fine-tune iteration 202 against MID, PRO100, 122 and frozen 207."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .buffer import ReplayBuffer
from .checkpoint import device_for, load, save
from .mixed_finetune import (copy_verified, digest, opening, play, summarize,
                             training_start, write_json, write_rows)
from .rules import BLACK, create_game
from .train import train_batch


MODE = 'mixed-opponent-202-v1'
BASE_ITERATION = 202
OPPONENTS = ('mid', 'pro100', 'pro122', 'pro207')
SETTINGS = ('self_play_games', 'mid_games', 'pro100_games', 'pro122_games',
            'pro207_games', 'opening_probability', 'opening_min_moves',
            'opening_max_moves', 'simulations', 'temperature_moves', 'max_moves',
            'batch_size', 'updates', 'buffer_size', 'learning_rate',
            'weight_decay', 'eval_every', 'eval_standard_games', 'eval_openings',
            'seed', 'eval_seed')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, help='Full iteration-202 checkpoint')
    p.add_argument('--mid', type=Path, help='Frozen MID checkpoint')
    p.add_argument('--pro100', type=Path, help='Frozen PRO100 checkpoint')
    p.add_argument('--pro122', type=Path, help='Frozen iteration-122 checkpoint')
    p.add_argument('--pro207', type=Path, help='Frozen iteration-207 checkpoint')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-iteration', type=int, default=307)
    p.add_argument('--max-hours', type=float, default=12.0,
                   help='Maximum accumulated completed-iteration compute hours')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--self-play-games', type=int, default=8)
    for name in OPPONENTS:
        p.add_argument(f'--{name}-games', type=int, default=4)
    p.add_argument('--opening-probability', type=float, default=0.75)
    p.add_argument('--opening-min-moves', type=int, default=4)
    p.add_argument('--opening-max-moves', type=int, default=8)
    p.add_argument('--simulations', type=int, default=64)
    p.add_argument('--temperature-moves', type=int, default=12)
    p.add_argument('--max-moves', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--updates', type=int, default=150)
    p.add_argument('--buffer-size', type=int, default=50000)
    p.add_argument('--learning-rate', type=float, default=0.00005)
    p.add_argument('--weight-decay', type=float, default=0.0001)
    p.add_argument('--eval-every', type=int, default=5)
    p.add_argument('--eval-standard-games', type=int, default=4)
    p.add_argument('--eval-openings', type=int, default=8)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--eval-seed', type=int, default=20261103)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    return p


def validate(args, p):
    if args.target_iteration < BASE_ITERATION or args.threads < 1:
        p.error('Target must be at least 202 and threads must be positive')
    if not np.isfinite(args.max_hours) or args.max_hours <= 0:
        p.error('--max-hours must be positive and finite')
    if args.self_play_games < 0 or any(getattr(args, name + '_games') < 2 or
                                      getattr(args, name + '_games') % 2 for name in OPPONENTS):
        p.error('Every fixed opponent needs a positive even game count')
    if not 0 < args.opening_probability <= 1 or not 1 <= args.opening_min_moves <= args.opening_max_moves:
        p.error('Opening probability or move range is invalid')
    if min(args.simulations, args.max_moves, args.batch_size, args.updates,
           args.buffer_size, args.eval_standard_games, args.eval_openings) < 1:
        p.error('Simulation, update, buffer and evaluation counts must be positive')
    if args.opening_max_moves >= args.max_moves or args.batch_size > args.buffer_size:
        p.error('Opening length or batch size exceeds its limit')
    if args.eval_standard_games % 2 or args.eval_every < 0 or args.temperature_moves < 0:
        p.error('Evaluation balance or training settings are invalid')
    if not (np.isfinite(args.learning_rate) and args.learning_rate > 0 and
            np.isfinite(args.weight_decay) and args.weight_decay >= 0):
        p.error('Learning rate or weight decay is invalid')


def evaluation_starts(config):
    rng = np.random.default_rng(config['eval_seed'])
    starts = []
    seen = set()
    while len(starts) < config['eval_openings']:
        state, actions = opening(rng, config['opening_min_moves'],
                                 config['opening_max_moves'])
        if state.key() not in seen:
            seen.add(state.key())
            starts.append((state, actions))
    return starts


def evaluate(candidate, opponents, config, varied):
    results = {}
    for opponent_index, name in enumerate(OPPONENTS):
        opponent = opponents[name]
        standard = []
        for game in range(config['eval_standard_games']):
            color = BLACK if game % 2 == 0 else -BLACK
            rng = np.random.default_rng(config['eval_seed'] + opponent_index * 100000 + game)
            _, row = play(candidate, opponent, rng, color, create_game(), [],
                          config['simulations'], config['temperature_moves'],
                          config['max_moves'], False)
            standard.append(row)
        varied_rows = []
        for index, (state, actions) in enumerate(varied):
            for color_index, color in enumerate((BLACK, -BLACK)):
                rng = np.random.default_rng(config['eval_seed'] + opponent_index * 100000
                                            + 10000 + index * 2 + color_index)
                _, row = play(candidate, opponent, rng, color, state, actions,
                              config['simulations'], config['temperature_moves'],
                              config['max_moves'], False)
                varied_rows.append(row)
        results[name] = dict(standard=summarize(standard), varied=summarize(varied_rows))
    return results


def prepare(args, output, device):
    paths = {'base202': args.base, **{name: getattr(args, name) for name in OPPONENTS}}
    if any(path is None or not path.is_file() for path in paths.values()):
        raise ValueError('New run requires --base, --mid, --pro100, --pro122 and --pro207')
    if any(output == path.resolve().parent for path in paths.values()):
        raise ValueError('Output must differ from input checkpoint directories')
    candidate, base_data = load(args.base, device)
    if base_data['iteration'] != BASE_ITERATION:
        raise ValueError(f'--base must be iteration 202, got {base_data["iteration"]}')
    expected = {'mid': 100, 'pro100': 100, 'pro122': 122, 'pro207': 207}
    for name in OPPONENTS:
        _, data = load(paths[name], 'cpu')
        if data['iteration'] != expected[name] or data['network_config'] != base_data['network_config']:
            raise ValueError(f'{name} has the wrong iteration or network architecture')
    hashes = {name: digest(path) for name, path in paths.items()}
    output.mkdir(parents=True, exist_ok=True)
    allowed = ({name + '.pt' for name in paths} |
               {name + '.pt.tmp' for name in paths} | {'latest.pt.tmp'})
    unexpected = {item.name for item in output.iterdir()} - allowed
    if unexpected:
        raise ValueError(f'Output is not empty: {output}')
    for name, path in paths.items():
        copy_verified(path, output / (name + '.pt'), hashes[name])
    config = dict(mode=MODE, base_iteration=BASE_ITERATION, frozen_sha256=hashes,
                  **{name: getattr(args, name) for name in SETTINGS})
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    replay = ReplayBuffer(args.buffer_size)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = np.random.default_rng(args.seed)
    save(output / 'latest.pt', candidate, optimizer, replay, rng, BASE_ITERATION, config)
    print(f'Prepared {output / "latest.pt"} from iteration 202; no games played.', flush=True)


def load_run(args, output, device):
    candidate, data = load(output / 'latest.pt', device)
    config = data['training_config']
    if config.get('mode') != MODE or config.get('base_iteration') != BASE_ITERATION:
        raise ValueError('Output is not an iteration-202 fine-tuning run')
    for name in SETTINGS:
        if getattr(args, name) != config[name]:
            raise ValueError(f'{name} differs from the saved run; use the same settings')
    paths = {'base202': args.base, **{name: getattr(args, name) for name in OPPONENTS}}
    for name, expected in config['frozen_sha256'].items():
        frozen = output / (name + '.pt')
        if not frozen.is_file() or digest(frozen) != expected:
            raise ValueError(f'Frozen checkpoint changed or is missing: {frozen}')
        if paths[name] is not None and digest(paths[name]) != expected:
            raise ValueError(f'Input checkpoint changed: {paths[name]}')
    if data['iteration'] < BASE_ITERATION:
        raise ValueError('Saved iteration precedes the base')
    replay = ReplayBuffer.from_dict(data['replay'])
    optimizer = torch.optim.AdamW(candidate.parameters())
    optimizer.load_state_dict(data['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = data['numpy_rng']
    torch.set_rng_state(data['torch_rng'])
    if device.type == 'cuda' and len(data['cuda_rng']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(data['cuda_rng'])
    opponents = {name: load(output / (name + '.pt'), device)[0] for name in OPPONENTS}
    return candidate, opponents, optimizer, replay, rng, config, data['iteration']


def finish_iteration(output, iteration, candidate, opponents, config, varied):
    metrics_path = output / 'metrics.jsonl'
    rows = ([json.loads(line) for line in metrics_path.read_text(encoding='utf-8').splitlines()]
            if metrics_path.exists() else [])
    expected = list(range(BASE_ITERATION + 1, iteration + 1))
    recorded = [row['iteration'] for row in rows]
    if recorded == expected:
        (output / 'pending.json').unlink(missing_ok=True)
        return rows
    if recorded != expected[:-1]:
        raise ValueError('Metrics log does not match the saved checkpoint')
    pending_path = output / 'pending.json'
    if not pending_path.is_file():
        raise ValueError('Saved checkpoint has no pending training metrics')
    metrics = json.loads(pending_path.read_text(encoding='utf-8'))
    if metrics['iteration'] != iteration:
        raise ValueError('Pending metrics do not match the saved checkpoint')
    if config['eval_every'] and (iteration - BASE_ITERATION) % config['eval_every'] == 0:
        evaluation_path = output / f'evaluation-{iteration}.json'
        if evaluation_path.is_file():
            evaluation = json.loads(evaluation_path.read_text(encoding='utf-8'))
        else:
            print(f'Iteration {iteration}: evaluating four frozen opponents', flush=True)
            began = time.perf_counter()
            evaluation = dict(results=evaluate(candidate, opponents, config, varied),
                              seconds=time.perf_counter() - began)
            write_json(evaluation_path, evaluation)
        metrics['evaluation'] = {name: {mode: {key: value for key, value in block.items()
                                               if key != 'results'}
                                        for mode, block in evaluation['results'][name].items()}
                                 for name in OPPONENTS}
        metrics['evaluation_seconds'] = evaluation['seconds']
        snapshot = output / f'iteration-{iteration}.pt'
        if not snapshot.exists():
            temporary = snapshot.with_suffix('.pt.tmp')
            shutil.copyfile(output / 'latest.pt', temporary)
            temporary.replace(snapshot)
    metrics['seconds'] = metrics['training_seconds'] + metrics.get('evaluation_seconds', 0)
    write_rows(metrics_path, rows + [metrics])
    pending_path.unlink(missing_ok=True)
    return rows + [metrics]


def estimated_next_seconds(rows, iteration, config):
    training = np.mean([r['training_seconds'] for r in rows[-5:]]) if rows else 280.0
    evaluations = [r['evaluation_seconds'] for r in rows if 'evaluation_seconds' in r]
    evaluation = np.mean(evaluations[-3:]) if evaluations else 720.0
    due = config['eval_every'] and (iteration - BASE_ITERATION) % config['eval_every'] == 0
    return float(training + (evaluation if due else 0))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    validate(args, p)
    output = args.output.resolve()
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    if not (output / 'latest.pt').exists():
        prepare(args, output, device)
    candidate, opponents, optimizer, replay, rng, config, current = load_run(args, output, device)
    if current > args.target_iteration:
        p.error(f'Saved iteration {current} exceeds target {args.target_iteration}')
    varied = evaluation_starts(config)
    print(f'Current: {current}; target: {args.target_iteration}; '
          f'remaining: {args.target_iteration - current}', flush=True)
    if args.prepare_only:
        return
    pending = output / 'pending.json'
    if pending.is_file() and json.loads(pending.read_text(encoding='utf-8'))['iteration'] == current + 1:
        pending.unlink()  # No checkpoint was saved for this incomplete iteration.
    rows = (finish_iteration(output, current, candidate, opponents, config, varied)
            if current > BASE_ITERATION else [])
    for iteration in range(current + 1, args.target_iteration + 1):
        elapsed = sum(row['seconds'] for row in rows)
        estimate = estimated_next_seconds(rows, iteration, config)
        remaining = args.max_hours * 3600 - elapsed
        if remaining < estimate:
            print(f'Compute budget reached: {elapsed / 3600:.2f}/{args.max_hours:.2f} '
                  f'hours; next iteration estimated {estimate / 60:.1f} minutes. '
                  f'Stopped after iteration {iteration - 1}.', flush=True)
            break
        started = time.perf_counter()
        schedule = [('self', None)] * config['self_play_games']
        for name in OPPONENTS:
            schedule.extend((name, BLACK if game % 2 == 0 else -BLACK)
                            for game in range(config[name + '_games']))
        rng.shuffle(schedule)
        used_openings = {state.key() for state, _ in varied}
        records = []
        for game_number, (name, color) in enumerate(schedule, 1):
            state, actions = training_start(rng, config, used_openings)
            samples, record = play(candidate, opponents.get(name), rng, color, state, actions,
                                   config['simulations'], config['temperature_moves'],
                                   config['max_moves'], True)
            replay.add(samples)
            record['type'] = name
            records.append(record)
            print(f'Iteration {iteration}, game {game_number}/{len(schedule)} ({name}, '
                  f'opening {len(actions)}): {record["turns"]} turns, '
                  f'{len(samples)} samples', flush=True)
        losses = []
        if len(replay) >= config['batch_size']:
            for update in range(config['updates']):
                losses.append(train_batch(candidate, optimizer,
                                          replay.sample(config['batch_size'], rng)))
                if (update + 1) % 25 == 0 or update + 1 == config['updates']:
                    print(f'Iteration {iteration}, updates {update + 1}/{config["updates"]}',
                          flush=True)
        metrics = dict(iteration=iteration, buffer=len(replay), games=records,
                       training_samples=sum(row['samples'] for row in records),
                       updates=len(losses), training_seconds=time.perf_counter() - started)
        if losses:
            metrics.update({name: float(np.mean([row[name] for row in losses]))
                            for name in losses[0]})
        write_json(pending, metrics)
        save(output / 'latest.pt', candidate, optimizer, replay, rng, iteration, config)
        rows = finish_iteration(output, iteration, candidate, opponents, config, varied)
        print(f'Iteration {iteration} complete; compute {sum(r["seconds"] for r in rows)/3600:.2f}h',
              flush=True)


if __name__ == '__main__':
    main()
