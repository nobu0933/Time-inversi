"""Continue the iteration-102 PRO candidate to a fixed target iteration."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .buffer import ReplayBuffer
from .checkpoint import device_for, load, save
from .compare_models import compare
from .finetune import evaluation_due, main as train


SOURCE_ITERATION = 102
DEFAULT_TARGET = 172
RUN_TAG = 'pro-long-from-102-v1'


def digest(path):
    checksum = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def copy_verified(source, destination, expected):
    if digest(source) != expected:
        raise ValueError(f'Frozen checkpoint does not match source: {source}')
    if destination.exists():
        if digest(destination) != expected:
            raise ValueError(f'Frozen checkpoint differs in output: {destination}')
        return
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    shutil.copyfile(source, temporary)
    if digest(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise ValueError(f'Copied frozen checkpoint differs: {destination}')
    temporary.replace(destination)


def prepare(source, output, seed, device):
    """Create the branch checkpoint without playing any training games."""
    model, data = load(source, device)
    original = data['training_config']
    if data['iteration'] != SOURCE_ITERATION or original.get('mode') != 'frozen-opponent-finetune-v1':
        raise ValueError('Source must be the full fine-tuning checkpoint at iteration 102')
    if original.get('base_iteration') != 100:
        raise ValueError('Source must descend from PRO-100')
    if not data.get('optimizer') or not data.get('replay') or not data.get('numpy_rng'):
        raise ValueError('Source lacks the optimizer, replay buffer or random state')
    source_hash = digest(source)
    config = dict(original)
    config.update(base_iteration=SOURCE_ITERATION, eval_every=5, eval_games=10,
                  seed=seed, long_run_tag=RUN_TAG,
                  source_checkpoint_sha256=source_hash, source_iteration=SOURCE_ITERATION)
    references = (('pro100.pt', 'pro100_sha256'), ('mid.pt', 'mid_sha256'))
    for filename, key in references:
        reference = source.parent / filename
        if not reference.is_file() or digest(reference) != original[key]:
            raise ValueError(f'Missing or changed frozen reference: {reference}')
        reference_model, reference_data = load(reference, 'cpu')
        if reference_data['network_config'] != data['network_config']:
            raise ValueError(f'Frozen reference has a different network: {reference}')
        del reference_model
    output.mkdir(parents=True, exist_ok=True)
    unexpected = {item.name for item in output.iterdir()} - {'pro100.pt', 'mid.pt', 'pro100.pt.tmp', 'mid.pt.tmp'}
    if unexpected:
        raise ValueError(f'Output directory is not empty: {output}')
    for filename, key in references:
        copy_verified(source.parent / filename, output / filename, original[key])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    optimizer.load_state_dict(data['optimizer'])
    replay = ReplayBuffer.from_dict(data['replay'])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    save(output / 'latest.pt', model, optimizer, replay, rng,
         SOURCE_ITERATION, config)
    print(f'Prepared {output / "latest.pt"} from iteration 102; no games played.', flush=True)


def recover_pending_evaluation(output, iteration, device):
    """Finish an evaluation interrupted after the training checkpoint was saved."""
    candidate, data = load(output / 'latest.pt', device)
    config = data['training_config']
    if iteration == SOURCE_ITERATION:
        return
    metrics_path = output / 'metrics.jsonl'
    rows = []
    if metrics_path.is_file():
        lines = metrics_path.read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(lines):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if index != len(lines) - 1:
                    raise ValueError('Metrics log has an invalid non-final record') from None
                print('Removing an interrupted final metrics record', flush=True)
                temporary = metrics_path.with_suffix('.jsonl.tmp')
                temporary.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
                temporary.replace(metrics_path)
    existing = next((row for row in rows if row.get('iteration') == iteration), None)
    if not evaluation_due(iteration, SOURCE_ITERATION, config['eval_every']):
        if existing is None:
            with metrics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps({'iteration': iteration,
                                         'training_log_interrupted': True}) + '\n')
        return
    path = output / f'evaluation-{iteration}.json'
    if path.is_file():
        if existing is None:
            recovered = json.loads(path.read_text(encoding='utf-8'))
            with metrics_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(recovered) + '\n')
        return
    if existing is not None and 'evaluation' in existing:
        return
    print(f'Completing interrupted evaluation at iteration {iteration}', flush=True)
    mid, _ = load(output / 'mid.pt', device)
    anchor, _ = load(output / 'pro100.pt', device)
    result = {
        'mid': compare(candidate, mid, config['eval_games'], config['simulations'],
                       config['seed'] + iteration * 1000, config['max_moves']),
        'pro100': compare(candidate, anchor, config['eval_games'], config['simulations'],
                          config['seed'] + iteration * 1000 + 500, config['max_moves']),
    }
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'iteration': iteration, 'evaluation': result,
                                     'recovered_after_interruption': True}, indent=2), encoding='utf-8')
    temporary.replace(path)
    shutil.copy2(output / 'latest.pt', output / f'iteration-{iteration}.pt')
    if existing is None:
        with metrics_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({'iteration': iteration, 'evaluation': result,
                                     'recovered_after_interruption': True}) + '\n')


def validate_existing(output, source, seed, device):
    checkpoint = output / 'latest.pt'
    if not checkpoint.is_file():
        raise ValueError(f'Output has no latest.pt: {output}')
    _, data = load(checkpoint, device)
    config = data['training_config']
    if config.get('long_run_tag') != RUN_TAG or config.get('source_iteration') != SOURCE_ITERATION:
        raise ValueError('Output is not this continuation run')
    if config.get('source_checkpoint_sha256') != digest(source):
        raise ValueError('Source checkpoint has changed')
    if config.get('seed') != seed or config.get('eval_every') != 5 or config.get('eval_games') != 10:
        raise ValueError('Run settings differ from the saved continuation')
    if config.get('base_iteration') != SOURCE_ITERATION:
        raise ValueError('Evaluation origin is not iteration 102')
    for filename, key in (('pro100.pt', 'pro100_sha256'), ('mid.pt', 'mid_sha256')):
        if not (output / filename).is_file() or digest(output / filename) != config[key]:
            raise ValueError(f'Frozen reference changed or is missing: {filename}')
    return int(data['iteration'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Full iteration-102.pt pilot checkpoint')
    parser.add_argument('--output', type=Path, required=True,
                        help='New, empty run directory; reuse on restart')
    parser.add_argument('--target-iteration', type=int, default=DEFAULT_TARGET)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_file():
        parser.error(f'Source checkpoint is missing: {source}')
    if output == source.parent or output == source or source in output.parents:
        parser.error('Output must be outside the source run directory')
    if args.target_iteration < SOURCE_ITERATION or args.threads < 1:
        parser.error('Target must be >= 102 and threads must be positive')
    device = device_for(args.device)
    torch.set_num_threads(args.threads)
    if not output.exists() or not (output / 'latest.pt').exists():
        prepare(source, output, args.seed, device)
    current = validate_existing(output, source, args.seed, device)
    if current > args.target_iteration:
        parser.error(f'Saved iteration {current} already exceeds target {args.target_iteration}')
    remaining = args.target_iteration - current
    print(f'Current: {current}; target: {args.target_iteration}; remaining: {remaining}', flush=True)
    if args.prepare_only:
        return
    recover_pending_evaluation(output, current, device)
    if remaining == 0:
        return
    train(['--resume', str(output / 'latest.pt'), '--iterations', str(remaining),
           '--device', args.device, '--threads', str(args.threads)])


if __name__ == '__main__':
    main()
