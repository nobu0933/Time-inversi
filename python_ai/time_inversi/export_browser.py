"""Export only inference weights as a static browser asset (no training data)."""
import argparse
import base64
import json
from pathlib import Path

from .checkpoint import load


def export(checkpoint, output):
    model, data = load(checkpoint)
    tensors = {}
    for name, value in model.state_dict().items():
        array = value.detach().cpu().numpy().astype('<f4')
        tensors[name] = dict(shape=list(array.shape), data=base64.b64encode(array.tobytes()).decode('ascii'))
    payload = dict(format='time-inversi-browser-v1', iteration=data['iteration'],
                   config=model.config_dict(), tensors=tensors)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('// Generated inference weights. Regenerate with time_inversi.export_browser.\n'
                      'globalThis.TimeInversiWeights=' + json.dumps(payload, separators=(',', ':')) + ';\n',
                      encoding='utf-8')
    print(f'Exported iteration {data["iteration"]}: {output} ({output.stat().st_size:,} bytes)')


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=root / 'python_ai/runs/experiment01/latest.pt')
    parser.add_argument('--output', type=Path, default=root / 'assets/ai/model.js')
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == '__main__':
    main()
