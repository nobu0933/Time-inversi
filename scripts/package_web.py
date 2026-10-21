"""Create a static hosting ZIP; excludes Python, checkpoints and training data."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path(__file__).resolve().parents[1]
output = root / 'dist/time-inversi-web.zip'
output.parent.mkdir(exist_ok=True)
files = [root / 'index.html']
for directory in ('src', 'styles', 'assets'):
    files.extend(sorted((root / directory).rglob('*')))
for model in ('model.js', 'pro-model.js'):
    if not (root / 'assets/ai' / model).is_file():
        raise SystemExit(f'Export {model} before packaging')
with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
    for file in files:
        if file.is_file():
            archive.write(file, file.relative_to(root).as_posix())
print(f'{output} ({output.stat().st_size:,} bytes)')
