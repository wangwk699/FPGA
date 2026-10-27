#!/usr/bin/env python3
"""Check the existing llm environment; do not create/install anything."""
from importlib import metadata
from pathlib import Path
import shlex
import sys


def main():
    if sys.version_info[:2] != (3, 11):
        raise SystemExit('Activate your existing Python 3.11 environment: conda activate llm')
    requirements = Path(__file__).resolve().parents[1] / 'requirements-reproduce.txt'
    missing = []
    for line in requirements.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        package, expected = line.split('==')
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            actual = None
        print(f'{package}: {actual or "missing"}; verified version: {expected}')
        if actual is None or actual.split('+')[0] != expected:
            missing.append(line)
    if missing:
        print('\nOnly the missing/different packages need attention:')
        print(shlex.join([sys.executable, '-m', 'pip', 'install', *missing]))
        raise SystemExit(1)
    print('Existing llm environment is ready; no Conda creation or package installation needed.')


if __name__ == '__main__':
    main()
