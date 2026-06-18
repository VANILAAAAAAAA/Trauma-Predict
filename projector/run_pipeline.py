#!/usr/bin/env python3
"""Run the vital numeric projector feasibility pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path) -> None:
    print('+', ' '.join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default='/home/vanila/code/EHR-Predict')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    wd = Path(args.workdir)
    py = sys.executable
    run([py, 'projector/build_vital_dataset.py', '--workdir', str(wd)], wd)
    run([py, 'projector/train_smoke.py', '--workdir', str(wd), '--epochs', str(args.epochs), '--device', args.device], wd)


if __name__ == '__main__':
    main()
