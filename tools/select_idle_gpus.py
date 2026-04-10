#!/usr/bin/env python
"""Select idle GPUs from nvidia-smi CSV output."""

from __future__ import annotations

import argparse
import re
import subprocess
from typing import Iterable


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _parse_number(value: str) -> float | None:
    match = _NUMBER_RE.search(value.strip())
    if match is None:
        return None
    return float(match.group(0))


def parse_nvidia_smi_rows(
    text: str,
    *,
    max_gpus: int,
    mem_max_mb: float,
    util_max: float,
) -> list[str]:
    if max_gpus <= 0:
        return []

    selected: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue

        idx_value = _parse_number(parts[0])
        mem_value = _parse_number(parts[1])
        util_value = _parse_number(parts[2])
        if idx_value is None or mem_value is None or util_value is None:
            continue
        if not idx_value.is_integer():
            continue

        idx = str(int(idx_value))
        if mem_value <= mem_max_mb and util_value <= util_max:
            selected.append(idx)
        if len(selected) >= max_gpus:
            break
    return selected


def query_nvidia_smi(command: Iterable[str] = ("nvidia-smi",)) -> str:
    cmd = [
        *command,
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return ""
    return completed.stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Print comma-separated idle GPU indexes.")
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--mem-max-mb", type=float, default=256)
    parser.add_argument("--util-max", type=float, default=5)
    args = parser.parse_args()

    selected = parse_nvidia_smi_rows(
        query_nvidia_smi(),
        max_gpus=max(args.max_gpus, 0),
        mem_max_mb=args.mem_max_mb,
        util_max=args.util_max,
    )
    print(",".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
