"""Smoke-test process-safe experiment summary appends without requiring PyTorch."""

from __future__ import annotations

import multiprocessing
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = type("Tensor", (), {})
    sys.modules["torch"] = torch_stub

from experiment_utils import _append_dataframe_csv_atomic


def _writer(summary_path: str, worker_id: int, rows_per_worker: int) -> None:
    for row_id in range(rows_per_worker):
        _append_dataframe_csv_atomic(
            pd.DataFrame([{"worker": worker_id, "row": row_id}]),
            summary_path,
        )


def main() -> None:
    worker_count = 6
    rows_per_worker = 12
    context = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory() as temp_dir:
        summary_path = str(Path(temp_dir) / "experiment_results.csv")
        processes = [
            context.Process(
                target=_writer,
                args=(summary_path, worker_id, rows_per_worker),
            )
            for worker_id in range(worker_count)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0, f"writer failed with exit code {process.exitcode}"

        summary = pd.read_csv(summary_path)
        assert len(summary) == worker_count * rows_per_worker
        assert not summary.duplicated(["worker", "row"]).any()

        corrupt_path = Path(temp_dir) / "corrupt.csv"
        corrupt_path.write_text('worker,row\n"unterminated', encoding="utf-8")
        original = corrupt_path.read_bytes()
        try:
            _append_dataframe_csv_atomic(
                pd.DataFrame([{"worker": 0, "row": 0}]),
                str(corrupt_path),
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("corrupt summaries must not be silently overwritten")
        assert corrupt_path.read_bytes() == original

    print("smoke_concurrent_results: ok")


if __name__ == "__main__":
    main()
