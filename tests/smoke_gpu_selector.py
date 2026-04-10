import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    from tools.select_idle_gpus import parse_nvidia_smi_rows

    sample = """

    , ,
    0, 1112 MiB, 91 %
    1, 0 MiB, 0 %
    N/A, N/A, N/A
    2, 2670 MiB, 46 %
    3, 0 MiB, 0 %
    """
    selected = parse_nvidia_smi_rows(sample, max_gpus=2, mem_max_mb=256, util_max=5)
    _assert(selected == ["1", "3"], f"Expected GPUs 1,3, got {selected!r}.")
    _assert(",".join(selected) == "1,3", "GPU list must not contain leading/trailing commas.")
    print("OK: GPU selector smoke checks passed.")


if __name__ == "__main__":
    main()
