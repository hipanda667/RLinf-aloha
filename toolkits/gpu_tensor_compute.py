#!/usr/bin/env python3
"""Allocate about 40 GiB of GPU tensor storage and run pure tensor compute.

This script intentionally does not load a model or read a dataset.  It keeps a
large tensor resident on one CUDA device and repeatedly performs tensor
operations plus BF16/FP16 matrix multiplication on the same device.

Example:
    python toolkits/gpu_tensor_compute.py --gpu 0 --target-gib 40 --duration 300

Use ``--duration 0`` to run until Ctrl-C.  The script refuses to start when
CUDA is unavailable or when the requested allocation would leave less than the
requested workspace margin free.
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from typing import Final

import torch

_BYTES_PER_GIB: Final[int] = 1024**3


def _dtype_from_name(name: str) -> torch.dtype:
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    try:
        return dtypes[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {sorted(dtypes)}") from exc


def _gib(value: int) -> float:
    return value / _BYTES_PER_GIB


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0).")
    parser.add_argument(
        "--target-gib",
        type=float,
        default=40.0,
        help="Resident tensor payload in GiB; default: 40.0.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Dtype of the large resident tensor and matmul operands.",
    )
    parser.add_argument(
        "--matmul-size",
        type=int,
        default=8192,
        help="Square matmul dimension; default: 8192.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Run duration in seconds; 0 means run until Ctrl-C.",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=5.0,
        help="Seconds between progress reports; default: 5.",
    )
    parser.add_argument(
        "--workspace-gib",
        type=float,
        default=1.0,
        help="Free-memory safety margin for matmul/workspace allocations.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.target_gib <= 0:
        raise SystemExit("--target-gib must be positive")
    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")
    if args.check_interval <= 0:
        raise SystemExit("--check-interval must be positive")
    if args.workspace_gib < 0:
        raise SystemExit("--workspace-gib must be non-negative")
    if args.matmul_size <= 0 or args.matmul_size % 8 != 0:
        raise SystemExit("--matmul-size must be positive and divisible by 8")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise SystemExit(
            f"Invalid --gpu {args.gpu}; visible CUDA devices: {torch.cuda.device_count()}"
        )

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dtype = _dtype_from_name(args.dtype)
    props = torch.cuda.get_device_properties(device)
    free_before, total_memory = torch.cuda.mem_get_info(device)

    target_bytes = int(args.target_gib * _BYTES_PER_GIB)
    workspace_bytes = int(args.workspace_gib * _BYTES_PER_GIB)
    if target_bytes + workspace_bytes > free_before:
        raise SystemExit(
            "Requested allocation is too large: "
            f"target={args.target_gib:.2f} GiB + workspace={args.workspace_gib:.2f} GiB, "
            f"free={_gib(free_before):.2f} GiB on {props.name}."
        )

    # Round down to a whole number of elements.  fill_ forces physical page
    # commitment so nvidia-smi reports the intended resident allocation.
    element_size = torch.empty((), dtype=dtype).element_size()
    num_elements = target_bytes // element_size
    resident_bytes = num_elements * element_size
    print(
        f"device={device} name={props.name} total={_gib(total_memory):.2f} GiB "
        f"free_before={_gib(free_before):.2f} GiB",
        flush=True,
    )
    print(
        f"dtype={dtype} resident_tensor={_gib(resident_bytes):.2f} GiB "
        f"elements={num_elements:,} matmul={args.matmul_size}x{args.matmul_size}",
        flush=True,
    )

    # Keep this tensor alive for the entire run.  Its values are updated below
    # so the workload is not just an allocation benchmark.
    resident = torch.empty(num_elements, dtype=dtype, device=device)
    resident.fill_(1.0)

    n = args.matmul_size
    lhs = torch.randn((n, n), dtype=dtype, device=device)
    rhs = torch.randn((n, n), dtype=dtype, device=device)
    result = torch.empty((n, n), dtype=dtype, device=device)
    torch.cuda.synchronize(device)

    stop_requested = False

    def _request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    started = time.monotonic()
    last_report = started
    iterations = 0
    while not stop_requested and (args.duration == 0 or time.monotonic() - started < args.duration):
        # Large in-place bandwidth workload over the resident tensor.
        resident.mul_(0.9999).add_(0.0001)
        # Tensor-core-friendly pure tensor compute workload.
        torch.mm(lhs, rhs, out=result)
        result.add_(lhs)
        iterations += 1

        now = time.monotonic()
        if now - last_report >= args.check_interval:
            torch.cuda.synchronize(device)
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            elapsed = now - started
            print(
                f"elapsed={elapsed:.1f}s iterations={iterations} "
                f"allocated={_gib(allocated):.2f}GiB "
                f"reserved={_gib(reserved):.2f}GiB",
                flush=True,
            )
            last_report = now

    torch.cuda.synchronize(device)
    elapsed = time.monotonic() - started
    print(f"stopping: elapsed={elapsed:.1f}s iterations={iterations}", flush=True)


if __name__ == "__main__":
    main()
