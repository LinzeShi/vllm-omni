# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Measure the #8304 fused wrapper across prefix lengths, without model weights."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import torch
import triton
import vllm

from tests.diffusion.models.sensenova_u1.test_sensenova_triton_kernel import (
    DEVICE,
    EPS,
    H_K,
    HEAD_DIM,
    SEED,
    make_input,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[7, 64, 257])
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    cache_dir = Path(os.environ["TRITON_CACHE_DIR"])
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise RuntimeError(f"A fresh Triton cache is required: {cache_dir}")

    spec = importlib.util.spec_from_file_location("stride_benchmark_kernel", args.kernel_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compiled = []
    launch = module.qk_norm_rope_kernel.run

    def record_kernel(*call_args, **kwargs):
        kernel = launch(*call_args, **kwargs)
        compiled.append(kernel)
        return kernel

    module.qk_norm_rope_kernel.run = record_kernel
    data = make_input(args.seq_len, torch.bfloat16, DEVICE, SEED, args.batch_size)
    value = torch.randn(args.batch_size, args.seq_len, H_K * 3, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    value = value[:, :, H_K : 2 * H_K]
    # Initialize the CUDA context and exercise allocation/synchronization before timing.
    torch.cuda.synchronize()
    samples = []
    for prefix in args.prefix_lengths:
        key_store = torch.empty(args.batch_size, prefix + args.seq_len, H_K, HEAD_DIM,
                                device=DEVICE, dtype=torch.bfloat16)
        value_store = torch.empty_like(key_store)
        key_out = key_store[:, prefix:].transpose(1, 2)
        value_out = value_store[:, prefix:].transpose(1, 2)

        def invoke():
            return module.triton_qk_norm_rope(*data.args(), EPS, key_output=key_out,
                                             value=value, value_output=value_out)

        torch.cuda.synchronize()
        start = time.perf_counter()
        invoke()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - start) * 1000
        first_kernel = compiled[-1]
        variants_seen = len({id(kernel) for kernel in compiled})
        for _ in range(args.warmup):
            invoke()
        torch.cuda.synchronize()
        warm_ms = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            invoke()
            torch.cuda.synchronize()
            warm_ms.append((time.perf_counter() - start) * 1000)
        assert all(kernel is first_kernel for kernel in compiled[-args.repeats:])
        samples.append({
            "prefix_tokens": prefix,
            "batch_stride": key_out.stride(0),
            "first_call_ms": first_ms,
            "warm_median_ms": statistics.median(warm_ms),
            "warm_samples_ms": warm_ms,
            "variants_seen": variants_seen,
            "kernel_hash": first_kernel.hash,
        })

    result = {
        "scope": "Fused QK norm/RoPE + V store wrapper only; no tokenizer or full model",
        "timing": "CPU wall time including wrapper launch and CUDA synchronization",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "triton": triton.__version__, "vllm": vllm.__version__,
        "kernel_source": str(args.kernel_source.resolve()),
        "source_sha256": hashlib.sha256(args.kernel_source.read_bytes()).hexdigest(),
        "triton_cache_dir": str(cache_dir),
        "seq_len": args.seq_len, "batch_size": args.batch_size,
        "dtype": "bfloat16", "warmup": args.warmup, "repeats": args.repeats,
        "samples": samples,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "samples": [
        {key: val for key, val in sample.items() if key not in ("warm_samples_ms", "kernel_hash")}
        for sample in samples
    ]}, indent=2))


if __name__ == "__main__":
    main()
