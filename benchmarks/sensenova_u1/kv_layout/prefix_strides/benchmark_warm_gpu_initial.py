# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Paired GPU-only timing of the old and fixed resident-suffix kernels."""

import importlib.util
import json
from pathlib import Path
import statistics

import torch

from tests.diffusion.models.sensenova_u1.test_sensenova_triton_kernel import (
    DEVICE, EPS, H_K, HEAD_DIM, SEED, make_input,
)


def load_kernel(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    evidence = Path(__file__).resolve().parent
    modules = {name: load_kernel(name, evidence / f"kernel_{name}.py") for name in ("before", "after")}
    rows = []
    for batch_size in (1, 2):
        for seq_len in (9, 260, 1024):
            data = make_input(seq_len, torch.bfloat16, DEVICE, SEED, batch_size)
            value = torch.randn(batch_size, seq_len, H_K * 3, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            value = value[:, :, H_K : 2 * H_K]
            prefix = 257
            k = torch.empty(batch_size, prefix + seq_len, H_K, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
            v = torch.empty_like(k)
            key_out, value_out = k[:, prefix:].transpose(1, 2), v[:, prefix:].transpose(1, 2)
            graphs = {}
            outputs = {}
            for name, module in modules.items():
                def invoke():
                    return module.triton_qk_norm_rope(*data.args(), EPS, key_output=key_out,
                                                     value=value, value_output=value_out)

                for _ in range(5):
                    invoke()
                outputs[name] = tuple(t.clone() for t in invoke()) + (value_out.clone(),)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                # Amortize Python launch cost; all calls use the same fixed inputs.
                with torch.cuda.graph(graph):
                    for _ in range(32):
                        invoke()
                graphs[name] = graph
            for old, new in zip(outputs["before"], outputs["after"]):
                torch.testing.assert_close(old, new, rtol=0, atol=0)
            times = {name: [] for name in modules}
            for round_index in range(10):
                order = ("before", "after") if round_index % 2 == 0 else ("after", "before")
                for name in order:
                    graph = graphs[name]
                    graph.replay()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(5):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times[name].append(start.elapsed_time(end) / (32 * 5))
            rows.append({"batch_size": batch_size, "seq_len": seq_len, "prefix_tokens": prefix,
                         "exact_qkv_match": True, "gpu_ms": times,
                         "median_gpu_ms": {name: statistics.median(values) for name, values in times.items()}})
            print(json.dumps({key: val for key, val in rows[-1].items() if key != "gpu_ms"}), flush=True)
    result = {"scope": "GPU execution only, 32 fixed-input calls per CUDA Graph; 5 replays per timing",
              "rounds": 10, "order": "alternating before/after", "dtype": "bfloat16",
              "gpu": torch.cuda.get_device_name(), "samples": rows}
    (evidence / "warm-gpu.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
