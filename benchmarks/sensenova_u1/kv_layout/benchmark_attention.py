# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch
from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.utils.network_utils import get_file_store_init_method

from vllm_omni.diffusion.models.sensenova_u1 import sensenova_u1_transformer as module
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1Config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["baseline", "candidate"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    results = []
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cuda"))), torch.inference_mode():
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method=get_file_store_init_method()
        )
        initialize_model_parallel()
        try:
            for seq, prefix in [(256, 128), (1024, 128), (1024, 2048)]:
                torch.manual_seed(42)
                config = SenseNovaU1Config(
                    llm_config=dict(
                        hidden_size=4096,
                        num_attention_heads=32,
                        num_key_value_heads=8,
                        attention_bias=False,
                        rms_norm_eps=1e-6,
                    )
                ).llm_config
                model = module.SenseNovaU1Attention(config, 0).to(device="cuda", dtype=torch.bfloat16)
                for t in model.parameters():
                    t.normal_(0, 0.02)
                cache = module.DynamicCache() if a.variant == "baseline" else module.FlashKVCache()
                k, v = torch.randn(2, 1, 8, prefix, 128, device="cuda", dtype=torch.bfloat16)
                cache.update(k, v, 0)
                module.prepare_flash_kv_cache(cache, seq, 1)
                x = torch.randn(1, seq, 4096, device="cuda", dtype=torch.bfloat16)
                rope = []
                for d in (64, 32, 32):
                    f = torch.randn(1, seq, d // 2, device="cuda", dtype=torch.float32).repeat(1, 1, 2)
                    rope.append((f.cos().to(x.dtype), f.sin().to(x.dtype)))

                run = partial(
                    model.forward_gen, x, None, None, cache, position_embeddings=tuple(rope), update_cache=False
                )

                for _ in range(5):
                    y = run()
                torch.accelerator.synchronize()
                torch.save(y.cpu(), out / f"output-{seq}-{prefix}.pt")
                torch.accelerator.reset_peak_memory_stats()
                allocated = torch.accelerator.memory_allocated()
                y = run()
                torch.accelerator.synchronize()
                peak = torch.accelerator.max_memory_allocated() - allocated
                wall = []
                gpu = []
                for _ in range(30):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.accelerator.synchronize()
                    t = time.perf_counter()
                    start.record()
                    y = run()
                    end.record()
                    end.synchronize()
                    wall.append((time.perf_counter() - t) * 1000)
                    gpu.append(start.elapsed_time(end))
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True,
                ) as prof:
                    for _ in range(3):
                        y = run()
                    torch.accelerator.synchronize()
                counts = {e.key: e.count for e in prof.key_averages() if e.key.startswith("aten::")}
                prof.export_chrome_trace(str(out / f"trace-{seq}-{prefix}.json"))
                results.append(
                    dict(
                        seq=seq,
                        prefix=prefix,
                        wall_ms=wall,
                        gpu_ms=gpu,
                        wall_median_ms=statistics.median(wall),
                        gpu_median_ms=statistics.median(gpu),
                        temporary_peak_bytes=peak,
                        operators_for_3_calls=counts,
                    )
                )
                del run, model, cache, x, rope, y
        finally:
            cleanup_dist_env_and_memory()
    (out / "results.json").write_text(
        json.dumps(
            {"variant": a.variant, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cases": results},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
