# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-process instrumentation for the fixed KV-layout experiment."""

import os

if os.environ.get("A6_RESULT_DIR"):
    import functools
    import gzip
    import json
    import shutil
    import time
    from pathlib import Path

    import torch

    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    root = Path(os.environ["A6_RESULT_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1Attention

    original_loop = SenseNovaU1Pipeline._run_denoising_loop
    original_step = SenseNovaU1Pipeline._denoise
    counts = {}
    active = None

    if os.environ.get("A6_TRACE") == "1":
        original_attention = SenseNovaU1Attention.forward_gen

        @functools.wraps(original_attention)
        def attention(self, *args, **kwargs):
            with torch.profiler.record_function("A6.forward_gen"):
                return original_attention(self, *args, **kwargs)

        SenseNovaU1Attention.forward_gen = attention

    @functools.wraps(original_step)
    def step(self, image_prediction, ns, t, z, image_embeds, caches, p, step_i, is_it2i):
        result = original_step(self, image_prediction, ns, t, z, image_embeds, caches, p, step_i, is_it2i)
        if active is not None and os.environ.get("A6_CAPTURE") == "1":
            torch.save(result.detach().cpu(), root / f"{active}-step-{step_i:03d}.pt")
        return result

    @functools.wraps(original_loop)
    def loop(self, ns, caches, p, *args, **kwargs):
        global active
        key = (tuple(p.image_size), p.num_steps, p.think_mode, bool(kwargs.get("is_it2i", False)))
        counts[key] = counts.get(key, 0) + 1
        index = counts[key]
        active = (
            f"{p.image_size[0]}x{p.image_size[1]}-s{p.num_steps}-think{int(p.think_mode)}-edit{int(key[3])}-n{index}"
        )
        diagnostic = os.environ.get("A6_TRACE") == "1" and p.num_steps == 50 and index == 2
        torch.accelerator.synchronize()
        allocated = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            if diagnostic:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    output = original_loop(self, ns, caches, p, *args, **kwargs)
                torch.accelerator.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
                stats = {e.key: {"count": e.count, "cpu_us": e.cpu_time_total} for e in prof.key_averages()}
                (root / "operators.json").write_text(json.dumps(stats, indent=2))
                scoped = {}
                for event in prof.events():
                    parent = event.cpu_parent
                    while parent is not None and parent.name != "A6.forward_gen":
                        parent = parent.cpu_parent
                    if parent is not None and event.name.startswith("aten::"):
                        scoped[event.name] = scoped.get(event.name, 0) + 1
                (root / "attention-operators.json").write_text(json.dumps(scoped, indent=2))
                trace = root / "denoise-trace.json"
                prof.export_chrome_trace(str(trace))
                with trace.open("rb") as src, gzip.open(str(trace) + ".gz", "wb") as dst:
                    shutil.copyfileobj(src, dst)
                trace.unlink()
            else:
                output = original_loop(self, ns, caches, p, *args, **kwargs)
                torch.accelerator.synchronize()
                elapsed = (time.perf_counter() - started) * 1000
            row = {
                "pid": os.getpid(),
                "key": active,
                "size": list(p.image_size),
                "steps": p.num_steps,
                "index": index,
                "diagnostic": diagnostic,
                "capture": os.environ.get("A6_CAPTURE") == "1",
                "denoise_ms": elapsed,
                "allocated_before_bytes": allocated,
                "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
                "temporary_peak_bytes": torch.accelerator.max_memory_allocated() - allocated,
                "think_text": args[0] if args else kwargs.get("think_text", ""),
            }
            with (root / f"loops-{os.getpid()}.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            return output
        finally:
            active = None

    SenseNovaU1Pipeline._denoise = step
    SenseNovaU1Pipeline._run_denoising_loop = loop
