# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Worker-process instrumentation for the fixed denoise-sync experiment."""

import os

if os.environ.get("A9_RESULT_DIR"):
    import functools
    import gzip
    import json
    import time
    from pathlib import Path

    import torch

    from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline

    original = SenseNovaU1Pipeline._run_denoising_loop
    counts = {}

    @functools.wraps(original)
    def measured(self, ns, caches, p, *args, **kwargs):
        root = Path(os.environ["A9_RESULT_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        key = (tuple(p.image_size), p.num_steps)
        counts[key] = counts.get(key, 0) + 1
        diagnostic = os.environ.get("A9_TRACE") == "1" and p.num_steps == 50 and counts[key] == 2
        torch.accelerator.synchronize()
        started = time.perf_counter()
        if diagnostic:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                output = original(self, ns, caches, p, *args, **kwargs)
            prof.export_chrome_trace(str(root / "denoise-trace.json"))
            stats = {event.key: {"count": event.count, "cpu_us": event.cpu_time_total} for event in prof.key_averages()}
            (root / "operators.json").write_text(json.dumps(stats, indent=2))
            raw = root / "denoise-trace.json"
            with raw.open("rb") as src, gzip.open(str(raw) + ".gz", "wb") as dst:
                import shutil

                shutil.copyfileobj(src, dst)
            raw.unlink()
        else:
            output = original(self, ns, caches, p, *args, **kwargs)
        torch.accelerator.synchronize()
        record = {
            "pid": os.getpid(),
            "size": list(p.image_size),
            "steps": p.num_steps,
            "index": counts[key],
            "diagnostic": diagnostic,
            "denoise_ms": (time.perf_counter() - started) * 1000,
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
        }
        with (root / f"loops-{os.getpid()}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return output

    SenseNovaU1Pipeline._run_denoising_loop = measured
