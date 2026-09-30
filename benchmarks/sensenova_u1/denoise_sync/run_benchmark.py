# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Run the fixed SenseNova-U1 H200 comparison workload; see README.md."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

from PIL import Image, ImageDraw

from vllm_omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--suite", choices=["benchmark", "correctness", "trace", "cache"], default="benchmark")
    ap.add_argument("--cache-backend", default=None)
    args = ap.parse_args()
    out = Path(args.output).resolve()
    if os.environ.get("A9_RESULT_DIR") is None or Path(os.environ["A9_RESULT_DIR"]).resolve() != out:
        ap.error("Set A9_RESULT_DIR to the --output directory; see README.md")
    if (os.environ.get("A9_TRACE") == "1") != (args.suite == "trace"):
        ap.error("Enable A9_TRACE=1 only for --suite trace")
    if args.suite == "benchmark" and os.environ.get("A9_CAPTURE") == "1":
        ap.error("Do not capture tensors during timing")
    if out.exists() and any(out.iterdir()):
        ap.error("Use an empty output directory for each run")
    out.mkdir(parents=True, exist_ok=True)
    (out / "environment.json").write_text(
        json.dumps(
            {k: importlib.metadata.version(k) for k in ["torch", "vllm", "vllm-omni", "transformers", "diffusers"]},
            indent=2,
        )
    )
    import vllm_omni

    (out / "source.txt").write_text(vllm_omni.__file__)
    prompt = "A red ceramic teapot on a wooden table beside a window, soft morning light, detailed product photograph."
    image = Image.new("RGB", (512, 512), (220, 225, 230))
    draw = ImageDraw.Draw(image)
    draw.rectangle((120, 140, 390, 410), fill=(180, 40, 30))
    draw.ellipse((155, 65, 350, 180), fill=(35, 120, 70))
    image.save(out / "reference.png")
    cases = [("t2i-1024", 1024, 1024, 50, 4.0, False, False, 1.0, (0.0, 1.0), 1, 5)]
    if args.suite == "trace":
        cases = [("t2i-1024", 1024, 1024, 50, 4.0, False, False, 1.0, (0.0, 1.0), 1, 1)]
    if args.suite == "correctness":
        cases = [
            ("cfg-off", 512, 512, 8, 1.0, False, False, 1.0, (0.0, 1.0), 0, 1),
            ("cfg-interval", 768, 512, 8, 4.0, False, False, 1.0, (0.25, 0.75), 0, 1),
            ("img2img", 512, 512, 8, 4.0, False, True, 2.0, (0.25, 0.75), 0, 1),
            ("think", 512, 512, 8, 4.0, True, False, 1.0, (0.0, 1.0), 0, 1),
        ]
    if args.suite == "cache":
        cases = [("cache", 512, 512, 8, 4.0, False, False, 1.0, (0.0, 1.0), 0, 1)]
    omni = Omni(model=args.model, tensor_parallel_size=1, cache_backend=args.cache_backend, log_stats=True)
    try:
        for name, w, h, steps, cfg, think, edit, imgcfg, interval, warmup, repeats in cases:
            for i in range(warmup + repeats):
                sampling = OmniDiffusionSamplingParams(
                    width=w,
                    height=h,
                    seed=42,
                    num_inference_steps=steps,
                    extra_args={
                        "cfg_scale": cfg,
                        "img_cfg_scale": imgcfg,
                        "cfg_norm": "none",
                        "timestep_shift": 3.0,
                        "cfg_interval": interval,
                        "batch_size": 1,
                        "think": think,
                        "t_eps": 0.02,
                    },
                )
                prompts = {
                    "prompt": prompt if not edit else "Turn this image into a watercolor painting.",
                    "modalities": ["image"],
                }
                if edit:
                    prompts["multi_modal_data"] = {"image": image}
                start = time.perf_counter()
                outputs = list(omni.generate(prompts=prompts, sampling_params_list=sampling))
                elapsed = (time.perf_counter() - start) * 1000
                images = [im for result in outputs for im in (getattr(result, "images", None) or [])]
                assert len(images) == 1, f"Expected one image, got {len(images)}"
                im = images[0]
                assert im.size == (w, h)
                file = f"{name}-{i}.png"
                im.save(out / file)
                row = {
                    "case": name,
                    "iteration": i,
                    "warmup": i < warmup,
                    "e2e_ms": elapsed,
                    "image": file,
                    "sha256_pixels": hashlib.sha256(im.tobytes()).hexdigest(),
                    "size": [w, h],
                    "steps": steps,
                }
                with (out / "requests.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print("A9_RESULT", json.dumps(row), flush=True)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
