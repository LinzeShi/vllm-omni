# SenseNova-U1 denoise synchronization benchmark

Reproduction harness for #8254 (A9 in #7677).
These are the original measurement scripts with portable paths, formatting,
output-directory checks and PyTorch accelerator APIs in place of CUDA-specific
synchronization/memory helpers. On CUDA those helpers use the same allocator.
The published numbers come from the original H200 runs, not a new run of this packaging.

## Get the reproduction files

These experiment files are archived in the author's fork, outside the PR's merge diff.
Clone the evidence branch and fetch the measured baseline before following the commands below:

```bash
git clone --single-branch --branch evidence/sensenova-a9 \
  https://github.com/LinzeShi/vllm-omni.git sensenova-a9-evidence
cd sensenova-a9-evidence
git fetch https://github.com/vllm-project/vllm-omni.git 62ebad2d7cd79da24653110d8043cc13fb744328
```

## Inspect the reported samples

From the repository root (no model, GPU or extra Python packages required):

```bash
python benchmarks/sensenova_u1/denoise_sync/summarize.py \
  --samples benchmarks/sensenova_u1/denoise_sync/samples.json
```

`samples.json` contains all ten measured requests per revision and the profiler
counts used by the PR table. Samples are transcribed from the retained JSONL files;
source SHA256 values identify the three production files actually measured.
They are evidence from one machine, not expected results for every GPU.

## Reproduce on a GPU

Use a vLLM-Omni development environment with vLLM 0.30.0, PyTorch 2.13.0,
CUDA 13.0, Transformers 5.14.1 and Diffusers 0.40.0. The original GPU was one
H200 SXM. Install the repository requirements before running this benchmark.
The full-model benchmark requires enough GPU memory for the checkpoint and generation workload.

Both variants use BF16, TP=1, 1024x1024, 50 steps, CFG=4, seed=42,
batch size 1, concurrency 1, think disabled and no TeaCache/Cache-DiT.
KV caching remains enabled. The prompt and all sampling settings are in
`run_benchmark.py`. Do not change these between variants.

Prepare two clean source trees and the exact model snapshot. Run from the evidence checkout:

```bash
BENCH="$(realpath benchmarks/sensenova_u1/denoise_sync)"
SENSENOVA_REPRO_ROOT="$(mktemp -d)"
MODEL="$SENSENOVA_REPRO_ROOT/model"
RESULTS="$SENSENOVA_REPRO_ROOT/results"

hf download sensenova/SenseNova-U1.5-8B-MoT \
  --revision 9feeeab8a2792514d109cd34589342a2cc1d4ab2 --local-dir "$MODEL"
git worktree add --detach "$SENSENOVA_REPRO_ROOT/base" 62ebad2d7cd79da24653110d8043cc13fb744328
git worktree add --detach "$SENSENOVA_REPRO_ROOT/candidate" 62ebad2d7cd79da24653110d8043cc13fb744328
git -C "$SENSENOVA_REPRO_ROOT/candidate" apply "$BENCH/measured.patch"
```

The baseline is fixed to the measured commit. `measured.patch` preserves the original measured production diff; it is not applied to the installed package automatically.
The recorded source hashes can be checked with `sha256sum` in the candidate tree.
The other optimization (A6) is not included in this comparison.

`sitecustomize.py` instruments each spawned worker; its directory must be on
`PYTHONPATH`. Keep its environment variables scoped to these commands. Run this
Bash function in the same shell as the variables above:

```bash
run_case() {
  local source_tree="$1" name="$2" suite="$3" trace=0 capture=0
  if [[ "$suite" == trace ]]; then trace=1; fi
  if [[ "$suite" == correctness ]]; then capture=1; fi
  PYTHONPATH="$BENCH/instrumentation:$source_tree" \
    A9_RESULT_DIR="$RESULTS/$name" A9_TRACE="$trace" A9_CAPTURE="$capture" \
    python "$BENCH/run_benchmark.py" --model "$MODEL" \
      --output "$RESULTS/$name" --suite "$suite"
}

run_case "$SENSENOVA_REPRO_ROOT/base" base-benchmark benchmark
run_case "$SENSENOVA_REPRO_ROOT/candidate" candidate-benchmark benchmark
run_case "$SENSENOVA_REPRO_ROOT/candidate" candidate-benchmark-round2 benchmark
run_case "$SENSENOVA_REPRO_ROOT/base" base-benchmark-round2 benchmark

run_case "$SENSENOVA_REPRO_ROOT/base" base-trace trace
run_case "$SENSENOVA_REPRO_ROOT/candidate" candidate-trace trace
python "$BENCH/summarize.py" --results "$RESULTS"

run_case "$SENSENOVA_REPRO_ROOT/base" base-correctness correctness
run_case "$SENSENOVA_REPRO_ROOT/candidate" candidate-correctness correctness
python "$BENCH/compare_outputs.py" "$RESULTS"
```

Each timing process performs one warmup and five measured requests. The second
round reverses the order. Profiling and correctness capture run in separate
processes; their elapsed times are excluded. Use new output directories for reruns.
`source.txt` records the imported package path so a wrong editable install is visible.

## Measurement boundaries

- E2E measures consumption of `Omni.generate`, after engine construction and before
  saving the output image. It excludes model loading, disk output and HTTP overhead.
- Denoising is timed around `_run_denoising_loop`, with device synchronization at
  both ends. Divide each measured loop by 50, then take the median of ten samples.
- Operator counts come from one profiled, warmed-up 50-step loop. They are counts,
  not transferred bytes or CUDA allocation counts. Profiling overhead and trace
  export/compression are excluded from the performance samples.
- Memory is PyTorch allocated memory in MiB, not device-wide `nvidia-smi` usage.

The original experiment did not reset the memory peak between requests. Its memory
column is the process high-water mark sampled after denoising, not incremental
request memory. This script preserves that definition.

The correctness cases cover CFG-off, restricted CFG intervals, image editing and
think mode. Repeated images are repeated requests, not distinct quality prompts.
The original 16 image pairs comprise 12 benchmark images (including warmups) and four correctness cases. No step-tensor or think-text comparison was recorded for this experiment.
