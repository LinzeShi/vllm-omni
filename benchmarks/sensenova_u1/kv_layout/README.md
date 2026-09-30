# SenseNova-U1 KV layout benchmark

Reproduction harness for #8304 (A6 in #7677).
These are the original measurement scripts with portable paths, formatting,
output-directory checks and PyTorch accelerator APIs in place of CUDA-specific
synchronization/memory helpers. On CUDA those helpers use the same allocator.
The published numbers come from the original H200 runs, not a new run of this packaging.

## Inspect the reported samples

From the repository root (no model, GPU or extra Python packages required):

```bash
python benchmarks/sensenova_u1/kv_layout/summarize.py \
  --samples benchmarks/sensenova_u1/kv_layout/samples.json
```

`samples.json` contains all ten measured requests per revision and the profiler
counts used by the PR table. Samples are transcribed from the retained JSONL files;
source SHA256 values identify the three production files actually measured.
They are evidence from one machine, not expected results for every GPU.

## Reproduce on a GPU

Use a vLLM-Omni development environment with vLLM 0.30.0, PyTorch 2.13.0,
CUDA 13.0, Transformers 5.14.1 and Diffusers 0.40.0. The original GPU was one
H200 SXM. Install the repository requirements before running this benchmark.
The complete model requires substantially more memory than the attention microbenchmark.

Both variants use BF16, TP=1, 1024x1024, 50 steps, CFG=4, seed=42,
batch size 1, concurrency 1, think disabled and no TeaCache/Cache-DiT.
KV caching remains enabled. The prompt and all sampling settings are in
`run_benchmark.py`. Do not change these between variants.

Prepare two clean source trees and the exact model snapshot. Run from the PR checkout:

```bash
BENCH="$(realpath benchmarks/sensenova_u1/kv_layout)"
SENSENOVA_REPRO_ROOT="$(mktemp -d)"
MODEL="$SENSENOVA_REPRO_ROOT/model"
RESULTS="$SENSENOVA_REPRO_ROOT/results"

hf download sensenova/SenseNova-U1.5-8B-MoT \
  --revision 9feeeab8a2792514d109cd34589342a2cc1d4ab2 --local-dir "$MODEL"
git worktree add --detach "$SENSENOVA_REPRO_ROOT/base" 0173cb374839c437df098e5a34b2f842769fb266
git worktree add --detach "$SENSENOVA_REPRO_ROOT/candidate" 0173cb374839c437df098e5a34b2f842769fb266
git diff beaade82ca813ee481b1b49469afa7a86fc72cd2^ beaade82ca813ee481b1b49469afa7a86fc72cd2 -- vllm_omni/diffusion/models/sensenova_u1 > "$SENSENOVA_REPRO_ROOT/change.patch"
git -C "$SENSENOVA_REPRO_ROOT/candidate" apply "$SENSENOVA_REPRO_ROOT/change.patch"
```

The baseline is fixed to the measured commit. The production diff of the original PR commit reconstructs the measured candidate.
The recorded source hashes can be checked with `sha256sum` in the candidate tree.
The other optimization (A9) is not included in this comparison.

`sitecustomize.py` instruments each spawned worker; its directory must be on
`PYTHONPATH`. Keep its environment variables scoped to these commands. Run this
Bash function in the same shell as the variables above:

```bash
run_case() {
  local source_tree="$1" name="$2" suite="$3" trace=0 capture=0
  if [[ "$suite" == trace ]]; then trace=1; fi
  if [[ "$suite" == correctness ]]; then capture=1; fi
  PYTHONPATH="$BENCH/instrumentation:$source_tree" \
    A6_RESULT_DIR="$RESULTS/$name" A6_TRACE="$trace" A6_CAPTURE="$capture" \
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

The peak is reset immediately before each denoise loop. The temporary peak subtracts
the allocated bytes at loop entry; it is unchanged in the full-model comparison.

The correctness cases cover CFG-off, restricted CFG intervals, image editing and
think mode. Repeated images are repeated requests, not distinct quality prompts.
The two random, unseeded startup dummy steps are not paired inputs. The comparison reports them separately and excludes only those from exact-match assertions.

## Attention allocation microbenchmark

This uses random weights and inputs, not the full checkpoint. The dimensions and
seeds match the original attention-only measurement. Run each variant in its own
process; five warmups, 30 timed calls and three profiled calls are fixed in the script.

```bash
PYTHONPATH="$SENSENOVA_REPRO_ROOT/base" python "$BENCH/benchmark_attention.py" \
  --variant baseline --out "$RESULTS/micro-baseline"
PYTHONPATH="$SENSENOVA_REPRO_ROOT/candidate" python "$BENCH/benchmark_attention.py" \
  --variant candidate --out "$RESULTS/micro-candidate"
```

The `temporary_peak_bytes` field is the peak above entry allocation for one call.
Divide `operators_for_3_calls` by three for per-call counts. The 1024-token cases
measured 50 to 24 MiB and five to zero `aten::copy_` calls per attention invocation.
This does not imply that the complete model has zero copies or halves its memory.
