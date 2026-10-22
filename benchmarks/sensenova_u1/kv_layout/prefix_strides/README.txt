SenseNova prefix-stride compilation check for PR #8304

Before: 93b8e75a755616b0f1be64eb1592ad16eb814c6d
After:  ba8270772d4f9250dced7a3f3184cb7aa0faa0dc
The two kernel snapshots are byte-for-byte copies of fused_rmsnorm_rope.py at those commits.

Scope: fused QK RMSNorm/RoPE and V store only. No model weights, tokenizer,
or complete generation requests. These measurements do not establish an
end-to-end speedup. The first call also includes Triton initialization.

Environment: RTX 4070 Laptop (8 GiB), Python 3.12, vLLM 0.30.0,
PyTorch 2.13.0+cu130, Triton 3.7.1, OMP_NUM_THREADS=4.

Cold measurements (benchmark_strides.py):
BF16, batch 1, image sequence length 1024, 32 Q heads, 8 KV heads, head dimension 128.
Prefixes 7, 64, 257 run consecutively. Each process starts with an empty Triton cache.
Order: before-r1, after-r1, after-r2, before-r2. Each prefix has a first call,
5 warmup calls, and 25 measured warm calls. Wall time includes CUDA synchronization.
Both runs observe three compiled variants before and one after.
The first prefix is noisy (1.1-6.1 seconds before, 3.8-4.0 seconds after).
Subsequent first calls are 139.85-169.59 ms before and 0.31-5.27 ms after.
The warm host samples vary substantially; do not use them to claim throughput gains.

Warm GPU measurements (benchmark_warm_gpu.py, warm-gpu.json):
Same process, BF16, prefix 257, batches 1/2, image sequence lengths 9/260/1024.
A CUDA Graph contains 32 calls; each timing has five graph replays, measured with
CUDA events. Both variants receive 100 graph replays of warmup. Ten measurement
rounds alternate the order. Median differences range from -0.53% to +0.98%.
Q/K/V match exactly for all six configurations.

The initial warm experiment had insufficient joint warmup. Its batch 1 / length 260
aggregate showed about +10%, with both variants changing from about 59us to 44us
mid-run. The cause was not instrumented. The initial script and samples are retained
as benchmark_warm_gpu_initial.py and warm-gpu-initial.json. The later experiment adds
joint warmup; it does not change production code. Do not discard the initial evidence.

Reproduction

Use an Omni checkout at the after commit above and a matching vLLM 0.30 environment.
No checkpoint download is needed. Run the following from that checkout's root.

    git fetch https://github.com/LinzeShi/vllm-omni.git evidence/sensenova-a6
    stride_archive=$(mktemp -d)
    git archive FETCH_HEAD benchmarks/sensenova_u1/kv_layout/prefix_strides | tar -x -C "$stride_archive"
    stride_archive="$stride_archive/benchmarks/sensenova_u1/kv_layout/prefix_strides"
    stride_run=$(mktemp -d)
    cp "$stride_archive"/*.py "$stride_run/"
    export OMP_NUM_THREADS=4
    for sample in before-r1 after-r1 after-r2 before-r2; do
        revision=${sample%-*}
        PYTHONPATH="$PWD" TRITON_CACHE_DIR="$stride_run/cache-$sample" python "$stride_run/benchmark_strides.py" \
            --kernel-source "$stride_run/kernel_$revision.py" --output "$stride_run/$sample.json"
    done
    PYTHONPATH="$PWD" python "$stride_run/benchmark_warm_gpu.py"

The scripts and results are separate from the upstream production patch.
Raw numeric samples are unchanged. Only local machine paths in JSON metadata were
replaced with basenames. Benchmark scripts gained SPDX headers for publication.
The unrelated local Omni generated version file still reported 0.31; actual source
was the commit pair above, and imported vLLM was 0.30.0 for every recorded run.
