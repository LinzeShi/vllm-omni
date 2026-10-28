PR #8437: HunyuanImage3 SigLIP2 CUDA graph evidence
================================================

Experiment date: 2026-10-27
PR: https://github.com/vllm-project/vllm-omni/pull/8437
Download both archives from: https://github.com/LinzeShi/vllm-omni/releases/tag/pr8437-fullmodel-20261027

This is a record of the experiment, including unresolved output differences.
Its analysis outcome is REVIEW_REQUIRED, not a claim of full generation parity.

Sources
-------
Omni base: ee37ec0620a99b3787dab5e5e3bd1f5cdfd0615f
PR source head: dc295e58c10a960fc96471c6de9ee1ac27b596ec
Candidate tree: 2f0debbfcfeba0620b4bf337ea523290249a5645
The candidate is the frozen base with this PR's patch applied.
Model: tencent/HunyuanImage-3.0-Instruct
Model revision: 2ec2c78bee7d4b94157341fba86c4c2c7b1858b2
Hardware: 4x A100-SXM4-80GB, TP=4, DP=1, BF16, default communication.
vLLM 0.32.0; full dependency versions and execution commands are in the bundle.

Results
-------
The base, candidate with graphs disabled, and candidate with graphs enabled
each had two independent launches. Twelve cases covered single and double
image inputs with two seeds, and all formal requests completed naturally.

Corresponding image embeddings matched bitwise across all four ranks and
six launches. Switching OFF/ON/ON/OFF in one process produced identical full
outputs. Independent launches differed on both the unmodified base and the
candidate with graphs enabled. The cause remains unresolved; the same-process
control retains startup graph allocation and cannot exclude an initialization
effect. The raw outputs and comparisons preserve these differences.

Encoding one image saved about 8.5 ms in both runs; two image inputs showed
no consistent benefit. In the first OFF/ON pair, complete outputs matched
and request medians differed by less than 0.4%. The second pair generated
different lengths and cannot isolate encoder speedup. No repeatable overall
request speedup is established. Graph reserve increased by 102 MiB/GPU.

Downloads
---------
review-evidence.tar.gz: 23,365,324 bytes (22.28 MiB).
Scripts, image inputs, full responses, logs, timing distributions, reports,
and the file inventory. Sufficient for reading the results; not sufficient
for running the complete verification commands by itself.

runtime-artifacts.tar.gz: 922,453,008 bytes (879.72 MiB).
Raw image embeddings, exact source snapshots and compiler caches required
by the packaged full verification and reproduction procedures.

These archives are Release assets; they are not stored as Git blobs.
No checkpoint weights are included. Failed and truncated preliminary runs
are retained separately from the formal comparison; see RUN_HISTORY.txt.

Verify and inspect
------------------
Download both archives and SHA256SUMS into the same directory. Use the
SHA256SUMS from the fixed evidence commit linked in the PR, then run:

  sha256sum -c SHA256SUMS
  tar -xzf review-evidence.tar.gz
  tar -xzf runtime-artifacts.tar.gz
  cd evidence
  python verify_files.py
  python recompute.py
  OMP_NUM_THREADS=4 python audit_raw.py

The checks require Python with PyTorch installed, but no GPU. They fail if
required files are missing. Both reports are recomputed from saved outputs;
this does not run a new model inference experiment. audit_raw.py writes
independent-audit-recomputed.json without overwriting the original report.

Start with evidence/README.txt for the complete method and limitations,
results/analysis.json for timing distributions and the original comparisons,
and independent-audit.json for the separate audit. GPU reproduction steps
are in evidence/README.txt and require a separately provisioned compatible host.
