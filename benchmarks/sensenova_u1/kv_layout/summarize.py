# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Summarize the recorded samples or a fresh two-round A/B run."""

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def collect(root: Path) -> dict:
    variants = {}
    for variant in ("base", "candidate"):
        rows = []
        for suffix in ("benchmark", "benchmark-round2"):
            folder = root / f"{variant}-{suffix}"
            requests = [r for r in read_jsonl(folder / "requests.jsonl") if not r["warmup"]]
            loops = [
                r
                for path in folder.glob("loops-*.jsonl")
                for r in read_jsonl(path)
                if r["steps"] == 50 and r["size"] == [1024, 1024] and r["index"] > 1
            ]
            if len(requests) != 5 or len(loops) != 5:
                raise ValueError(f"{folder}: expected five requests and five measured loops")
            if any(r["diagnostic"] or r.get("capture", False) for r in loops):
                raise ValueError(f"{folder}: profiling/capture must be separate from timing")
            for request, loop in zip(requests, loops):
                rows.append({"e2e_ms": request["e2e_ms"], **loop})
        ops = json.loads((root / f"{variant}-trace/operators.json").read_text())
        variants[variant] = {
            "requests": rows,
            "trace_counts": {
                name: ops[name]["count"]
                for name in ("aten::copy_", "aten::_local_scalar_dense", "cudaStreamSynchronize")
                if name in ops
            },
        }
    return variants


def summarize(variants: dict) -> dict:
    result = {}
    for variant, data in variants.items():
        rows = data["requests"]
        if len(rows) != 10:
            raise ValueError(f"{variant}: expected ten measured requests")
        e2e = [r["e2e_ms"] for r in rows]
        result[variant] = {
            "requests": len(rows),
            "e2e_median_ms": statistics.median(e2e),
            "e2e_range_ms": [min(e2e), max(e2e)],
            "mean_step_median_ms": statistics.median(r["denoise_ms"] / 50 for r in rows),
            "peak_allocated_mib": statistics.median(r["peak_allocated_bytes"] / 2**20 for r in rows),
            "trace_counts": data["trace_counts"],
        }
        if all("temporary_peak_bytes" in r for r in rows):
            result[variant]["temporary_peak_mib"] = statistics.median(r["temporary_peak_bytes"] / 2**20 for r in rows)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--samples", type=Path, help="Recorded samples.json; no GPU or model required")
    group.add_argument("--results", type=Path, help="Directory containing fresh base/candidate runs")
    args = parser.parse_args()
    data = json.loads(args.samples.read_text())["variants"] if args.samples else collect(args.results)
    print(json.dumps(summarize(data), indent=2))
