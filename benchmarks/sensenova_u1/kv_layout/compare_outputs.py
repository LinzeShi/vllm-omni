# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Check paired image outputs; optional denoise tensors use exact equality."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def compare(root: Path, suites: tuple[str, ...], capture: bool) -> dict:
    report = {"images": [], "steps": [], "unpaired_startup_steps": [], "think": []}
    for suite in suites:
        base, candidate = root / f"base-{suite}", root / f"candidate-{suite}"
        names = {p.name for p in base.glob("*.png") if p.name != "reference.png"}
        assert names and names == {p.name for p in candidate.glob("*.png") if p.name != "reference.png"}
        for name in sorted(names):
            with Image.open(base / name) as left, Image.open(candidate / name) as right:
                equal = np.array_equal(np.asarray(left), np.asarray(right))
            report["images"].append({"suite": suite, "file": name, "equal": equal})
        if suite != "correctness" or not capture:
            continue
        names = {p.name for p in base.glob("*.pt")}
        assert names and names == {p.name for p in candidate.glob("*.pt")}
        for name in sorted(names):
            x = torch.load(base / name, map_location="cpu", weights_only=True)
            y = torch.load(candidate / name, map_location="cpu", weights_only=True)
            assert x.shape == y.shape and x.dtype == y.dtype
            unpaired = name.startswith("512x512-s2-think0-edit1-n1-step-")
            record = {
                "file": name,
                "equal": torch.equal(x, y),
                "finite": bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                "max_abs": (x.float() - y.float()).abs().max().item(),
            }
            report["unpaired_startup_steps" if unpaired else "steps"].append(record)

        def texts(folder):
            return {
                r["key"]: r["think_text"]
                for p in folder.glob("loops-*.jsonl")
                for line in p.read_text().splitlines()
                if (r := json.loads(line))["think_text"]
            }

        left, right = texts(base), texts(candidate)
        assert left and left.keys() == right.keys()
        report["think"] = [{"case": k, "equal": left[k] == right[k]} for k in left]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    result = compare(args.results, ("benchmark", "benchmark-round2", "correctness", "trace"), True)
    print(json.dumps(result, indent=2))
    assert all(r["equal"] and r.get("finite", True) for key in ("images", "steps", "think") for r in result[key])
