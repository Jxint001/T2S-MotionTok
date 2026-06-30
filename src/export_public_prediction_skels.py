#!/usr/bin/env python3
"""Export local [T,178,3] public PHOENIX predictions back to .skels format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def pose_to_skel_line(pose: torch.Tensor, dummy_value: float) -> str:
    pose = torch.as_tensor(pose, dtype=torch.float32)
    if pose.ndim != 3 or tuple(pose.shape[1:]) != (178, 3):
        raise ValueError(f"bad local pose shape: {tuple(pose.shape)}")
    flat150 = pose[:, :50, :].contiguous().view(pose.shape[0], 150)
    dummy = torch.full((pose.shape[0], 1), float(dummy_value), dtype=torch.float32)
    flat151 = torch.cat([flat150, dummy], dim=1).reshape(-1)
    return " ".join(f"{float(x):.6f}" for x in flat151)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-pt", type=Path, required=True)
    parser.add_argument("--phoenix-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev", "test"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dummy-value", type=float, default=0.0)
    args = parser.parse_args()

    pred = load_torch(args.prediction_pt)
    files = read_lines(args.phoenix_dir / f"{args.split}.files")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_skel = args.out_dir / f"{args.split}.skels"
    missing = []
    lengths = []
    with out_skel.open("w", encoding="utf-8") as f:
        for sid in files:
            if sid not in pred:
                missing.append(sid)
                f.write("\n")
                continue
            pose = pred[sid]
            lengths.append(int(pose.shape[0]))
            f.write(pose_to_skel_line(pose, args.dummy_value) + "\n")

    for suffix in ("files", "gloss", "text"):
        src = args.phoenix_dir / f"{args.split}.{suffix}"
        dst = args.out_dir / f"{args.split}.{suffix}"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)

    summary = {
        "prediction_pt": str(args.prediction_pt),
        "split": args.split,
        "num_source_files": len(files),
        "num_predictions": len(pred),
        "num_missing": len(missing),
        "missing": missing[:50],
        "output_skels": str(out_skel),
        "format": "151 values/frame; first 150 from local[:, :50, :], last dummy",
        "dummy_value": float(args.dummy_value),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_mean": sum(lengths) / max(len(lengths), 1),
    }
    write_json(args.out_dir / f"{args.split}_export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
