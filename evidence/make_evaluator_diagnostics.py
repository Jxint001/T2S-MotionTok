#!/usr/bin/env python3
"""Build per-sample evaluator diagnostic tables and optional GT/Ours frame slices."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from common import as_float, ensure_out_dir, read_json, wants_format, write_csv, write_markdown

BASE_COLUMNS = [
    "sample_id",
    "gt_text",
    "bt_output_from_gt_pose",
    "bt_output_from_ours_pose",
    "bt_wer",
    "duration_ratio",
    "gt_jerk",
    "ours_jerk",
    "jerk_ratio",
    "slice_figure",
    "note",
]


def read_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.cases_json:
        data = read_json(args.cases_json)
        if isinstance(data, list):
            return [dict(x) for x in data]
        if isinstance(data, dict) and isinstance(data.get("cases"), list):
            return [dict(x) for x in data["cases"]]
        if isinstance(data, dict):
            return [dict(data)]
        raise ValueError("cases JSON must be an object, a list, or an object with a cases list")
    if args.cases_csv:
        with args.cases_csv.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    raise ValueError("provide --cases-json or --cases-csv")


def pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def nested_jerk(row: dict[str, Any], label: str) -> float | None:
    stats = row.get("motion_stats")
    if isinstance(stats, dict) and isinstance(stats.get(label), dict):
        return as_float(stats[label].get("jerk"))
    return None


def load_pose_file(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import torch

    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise ValueError(f"pose file must contain a dict keyed by sample id: {path}")
    return data


def extract_pose(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.float()
    if isinstance(value, dict):
        for key in ("pose", "poses", "skel", "skeleton", "keypoints"):
            item = value.get(key)
            if isinstance(item, torch.Tensor):
                return item.float()
    return None


def jerk_value(pose) -> float | None:
    if pose is None or pose.shape[0] < 4:
        return None
    import torch

    diff = torch.diff(pose.float(), n=3, dim=0)
    flat = diff.reshape(diff.shape[0], -1)
    return float(torch.linalg.norm(flat, dim=1).mean().item())


def duration_ratio(gt_pose, ours_pose) -> float | None:
    if gt_pose is None or ours_pose is None or gt_pose.shape[0] == 0:
        return None
    return float(ours_pose.shape[0] / gt_pose.shape[0])


def frame_at(pose, rel: float):
    idx = int(round(rel * (pose.shape[0] - 1)))
    return pose[idx]


def render_slice_figure(sample_id: str, gt_pose, ours_pose, out_dir: Path) -> str | None:
    if gt_pose is None or ours_pose is None:
        return None
    import matplotlib.pyplot as plt

    rels = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(rels), figsize=(11, 4.2), dpi=170)
    for row_idx, (label, pose) in enumerate((("GT", gt_pose), ("Ours", ours_pose))):
        for col_idx, rel in enumerate(rels):
            ax = axes[row_idx][col_idx]
            frame = frame_at(pose, rel).reshape(-1, pose.shape[-1])
            x = frame[:, 0].numpy()
            y = frame[:, 1].numpy()
            ax.scatter(x, -y, s=5, c="#171717", alpha=0.9)
            ax.set_title(f"{label} {int(rel * 100)}%", fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
    fig.tight_layout(pad=0.3)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sample_id)
    path = out_dir / f"diagnostic_slices_{safe}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def normalize_case(row: dict[str, Any], gt_poses: dict[str, Any], ours_poses: dict[str, Any], render: bool, out_dir: Path) -> dict[str, Any]:
    sample_id = str(pick(row, "sample_id", "id", "name") or "unknown")
    gt_pose = extract_pose(gt_poses.get(sample_id)) if gt_poses else None
    ours_pose = extract_pose(ours_poses.get(sample_id)) if ours_poses else None
    gt_jerk = as_float(pick(row, "gt_jerk", "jerk_gt"))
    ours_jerk = as_float(pick(row, "ours_jerk", "jerk_ours"))
    if gt_jerk is None:
        gt_jerk = nested_jerk(row, "GT") or jerk_value(gt_pose)
    if ours_jerk is None:
        ours_jerk = nested_jerk(row, "Ours") or jerk_value(ours_pose)
    dur = as_float(pick(row, "duration_ratio"))
    if dur is None:
        dur = duration_ratio(gt_pose, ours_pose)
    jerk_ratio = None
    if gt_jerk is not None and ours_jerk is not None and gt_jerk != 0:
        jerk_ratio = ours_jerk / gt_jerk
    figure = None
    if render:
        figure = render_slice_figure(sample_id, gt_pose, ours_pose, out_dir / "diagnostic_slices")
    return {
        "sample_id": sample_id,
        "gt_text": pick(row, "gt_text", "reference_text", "text") or "",
        "bt_output_from_gt_pose": pick(row, "bt_output_from_gt_pose", "bt_gt", "gt_bt_text") or "",
        "bt_output_from_ours_pose": pick(row, "bt_output_from_ours_pose", "bt_ours", "ours_bt_text") or "",
        "bt_wer": as_float(pick(row, "bt_wer", "wer")),
        "duration_ratio": dur,
        "gt_jerk": gt_jerk,
        "ours_jerk": ours_jerk,
        "jerk_ratio": jerk_ratio,
        "slice_figure": figure or pick(row, "slice_figure", "figure") or "",
        "note": pick(row, "note", "caption_sentence") or "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases-json", type=Path)
    source.add_argument("--cases-csv", type=Path)
    parser.add_argument("--gt-pose-pt", type=Path, default=None)
    parser.add_argument("--ours-pose-pt", type=Path, default=None)
    parser.add_argument("--render-slices", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evidence"))
    parser.add_argument("--format", default="csv,md", help="Comma-separated outputs: csv, md, or csv,md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    cases = read_cases(args)
    gt_poses = load_pose_file(args.gt_pose_pt)
    ours_poses = load_pose_file(args.ours_pose_pt)
    if args.render_slices and (not gt_poses or not ours_poses):
        raise SystemExit("--render-slices requires both --gt-pose-pt and --ours-pose-pt")
    rows = [normalize_case(row, gt_poses, ours_poses, args.render_slices, out_dir) for row in cases]
    if wants_format(args.format, "csv"):
        write_csv(out_dir / "evaluator_diagnostics.csv", rows, BASE_COLUMNS)
    if wants_format(args.format, "md"):
        write_markdown(out_dir / "evaluator_diagnostics.md", rows, BASE_COLUMNS, "Evaluator Diagnostic Examples")


if __name__ == "__main__":
    main()
