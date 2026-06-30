#!/usr/bin/env python3
"""Select good/bad dev samples for T2S-MotionTok visual inspection."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def toks(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, token_b in enumerate(b, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + int(token_a != token_b),
            )
        prev = cur
    return prev[-1]


def motion_stats(pose: Any) -> dict[str, float]:
    if isinstance(pose, torch.Tensor):
        arr = pose.float()
    else:
        arr = torch.as_tensor(pose, dtype=torch.float32)
    if arr.shape[0] < 2:
        return {"frames": float(arr.shape[0]), "mean_step": 0.0, "jerk": 0.0}
    step = torch.linalg.norm(arr[1:] - arr[:-1], dim=2).mean(dim=1)
    if step.shape[0] < 2:
        jerk = torch.tensor(0.0)
    else:
        jerk = torch.abs(step[1:] - step[:-1]).mean()
    return {"frames": float(arr.shape[0]), "mean_step": float(step.mean()), "jerk": float(jerk)}


def run(args: argparse.Namespace) -> None:
    gt = load_torch(args.gt_dev_pt)
    pred = load_torch(args.pred_pt)
    bt_texts = load_torch(args.text_preds_pt)
    if len(bt_texts) != len(gt):
        raise ValueError(f"text pred count {len(bt_texts)} != dev count {len(gt)}")

    rows: list[dict[str, Any]] = []
    for idx, (sid, row) in enumerate(gt.items()):
        if sid not in pred:
            continue
        ref = str(row.get("text", ""))
        hyp = str(bt_texts[idx])
        ref_tokens = toks(ref)
        hyp_tokens = toks(hyp)
        wer = edit_distance(ref_tokens, hyp_tokens) / max(1, len(ref_tokens))
        gt_len = int(row["poses_3d"].shape[0])
        pred_len = int(pred[sid].shape[0])
        pred_stats = motion_stats(pred[sid])
        gt_stats = motion_stats(row["poses_3d"])
        rows.append(
            {
                "sample_id": sid,
                "text": ref,
                "bt_text": hyp,
                "gt_gloss": str(row.get("gloss", "")),
                "bt_wer": wer,
                "gt_frames": gt_len,
                "pred_frames": pred_len,
                "duration_ratio": pred_len / max(1, gt_len),
                "pred_motion": pred_stats,
                "gt_motion": gt_stats,
            }
        )

    pool = [r for r in rows if args.min_gt_frames <= r["gt_frames"] <= args.max_gt_frames]
    good = sorted(pool, key=lambda r: (r["bt_wer"], abs(r["duration_ratio"] - 1.0), r["pred_motion"]["jerk"]))[: args.num_each]
    bad = sorted(pool, key=lambda r: (-r["bt_wer"], abs(r["duration_ratio"] - 1.0), -r["pred_motion"]["jerk"]))[: args.num_each]
    selected = {"selection_rule": "good=lowest BT WER with sane GT length; bad=highest BT WER with sane GT length", "good": good, "bad": bad}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    print("GOOD")
    for row in good:
        print(f"{row['sample_id']} wer={row['bt_wer']:.3f} gt={row['gt_frames']} pred={row['pred_frames']} ratio={row['duration_ratio']:.2f}")
        print(f"  ref: {row['text']}")
        print(f"  bt : {row['bt_text']}")
    print("BAD")
    for row in bad:
        print(f"{row['sample_id']} wer={row['bt_wer']:.3f} gt={row['gt_frames']} pred={row['pred_frames']} ratio={row['duration_ratio']:.2f}")
        print(f"  ref: {row['text']}")
        print(f"  bt : {row['bt_text']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-dev-pt", type=Path, required=True)
    parser.add_argument("--pred-pt", type=Path, required=True)
    parser.add_argument("--text-preds-pt", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--num-each", type=int, default=2)
    parser.add_argument("--min-gt-frames", type=int, default=35)
    parser.add_argument("--max-gt-frames", type=int, default=170)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
