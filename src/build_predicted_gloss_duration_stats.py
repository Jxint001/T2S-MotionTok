#!/usr/bin/env python3
"""用 train predicted gloss 重新估计 duration stats。

边界：
- 只读取 train split 的 cached RVQ token rows 和 train predicted-gloss json；
- 不读取 dev/test pose 或 test.pt；
- 输出仍是 gloss -> frame/gloss median duration 表，只用于 clean non-BT 对齐诊断。
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rvq_prior_experiment import load_torch, read_json, split_gloss, write_json


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def median(values: list[float]) -> float:
    ordered = sorted(values)
    return float(ordered[len(ordered) // 2])


def pick_predicted_gloss(row: dict[str, Any], rank: int) -> str | None:
    candidates = row.get("candidates", [])
    if candidates:
        if rank >= len(candidates):
            return None
        return candidates[rank].get("pred_gloss")
    return row.get("pred_gloss")


def build_stats(args: argparse.Namespace) -> dict[str, Any]:
    train_rows = load_torch(args.token_cache_dir / "train_rvq_tokens.pt")
    topk_rows = read_json(args.topk_json)
    base_stats = read_json(args.base_duration_json or (args.token_cache_dir / "duration_stats.json"))
    base_global = float(base_stats["global"])
    base_per = {str(key): float(value) for key, value in base_stats["per_gloss"].items()}

    cap_min = base_global * float(args.cap_min_global_mult)
    cap_max = base_global * float(args.cap_max_global_mult)
    rank = int(args.rank)
    observations: dict[str, list[float]] = defaultdict(list)
    ratios: list[float] = []
    missing_topk = 0
    missing_rank = 0
    empty_pred = 0
    used_rows = 0

    for row in train_rows:
        sid = str(row.get("sample_id", ""))
        topk_row = topk_rows.get(sid)
        if topk_row is None:
            missing_topk += 1
            continue
        pred_gloss = pick_predicted_gloss(topk_row, rank)
        if pred_gloss is None:
            missing_rank += 1
            continue
        toks = split_gloss(pred_gloss)
        if not toks:
            empty_pred += 1
            continue
        if "frame_len" not in row:
            raise KeyError(f"train cached row {sid} missing frame_len")
        ratio = float(row["frame_len"]) / len(toks)
        ratio = min(cap_max, max(cap_min, ratio))
        ratios.append(ratio)
        for tok in toks:
            observations[tok].append(ratio)
        used_rows += 1

    if not ratios:
        raise ValueError("no usable train predicted-gloss rows for duration stats")

    pred_global = median(ratios)
    pred_global = min(cap_max, max(cap_min, pred_global))
    per_gloss: dict[str, float] = {}
    counts: dict[str, int] = {}
    all_tokens = set(base_per) | set(observations)
    min_count = int(args.min_count)
    shrink_k = float(args.shrink_k)
    for tok in sorted(all_tokens):
        obs = observations.get(tok, [])
        counts[tok] = len(obs)
        base_value = base_per.get(tok, base_global)
        if len(obs) >= min_count:
            alpha = len(obs) / (len(obs) + shrink_k)
            value = alpha * median(obs) + (1.0 - alpha) * base_value
        else:
            value = base_value
        per_gloss[tok] = float(min(cap_max, max(cap_min, value)))

    return {
        "global": float(pred_global),
        "per_gloss": per_gloss,
        "metadata": {
            "created_at": now_iso(),
            "method": "train_predicted_gloss_rank_duration_with_baseline_shrinkage",
            "token_cache_dir": str(args.token_cache_dir),
            "topk_json": str(args.topk_json),
            "base_duration_json": str(args.base_duration_json or (args.token_cache_dir / "duration_stats.json")),
            "rank": rank,
            "min_count": min_count,
            "shrink_k": shrink_k,
            "cap_min_global_mult": float(args.cap_min_global_mult),
            "cap_max_global_mult": float(args.cap_max_global_mult),
            "cap_min": float(cap_min),
            "cap_max": float(cap_max),
            "base_global": base_global,
            "pred_global": float(pred_global),
            "train_rows": len(train_rows),
            "used_rows": used_rows,
            "missing_topk": missing_topk,
            "missing_rank": missing_rank,
            "empty_pred": empty_pred,
            "tokens_with_pred_observations": len(observations),
            "tokens_total": len(per_gloss),
            "tokens_above_min_count": sum(1 for value in counts.values() if value >= min_count),
            "uses_dev_or_test_targets": False,
            "uses_official_bt_for_training_or_selection": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    parser.add_argument("--topk-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-duration-json", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--shrink-k", type=float, default=5.0)
    parser.add_argument("--cap-min-global-mult", type=float, default=0.5)
    parser.add_argument("--cap-max-global-mult", type=float, default=2.0)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    if args.rank < 0:
        raise ValueError("--rank must be >= 0")
    if args.min_count < 1:
        raise ValueError("--min-count must be >= 1")
    if args.shrink_k < 0:
        raise ValueError("--shrink-k must be >= 0")
    if args.cap_min_global_mult <= 0 or args.cap_max_global_mult <= 0:
        raise ValueError("duration caps must be positive")
    if args.cap_min_global_mult >= args.cap_max_global_mult:
        raise ValueError("--cap-min-global-mult must be smaller than --cap-max-global-mult")
    write_json(args.out, build_stats(args))


if __name__ == "__main__":
    run(parse_args())
