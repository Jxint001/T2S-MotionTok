#!/usr/bin/env python3
"""Visualize GT, released winner-GUS, and coarse prior/coarse prior test decode poses side by side."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


HAND_INDEX = list(range(8, 50))
EPS = 1e-8

GUS_WEIGHT_GRID = {
    "gus_balanced": {"alpha": 0.7, "beta": 0.3, "lambda_transition": 0.5, "gamma": 1.0, "delta": 0.25, "rho": 2.0},
    "gus_join_heavy": {"alpha": 0.4, "beta": 0.2, "lambda_transition": 1.0, "gamma": 1.0, "delta": 0.5, "rho": 3.0},
    "gus_duration": {"alpha": 1.0, "beta": 0.1, "lambda_transition": 0.25, "gamma": 0.5, "delta": 0.1, "rho": 1.0},
    "gus_motion": {"alpha": 0.3, "beta": 1.0, "lambda_transition": 0.25, "gamma": 0.5, "delta": 0.2, "rho": 1.0},
}


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_pickle(path: Path) -> Any:
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)


def to_np_pose(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1:] != (178, 3):
        raise ValueError(f"bad pose shape: {arr.shape}")
    return arr


def normalize_gloss(gloss: str) -> str:
    return str(gloss).strip().upper()


def split_gloss_sequence(gloss_text: str) -> list[str]:
    return [normalize_gloss(t) for t in str(gloss_text).split() if str(t).strip()]


def motion_amount(seg: np.ndarray) -> float:
    arr = to_np_pose(seg)
    if len(arr) < 2:
        return 0.0
    diff = arr[1:] - arr[:-1]
    return float(np.linalg.norm(diff, axis=2).mean(axis=1).sum())


def endpoint_cost(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(to_np_pose(a)[-1] - to_np_pose(b)[0], axis=1).mean())


def velocity_cost(a: np.ndarray, b: np.ndarray) -> float:
    aa = to_np_pose(a)
    bb = to_np_pose(b)
    if len(aa) < 2 or len(bb) < 2:
        return 0.0
    va = aa[-1] - aa[-2]
    vb = bb[1] - bb[0]
    return float(np.linalg.norm(va - vb, axis=1).mean())


def hand_boundary_cost(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(to_np_pose(a)[-1, HAND_INDEX] - to_np_pose(b)[0, HAND_INDEX], axis=1).mean())


def compute_dict_features(dictionary: dict[str, list[np.ndarray]], max_candidates: int = 5) -> tuple[dict[tuple[str, int], dict[str, float]], dict[str, float], dict[str, float], tuple[np.ndarray, dict[str, Any]]]:
    features: dict[tuple[str, int], dict[str, float]] = {}
    durations_by_gloss: dict[str, float] = {}
    motions_by_gloss: dict[str, float] = {}
    all_units: list[tuple[str, int, np.ndarray, dict[str, float]]] = []
    for gloss, candidates in dictionary.items():
        selected = candidates[:max_candidates] if max_candidates else candidates
        durations = [len(c) for c in selected]
        motions = [motion_amount(c) for c in selected]
        durations_by_gloss[gloss] = float(np.median(durations)) if durations else 1.0
        motions_by_gloss[gloss] = float(np.median(motions)) if motions else 1.0
        for idx, seg in enumerate(selected):
            feat = {"len": float(len(seg)), "motion": motion_amount(seg)}
            features[(gloss, idx)] = feat
            all_units.append((gloss, idx, seg, feat))
    return features, durations_by_gloss, motions_by_gloss, choose_fallback_unit(all_units)


def choose_fallback_unit(all_units: list[tuple[str, int, np.ndarray, dict[str, float]]]) -> tuple[np.ndarray, dict[str, Any]]:
    if not all_units:
        return np.zeros((8, 178, 3), dtype=np.float32), {"gloss": None, "candidate_index": None}
    lengths = [u[3]["len"] for u in all_units]
    motions = [u[3]["motion"] for u in all_units]
    med_len = float(np.median(lengths))
    med_motion = float(np.median(motions))
    best = min(
        all_units,
        key=lambda u: abs(u[3]["len"] - med_len) / max(med_len, EPS)
        + abs(u[3]["motion"] - med_motion) / max(med_motion, EPS),
    )
    return best[2], {"gloss": best[0], "candidate_index": best[1], "length": best[3]["len"], "motion": best[3]["motion"]}


def local_cost(gloss: str, feat: dict[str, float], dur_med: dict[str, float], mot_med: dict[str, float], weights: dict[str, float]) -> float:
    d = abs(feat["len"] - dur_med.get(gloss, feat["len"])) / max(dur_med.get(gloss, feat["len"]), EPS)
    m = abs(feat["motion"] - mot_med.get(gloss, feat["motion"])) / max(mot_med.get(gloss, feat["motion"]), EPS)
    return weights["alpha"] * d + weights["beta"] * m


def trans_cost(a: np.ndarray, b: np.ndarray, weights: dict[str, float]) -> float:
    return (
        weights["gamma"] * endpoint_cost(a, b)
        + weights["delta"] * velocity_cost(a, b)
        + weights["rho"] * hand_boundary_cost(a, b)
    )


def select_units_gus(
    glosses: list[str],
    dictionary: dict[str, list[np.ndarray]],
    features: dict[tuple[str, int], dict[str, float]],
    dur_med: dict[str, float],
    mot_med: dict[str, float],
    fallback: tuple[np.ndarray, dict[str, Any]],
    weights: dict[str, float],
    max_candidates: int = 5,
) -> tuple[list[np.ndarray], list[str], list[int]]:
    present: list[str] = []
    missing: list[str] = []
    for gloss in glosses:
        if dictionary.get(gloss):
            present.append(gloss)
        else:
            missing.append(gloss)
    if not present:
        return [fallback[0]], missing, []
    cand_lists = [dictionary[g][:max_candidates] for g in present]
    dp: list[list[float]] = []
    back: list[list[int]] = []
    first_gloss = present[0]
    first_row = [local_cost(first_gloss, features[(first_gloss, k)], dur_med, mot_med, weights) for k in range(len(cand_lists[0]))]
    dp.append(first_row)
    back.append([-1] * len(first_row))
    for j in range(1, len(present)):
        gloss = present[j]
        row: list[float] = []
        brow: list[int] = []
        for k, seg in enumerate(cand_lists[j]):
            lc = local_cost(gloss, features[(gloss, k)], dur_med, mot_med, weights)
            best_val = None
            best_i = 0
            for i, prev_seg in enumerate(cand_lists[j - 1]):
                val = dp[j - 1][i] + weights["lambda_transition"] * trans_cost(prev_seg, seg, weights) + lc
                if best_val is None or val < best_val:
                    best_val = val
                    best_i = i
            row.append(float(best_val))
            brow.append(best_i)
        dp.append(row)
        back.append(brow)
    last = int(np.argmin(np.asarray(dp[-1], dtype=np.float64)))
    path = [last]
    for j in range(len(present) - 1, 0, -1):
        last = back[j][last]
        path.append(last)
    path = list(reversed(path))
    selected = [cand_lists[j][idx] for j, idx in enumerate(path)]
    return selected, missing, path


def concat_units(units: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([to_np_pose(u) for u in units], axis=0).astype(np.float32)


def load_winner_dictionary(path: Path) -> dict[str, list[np.ndarray]]:
    raw = load_pickle(path)
    return {normalize_gloss(g): [to_np_pose(seg) for seg in segs] for g, segs in raw.items()}


def winner_pose_for_sample(
    sample_id: str,
    t2g: dict[str, dict[str, str]],
    dictionary: dict[str, list[np.ndarray]],
    features: dict[tuple[str, int], dict[str, float]],
    dur_med: dict[str, float],
    mot_med: dict[str, float],
    fallback: tuple[np.ndarray, dict[str, Any]],
    weights_name: str,
    max_candidates: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    pred_gloss = str(t2g[sample_id]["gls_hyp"])
    glosses = split_gloss_sequence(pred_gloss)
    units, missing, path = select_units_gus(
        glosses,
        dictionary,
        features,
        dur_med,
        mot_med,
        fallback,
        GUS_WEIGHT_GRID[weights_name],
        max_candidates=max_candidates,
    )
    return concat_units(units), {"pred_gloss": pred_gloss, "missing": missing, "candidate_path": path}


def pose_limits(poses: list[np.ndarray]) -> tuple[float, float, float, float]:
    xy = np.concatenate([p[:, :, :2].reshape(-1, 2) for p in poses], axis=0)
    xy = xy[np.isfinite(xy).all(axis=1)]
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    cx = float((lo[0] + hi[0]) / 2)
    cy = float((lo[1] + hi[1]) / 2)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-3))
    span *= 1.2
    return cx - span / 2, cx + span / 2, cy - span / 2, cy + span / 2


def frame_at_relative_time(pose: np.ndarray, rel: float) -> np.ndarray:
    idx = int(round(rel * max(len(pose) - 1, 0)))
    return pose[min(max(idx, 0), len(pose) - 1)]


def draw_pose(ax: Any, frame: np.ndarray, title: str, limits: tuple[float, float, float, float]) -> None:
    ax.scatter(frame[:, 0], frame[:, 1], s=5, c="black", alpha=0.9, linewidths=0)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig_to_image(fig: Any) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    return Image.fromarray(data.copy())


def make_gif(sample_id: str, poses: dict[str, np.ndarray], out_path: Path, num_frames: int, fps: float) -> None:
    labels = list(poses.keys())
    limits = pose_limits([poses[k] for k in labels])
    frames: list[Image.Image] = []
    for i in range(num_frames):
        rel = i / max(num_frames - 1, 1)
        fig, axes = plt.subplots(1, len(labels), figsize=(3.0 * len(labels), 3.2), dpi=120)
        for ax, label in zip(axes, labels):
            pose = poses[label]
            frame = frame_at_relative_time(pose, rel)
            idx = int(round(rel * max(len(pose) - 1, 0)))
            draw_pose(ax, frame, f"{label} f{idx}/{len(pose)-1}", limits)
        fig.suptitle(f"{sample_id} relative frame {i}/{num_frames-1}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        frames.append(fig_to_image(fig))
        plt.close(fig)
    duration_ms = int(round(1000.0 / fps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def make_slices(sample_id: str, poses: dict[str, np.ndarray], out_path: Path, rels: list[float]) -> None:
    labels = list(poses.keys())
    limits = pose_limits([poses[k] for k in labels])
    fig, axes = plt.subplots(len(labels), len(rels), figsize=(2.3 * len(rels), 2.1 * len(labels)), dpi=140)
    if len(labels) == 1:
        axes = np.expand_dims(axes, 0)
    for r, label in enumerate(labels):
        pose = poses[label]
        for c, rel in enumerate(rels):
            idx = int(round(rel * max(len(pose) - 1, 0)))
            title = f"{label} {int(rel * 100):02d}% f{idx}"
            draw_pose(axes[r, c], pose[idx], title, limits)
    fig.suptitle(sample_id, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def motion_stats(pose: np.ndarray) -> dict[str, float]:
    if len(pose) < 2:
        return {"frames": float(len(pose)), "path": 0.0, "mean_step": 0.0}
    step = np.linalg.norm(pose[1:] - pose[:-1], axis=2).mean(axis=1)
    return {"frames": float(len(pose)), "path": float(step.sum()), "mean_step": float(step.mean())}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def safe_stem(text: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text).strip())
    return stem.strip("_") or "pred"


def run(args: argparse.Namespace) -> None:
    gt = load_torch(args.gt_test_pt)
    model_preds = [(args.ours_label, load_torch(args.ours_pred_pt))]
    for item in args.extra_pred:
        if "=" not in item:
            raise ValueError(f"--extra-pred must be label=/path/to/pred.pt, got: {item}")
        label, path = item.split("=", 1)
        model_preds.append((label.strip(), load_torch(Path(path))))
    t2g = load_pickle(args.winner_t2g_pkl)
    dictionary = load_winner_dictionary(args.winner_g2p_pkl)
    features, dur_med, mot_med, fallback = compute_dict_features(dictionary, max_candidates=args.max_candidates)

    sample_ids = args.sample_id or [
        "28January_2010_Thursday_heute-2882",
        "26July_2010_Monday_tagesschau-6266",
        "27April_2010_Tuesday_heute-1028",
    ]
    summaries: dict[str, Any] = {}
    for sample_id in sample_ids:
        if sample_id not in gt:
            raise KeyError(f"{sample_id} not in GT split")
        if sample_id not in t2g:
            raise KeyError(f"{sample_id} not in winner T2G predictions")
        for label, pred in model_preds:
            if sample_id not in pred:
                raise KeyError(f"{sample_id} not in {label} prediction")
        winner_pose, winner_meta = winner_pose_for_sample(
            sample_id,
            t2g,
            dictionary,
            features,
            dur_med,
            mot_med,
            fallback,
            args.weights_name,
            args.max_candidates,
        )
        poses = {
            "GT": to_np_pose(gt[sample_id]["poses_3d"]),
            f"winner-{args.weights_name}": winner_pose,
        }
        for label, pred in model_preds:
            poses[label] = to_np_pose(pred[sample_id])
        stem = f"{sample_id}_gt_winner_{safe_stem(args.ours_label).lower()}"
        gif_path = args.out_dir / f"{stem}.gif"
        meta_path = args.out_dir / f"{stem}_meta.json"
        make_gif(sample_id, poses, gif_path, num_frames=args.gif_frames, fps=args.fps)
        png_path = None
        if not args.skip_slices:
            png_path = args.out_dir / f"{stem}_slices.png"
            make_slices(sample_id, poses, png_path, rels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        summaries[sample_id] = {
            "gif": str(gif_path),
            "slices_png": str(png_path) if png_path is not None else None,
            "meta": str(meta_path),
            "text": str(gt[sample_id].get("text", "")),
            "gt_gloss": str(gt[sample_id].get("gloss", "")),
            "winner": winner_meta,
            "lengths": {label: int(pose.shape[0]) for label, pose in poses.items()},
            "motion_stats": {label: motion_stats(pose) for label, pose in poses.items()},
        }
        write_json(meta_path, summaries[sample_id])
    write_json(args.out_dir / "pose_comparison_gt_winner_coarse_prior_visual_summary.json", summaries)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-test-pt", type=Path, default=root / "data" / "slrtp" / "test.pt")
    parser.add_argument(
        "--ours-pred-pt",
        type=Path,
        default=root / "outputs" / "predictions" / "ours.pt",
    )
    parser.add_argument("--ours-label", default="coarse prior")
    parser.add_argument("--extra-pred", action="append", default=[])
    parser.add_argument("--winner-g2p-pkl", type=Path, default=root / "data" / "winner_preextracted" / "phoenix_gloss2pose_results.pkl")
    parser.add_argument(
        "--winner-t2g-pkl",
        type=Path,
        default=root / "data" / "winner_preextracted" / "phoenix_text2gloss_results.pkl",
    )
    parser.add_argument("--out-dir", type=Path, default=root / "outputs" / "pose_comparison_visuals")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--weights-name", choices=sorted(GUS_WEIGHT_GRID), default="gus_join_heavy")
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--gif-frames", type=int, default=48)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--skip-slices", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
