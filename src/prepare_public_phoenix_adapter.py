#!/usr/bin/env python3
"""Prepare PHOENIX14T public G2P data for the local RVQ-token pipeline.

The G2P-DDM public format stores each frame as 151 flattened values and its
loader drops the last value, yielding 150D = 50 joints x 3 coordinates.  The
local RVQ scripts are currently wired to SLRTP-style [T, 178, 3] tensors, so
this adapter places the 50 public joints in the first 50 slots and zero-fills
the remaining joints.  Generated poses can later be exported back by taking
the first 50 joints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


SPLITS = ("train", "dev", "test")


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_skel_line(line: str) -> torch.Tensor:
    values = torch.tensor([float(x) for x in line.strip().split()], dtype=torch.float32)
    if values.numel() == 0 or values.numel() % 151 != 0:
        raise ValueError(f"bad skels line with {values.numel()} values")
    frames_151 = values.view(-1, 151)
    pose_50x3 = frames_151[:, :150].contiguous().view(-1, 50, 3)
    pose_178x3 = torch.zeros((pose_50x3.shape[0], 178, 3), dtype=torch.float32)
    pose_178x3[:, :50, :] = pose_50x3
    return pose_178x3


def allocate_uniform_lengths(frame_len: int, gloss: str) -> list[int]:
    toks = [tok for tok in gloss.split() if tok]
    if not toks:
        return [frame_len]
    base = frame_len // len(toks)
    rem = frame_len % len(toks)
    return [base + (1 if i < rem else 0) for i in range(len(toks))]


def convert_split(phoenix_dir: Path, out_data_dir: Path, split: str) -> dict[str, Any]:
    files = read_lines(phoenix_dir / f"{split}.files")
    glosses = read_lines(phoenix_dir / f"{split}.gloss")
    texts = read_lines(phoenix_dir / f"{split}.text")
    skels = read_lines(phoenix_dir / f"{split}.skels")
    if not (len(files) == len(glosses) == len(texts) == len(skels)):
        raise ValueError(
            f"{split} line count mismatch: files={len(files)} gloss={len(glosses)} "
            f"text={len(texts)} skels={len(skels)}"
        )

    rows: dict[str, dict[str, Any]] = {}
    lengths: list[int] = []
    gloss_lengths: list[int] = []
    for idx, (file_id, gloss, text, skel_line) in enumerate(zip(files, glosses, texts, skels)):
        sample_id = file_id.strip() or f"{split}_{idx:05d}"
        if sample_id in rows:
            sample_id = f"{sample_id}__{idx:05d}"
        pose = parse_skel_line(skel_line)
        rows[sample_id] = {
            "sample_id": sample_id,
            "name": sample_id,
            "original_file": file_id,
            "gloss": gloss,
            "text": text,
            "poses_3d": pose,
            "source_pose_format": "phoenix_151_flat_drop_last_to_50x3_padded_to_178x3",
            "source_pose_joints": 50,
        }
        lengths.append(int(pose.shape[0]))
        gloss_lengths.append(len([tok for tok in gloss.split() if tok]))

    out_data_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out_data_dir / f"{split}.pt")
    return {
        "split": split,
        "count": len(rows),
        "frame_len_min": min(lengths) if lengths else 0,
        "frame_len_max": max(lengths) if lengths else 0,
        "frame_len_mean": sum(lengths) / max(len(lengths), 1),
        "gloss_len_min": min(gloss_lengths) if gloss_lengths else 0,
        "gloss_len_max": max(gloss_lengths) if gloss_lengths else 0,
        "gloss_len_mean": sum(gloss_lengths) / max(len(gloss_lengths), 1),
        "pt_path": str(out_data_dir / f"{split}.pt"),
    }


def build_vocab_and_leng(phoenix_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    vocab_tokens: set[str] = set()
    leng_summary: dict[str, Any] = {}
    for split in SPLITS:
        glosses = read_lines(phoenix_dir / f"{split}.gloss")
        skels = read_lines(phoenix_dir / f"{split}.skels")
        leng_lines = []
        for gloss, skel_line in zip(glosses, skels):
            values = skel_line.strip().split()
            if len(values) % 151 != 0:
                raise ValueError(f"bad {split}.skels line with {len(values)} values")
            frame_len = len(values) // 151
            vocab_tokens.update(tok for tok in gloss.split() if tok)
            leng_lines.append(" ".join(str(x) for x in allocate_uniform_lengths(frame_len, gloss)))
        (out_dir / f"{split}.leng").write_text("\n".join(leng_lines) + "\n", encoding="utf-8")
        leng_summary[split] = {"rows": len(leng_lines), "source": "uniform per-sample gloss/frame allocation"}

    vocab_lines = ["<pad>", "<unk>", "<s>", "</s>"] + sorted(vocab_tokens)
    (out_dir / "src_vocab.txt").write_text("\n".join(vocab_lines) + "\n", encoding="utf-8")

    for split in SPLITS:
        for suffix in ("files", "gloss", "skels", "text"):
            src = phoenix_dir / f"{split}.{suffix}"
            dst = out_dir / f"{split}.{suffix}"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)

    return {
        "src_vocab": str(out_dir / "src_vocab.txt"),
        "vocab_size_including_specials": len(vocab_lines),
        "leng": leng_summary,
        "note": ".leng is generated by uniform allocation, not an official PHOENIX alignment file.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--phoenix-dir", type=Path, default=default_root / "data" / "public_phoenix" / "phoenix")
    parser.add_argument(
        "--out-project-root",
        type=Path,
        default=default_root / "outputs" / "public_phoenix_project",
    )
    parser.add_argument(
        "--g2p-data-out",
        type=Path,
        default=default_root / "outputs" / "g2p_public_phoenix_data",
    )
    args = parser.parse_args()

    out_data_dir = args.out_project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data"
    split_summaries = [convert_split(args.phoenix_dir, out_data_dir, split) for split in SPLITS]
    g2p_summary = build_vocab_and_leng(args.phoenix_dir, args.g2p_data_out)
    source_hashes = {
        f"{split}.{suffix}": file_sha1(args.phoenix_dir / f"{split}.{suffix}")
        for split in SPLITS
        for suffix in ("files", "gloss", "skels", "text")
    }
    manifest = {
        "created_for": "public_phoenix14t_g2p_adapter",
        "phoenix_dir": str(args.phoenix_dir),
        "out_project_root": str(args.out_project_root),
        "out_data_dir": str(out_data_dir),
        "g2p_data_out": str(args.g2p_data_out),
        "paper_split_counts_expected": {"train": 7096, "dev": 519, "test": 642},
        "split_summaries": split_summaries,
        "g2p_auxiliary": g2p_summary,
        "source_sha1": source_hashes,
        "pose_mapping": {
            "source": "G2P-DDM/ProgressiveTransformersSLP style flattened .skels",
            "source_values_per_frame": 151,
            "used_values_per_frame": 150,
            "used_joints": 50,
            "local_tensor_shape": "[T, 178, 3]",
            "mapping": "local[:, :50, :] = source[:, :150].reshape(T, 50, 3); local[:, 50:, :] = 0",
            "export_back": "take local[:, :50, :].reshape(T, 150) and append one dummy value per frame if .skels is needed",
        },
        "clean_boundary": {
            "train_uses": "train.gloss/train.skels only",
            "dev_test_uses": "dev/test GT gloss as G2P input; dev/test pose only for evaluation/export inspection",
            "uses_text2gloss": False,
            "uses_test_for_training_or_model_selection": False,
        },
    }
    write_json(args.out_project_root / "public_phoenix_adapter_manifest.json", manifest)
    write_json(args.g2p_data_out / "public_phoenix_adapter_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
