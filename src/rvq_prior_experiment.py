#!/usr/bin/env python3
"""Gloss-conditioned RVQ token prior.

本脚本冻结 Phase2 选出的 RVQ tokenizer，只训练 gloss -> RVQ token prior。

数据边界：
- prior target 只来自 train pose 经冻结 RVQ tokenizer 编码；
- 不使用 B-best/winner pose 训练 prior；
- dev pose 只由 official evaluator 读取；
- test.pt 不读取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from rvq_tokenizer_experiment import (
    denormalize_pose,
    extract_window,
    load_model_from_ckpt,
    load_torch,
    make_starts,
    model_decode_batch,
    normalize_pose,
    posthoc_quantize,
    read_json,
    to_pose,
    window_features,
    write_json,
)


PAD = 0
UNK = 1
SEP = 2


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def split_gloss(text: str) -> list[str]:
    return [tok.strip().upper() for tok in str(text).split() if tok.strip()]


def split_text(text: str) -> list[str]:
    return [tok.strip(" \t\r\n.,;:!?()[]{}\"'").lower() for tok in str(text).split() if tok.strip(" \t\r\n.,;:!?()[]{}\"'")]


def join_gloss(tokens: list[str]) -> str:
    return " ".join(tok for tok in tokens if tok)


def condition_tag(args: argparse.Namespace) -> str:
    return str(getattr(args, "condition_mode", "gloss"))


def build_vocab(train: dict[str, dict[str, Any]], args: argparse.Namespace | None = None) -> dict[str, int]:
    mode = condition_tag(args or argparse.Namespace(condition_mode="gloss"))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    if mode == "gloss":
        for row in train.values():
            for tok in split_gloss(row.get("gloss", "")):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
        return vocab
    vocab["<sep>"] = SEP
    for row in train.values():
        if mode in {"text", "text_gloss"}:
            for tok in split_text(row.get("text", "")):
                key = f"T:{tok}"
                if key not in vocab:
                    vocab[key] = len(vocab)
        if mode in {"gloss", "text_gloss"}:
            for tok in split_gloss(row.get("gloss", "")):
                key = f"G:{tok}"
                if key not in vocab:
                    vocab[key] = len(vocab)
    return vocab


def encode_gloss(text: str, vocab: dict[str, int]) -> torch.Tensor:
    ids = [vocab.get(tok, UNK) for tok in split_gloss(text)]
    if not ids:
        ids = [UNK]
    return torch.tensor(ids, dtype=torch.long)


def encode_condition_values(text: str, gloss: str, vocab: dict[str, int], args: argparse.Namespace) -> torch.Tensor:
    mode = condition_tag(args)
    if mode == "gloss":
        return encode_gloss(gloss, vocab)
    ids: list[int] = []
    if mode in {"text", "text_gloss"}:
        ids.extend(vocab.get(f"T:{tok}", UNK) for tok in split_text(text))
    if mode == "text_gloss":
        ids.append(vocab.get("<sep>", SEP))
    if mode in {"gloss", "text_gloss"}:
        ids.extend(vocab.get(f"G:{tok}", UNK) for tok in split_gloss(gloss))
    if not ids:
        ids = [UNK]
    max_len = int(getattr(args, "max_condition_len", 0) or 0)
    if max_len > 0 and len(ids) > max_len:
        ids = ids[:max_len]
    return torch.tensor(ids, dtype=torch.long)


def encode_condition(row: dict[str, Any], vocab: dict[str, int], args: argparse.Namespace, gloss_override: str | None = None) -> torch.Tensor:
    return encode_condition_values(row.get("text", ""), gloss_override if gloss_override is not None else row.get("gloss", ""), vocab, args)


def condition_boundary(args: argparse.Namespace) -> dict[str, Any]:
    rank = int(getattr(args, "candidate_rank", 0) or 0)
    mode = condition_tag(args)
    if mode == "gloss":
        return {
            "train_condition": "train GT gloss",
            "dev_condition": f"rank-{rank} text2gloss predicted gloss from dev top-k json",
            "uses_train_gt_gloss": True,
            "uses_dev_gt_gloss": False,
            "uses_source_text": False,
        }
    if mode == "text":
        return {
            "train_condition": "train source text",
            "dev_condition": "dev source text from top-k json",
            "uses_train_gt_gloss": False,
            "uses_dev_gt_gloss": False,
            "uses_source_text": True,
        }
    return {
        "train_condition": "train source text + train GT gloss",
        "dev_condition": f"dev source text from top-k json + rank-{rank} text2gloss predicted gloss",
        "uses_train_gt_gloss": True,
        "uses_dev_gt_gloss": False,
        "uses_source_text": True,
    }


def eval_condition_description(args: argparse.Namespace, split: str) -> str:
    rank = int(getattr(args, "candidate_rank", 0) or 0)
    mode = condition_tag(args)
    if mode == "gloss":
        return f"rank-{rank} text2gloss predicted gloss from {split} top-k json"
    if mode == "text":
        return f"{split} source text from top-k json"
    return f"{split} source text from top-k json + rank-{rank} text2gloss predicted gloss"


def topk_path(project_root: Path, split: str, config_id: str) -> Path:
    return project_root / "exps" / "task3" / "outputs" / "task3_followup_abc" / "text2gloss_decoding" / f"{split}_topk_gloss_{config_id}.json"


def predict_frame_len(gloss: str, duration: dict[str, Any], min_len: int, max_len: int) -> int:
    toks = split_gloss(gloss)
    if not toks:
        return min_len
    frames = sum(float(duration["per_gloss"].get(tok, duration["global"])) for tok in toks)
    return int(max(min_len, min(max_len, round(frames))))


def duration_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = []
    per_gloss: dict[str, list[float]] = {}
    for row in rows:
        toks = split_gloss(row["gloss"])
        if not toks:
            continue
        ratio = row["frame_len"] / len(toks)
        ratios.append(ratio)
        for tok in toks:
            per_gloss.setdefault(tok, []).append(ratio)
    ratios = sorted(ratios)
    global_ratio = ratios[len(ratios) // 2] if ratios else 12.0
    return {"global": global_ratio, "per_gloss": {k: sorted(v)[len(v) // 2] for k, v in per_gloss.items()}}


def gloss_token_pool(rows: list[dict[str, Any]]) -> list[str]:
    seen = set()
    pool = []
    for row in rows:
        for tok in split_gloss(row.get("gloss", "")):
            if tok not in seen:
                seen.add(tok)
                pool.append(tok)
    return pool


def corrupt_gloss(gloss: str, token_pool: list[str], rng: random.Random, args: argparse.Namespace) -> str:
    toks = split_gloss(gloss)
    if not toks:
        return gloss
    out = []
    for tok in toks:
        if rng.random() < args.gloss_noise_drop_prob and len(toks) > 1:
            continue
        if token_pool and rng.random() < args.gloss_noise_replace_prob:
            out.append(rng.choice(token_pool))
        else:
            out.append(tok)
    if not out:
        out = [rng.choice(toks)]
    if len(out) > 1 and args.gloss_noise_swap_prob > 0:
        idx = 0
        while idx + 1 < len(out):
            if rng.random() < args.gloss_noise_swap_prob:
                out[idx], out[idx + 1] = out[idx + 1], out[idx]
                idx += 2
            else:
                idx += 1
    return join_gloss(out)


def load_train_predicted_gloss(args: argparse.Namespace) -> tuple[dict[str, str], str | None]:
    copies = int(getattr(args, "train_predicted_gloss_copies", 0) or 0)
    if copies <= 0:
        return {}, None
    path = args.train_predicted_gloss_json or topk_path(args.project_root, "train", args.config_id)
    rows = read_json(path)
    pred = {}
    rank = int(args.train_predicted_gloss_rank)
    for sid, row in rows.items():
        if "candidates" in row:
            candidates = row.get("candidates", [])
            if not candidates:
                continue
            if rank >= len(candidates):
                raise ValueError(f"{sid} has {len(candidates)} train predicted gloss candidates, cannot use rank {rank}")
            pred[sid] = candidates[rank]["pred_gloss"]
        elif row.get("pred_gloss"):
            pred[sid] = row["pred_gloss"]
    return pred, str(path)


def augment_train_rows(args: argparse.Namespace, rows: list[dict[str, Any]], vocab: dict[str, int], train_predicted_gloss: dict[str, str] | None = None) -> list[dict[str, Any]]:
    noise_copies = int(getattr(args, "gloss_noise_copies", 0) or 0)
    pred_copies = int(getattr(args, "train_predicted_gloss_copies", 0) or 0)
    if noise_copies <= 0 and pred_copies <= 0:
        return rows
    if condition_tag(args) == "text":
        raise ValueError("condition augmentation requires condition_mode gloss or text_gloss")
    rng = random.Random(args.seed + 991)
    pool = gloss_token_pool(rows)
    augmented = list(rows)
    if pred_copies > 0:
        if not train_predicted_gloss:
            raise ValueError("train predicted gloss augmentation requested but no train predicted gloss rows were loaded")
        for row in rows:
            sid = row.get("sample_id")
            if sid not in train_predicted_gloss:
                continue
            pred_gloss = train_predicted_gloss[sid]
            for _ in range(pred_copies):
                item = dict(row)
                item["condition_ids"] = encode_condition(row, vocab, args, gloss_override=pred_gloss)
                item["gloss_ids"] = item["condition_ids"]
                item["condition_gloss"] = pred_gloss
                item["condition_augmented"] = "train_predicted_gloss"
                augmented.append(item)
    for row in rows:
        for _ in range(noise_copies):
            noisy = corrupt_gloss(row.get("gloss", ""), pool, rng, args)
            item = dict(row)
            item["condition_ids"] = encode_condition(row, vocab, args, gloss_override=noisy)
            item["gloss_ids"] = item["condition_ids"]
            item["condition_gloss"] = noisy
            item["condition_augmented"] = "synthetic_noise"
            augmented.append(item)
    return augmented


def encode_pose_tokens(model: nn.Module, pose: torch.Tensor, norm: dict[str, torch.Tensor], codebooks: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    norm_pose = normalize_pose(pose, norm)
    starts = make_starts(pose.shape[0], args.window_size, args.stride)
    windows = [extract_window(norm_pose, start, args.window_size) for start in starts]
    tokens = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(windows), args.encode_batch_size):
            batch = torch.stack(windows[offset : offset + args.encode_batch_size], dim=0).to(args.device)
            z = model.encoder(window_features(batch, getattr(model, "feature_args", args)))
            _, idx = posthoc_quantize(z, codebooks.to(args.device))
            tokens.append(idx.cpu())
    return torch.cat(tokens, dim=0)


def decode_overlap_weights(args: argparse.Namespace) -> torch.Tensor:
    window_size = int(args.window_size)
    mode = str(getattr(args, "decode_overlap_window", "uniform"))
    if mode == "uniform":
        weights = torch.ones(window_size, dtype=torch.float32)
    elif mode == "triangular":
        raw = torch.bartlett_window(window_size + 2, periodic=False, dtype=torch.float32)[1:-1]
        weights = raw / raw.max().clamp_min(1e-6)
    elif mode == "hann":
        raw = torch.hann_window(window_size + 2, periodic=False, dtype=torch.float32)[1:-1]
        raw = raw / raw.max().clamp_min(1e-6)
        floor = float(getattr(args, "decode_overlap_floor", 0.25))
        weights = floor + (1.0 - floor) * raw
    else:
        raise ValueError(f"unknown --decode-overlap-window: {mode}")
    return weights.clamp_min(1e-6).view(window_size, 1, 1)


def decode_tokens_to_pose(model: nn.Module, tokens: torch.Tensor, norm: dict[str, torch.Tensor], codebooks: torch.Tensor, frame_len: int, args: argparse.Namespace) -> torch.Tensor:
    starts = make_starts(frame_len, args.window_size, args.stride)
    token_len = min(len(tokens), len(starts))
    starts = starts[:token_len]
    tokens = tokens[:token_len]
    recon_sum = torch.zeros(max(frame_len, starts[-1] + args.window_size), 178, 3)
    counts = torch.zeros(recon_sum.shape[0], 1, 1)
    overlap_weights = decode_overlap_weights(args)
    model.eval()
    with torch.no_grad():
        for offset in range(0, token_len, args.decode_batch_size):
            cur = tokens[offset : offset + args.decode_batch_size].to(args.device)
            q = torch.zeros(cur.shape[0], codebooks.shape[-1], device=args.device)
            for layer in range(codebooks.shape[0]):
                q = q + codebooks[layer].to(args.device)[cur[:, layer].clamp(0, codebooks.shape[1] - 1)]
            windows = model.decoder(q).reshape(-1, args.window_size, 178, 3).cpu()
            for local_idx, window in enumerate(windows):
                start = starts[offset + local_idx]
                recon_sum[start : start + args.window_size] += window * overlap_weights
                counts[start : start + args.window_size] += overlap_weights
    recon = recon_sum / counts.clamp_min(1.0)
    return denormalize_pose(recon[:frame_len], norm)


def build_token_rows(args: argparse.Namespace, train: dict[str, dict[str, Any]], vocab: dict[str, int], out_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tag = condition_tag(args)
    suffix = "" if tag == "gloss" else f"_{tag}"
    path = out_dir / "tokenizer" / f"train_rvq_tokens{suffix}.pt"
    dur_path = out_dir / "tokenizer" / f"duration_stats{suffix}.json"
    if path.exists() and dur_path.exists() and not args.force:
        return load_torch(path), read_json(dur_path)
    model, norm, ckpt_data = load_model_from_ckpt(args.tokenizer_ckpt, args)
    codebooks = ckpt_data["posthoc_codebooks"].to(args.device)
    rows = []
    ids = sorted(train)
    if args.max_train_samples > 0:
        ids = ids[: args.max_train_samples]
    for sid in ids:
        row = train[sid]
        pose = to_pose(row["poses_3d"])
        tokens = encode_pose_tokens(model, pose, norm, codebooks, args)
        rows.append(
            {
                "sample_id": sid,
                "text": row.get("text", ""),
                "gloss": row["gloss"],
                "gloss_ids": encode_gloss(row["gloss"], vocab) if tag == "gloss" else encode_condition(row, vocab, args),
                "condition_ids": encode_condition(row, vocab, args),
                "tokens": tokens,
                "frame_len": int(pose.shape[0]),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rows, path)
    dur = duration_stats(rows)
    write_json(dur_path, dur)
    write_json(
        out_dir / "tokenizer" / f"train_rvq_token_summary{suffix}.json",
        {
            "num_samples": len(rows),
            "condition_mode": tag,
            "q_layers": int(rows[0]["tokens"].shape[1]),
            "token_len_mean": sum(int(x["tokens"].shape[0]) for x in rows) / max(len(rows), 1),
            "condition_len_mean": sum(int(x["condition_ids"].shape[0]) for x in rows) / max(len(rows), 1),
            "degeneracy_checks": {
                "target_is_train_gt_pose_frozen_rvq": True,
                "uses_bbest_for_training": False,
                "uses_test_pt": False,
                "token_is_gloss": False,
                "token_is_single_pose_frame": False,
            },
        },
    )
    return rows, dur


class TokenDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    condition = pad_sequence([x.get("condition_ids", x["gloss_ids"]) for x in batch], batch_first=True, padding_value=PAD)
    max_t = max(x["tokens"].shape[0] for x in batch)
    q = batch[0]["tokens"].shape[1]
    tokens = torch.full((len(batch), max_t, q), -100, dtype=torch.long)
    frame_lens = torch.tensor([x["frame_len"] for x in batch], dtype=torch.long)
    token_lens = torch.tensor([x["tokens"].shape[0] for x in batch], dtype=torch.long)
    for i, row in enumerate(batch):
        tokens[i, : row["tokens"].shape[0]] = row["tokens"]
    return {"condition": condition, "gloss": condition, "tokens": tokens, "frame_lens": frame_lens, "token_lens": token_lens}


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1]]


class RVQTokenNAT(nn.Module):
    def __init__(self, vocab_size: int, n_codes: int, q_layers: int, dim: int, heads: int, enc_layers: int, dec_layers: int, dropout: float, max_tokens: int):
        super().__init__()
        self.n_codes = n_codes
        self.q_layers = q_layers
        self.gloss_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.query = nn.Parameter(torch.randn(max_tokens, dim) * 0.02)
        self.pos = PositionalEncoding(dim, max_tokens + 256)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.heads = nn.ModuleList([nn.Linear(dim, n_codes) for _ in range(q_layers)])

    def forward(self, gloss: torch.Tensor, out_len: int) -> list[torch.Tensor]:
        pad = gloss.eq(PAD)
        mem = self.encoder(self.pos(self.gloss_emb(gloss)), src_key_padding_mask=pad)
        q = self.query[:out_len].unsqueeze(0).expand(gloss.shape[0], -1, -1)
        dec = self.decoder(self.pos(q), mem, memory_key_padding_mask=pad)
        return [head(dec) for head in self.heads]


def build_token_class_weights(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[torch.Tensor] | None, dict[str, Any]]:
    power = float(getattr(args, "token_class_weight_power", 0.0) or 0.0)
    if power <= 0:
        return None, {"enabled": False}
    q_layers = int(rows[0]["tokens"].shape[1])
    limit = int(args.train_pred_layers) if int(args.train_pred_layers) > 0 else q_layers
    limit = min(limit, q_layers)
    smooth = float(getattr(args, "token_class_weight_smoothing", 1.0) or 1.0)
    max_weight = float(getattr(args, "token_class_weight_max", 0.0) or 0.0)
    weights = []
    layer_summary = []
    for layer in range(limit):
        counts = torch.zeros(args.n_codes, dtype=torch.float32)
        for row in rows:
            cur = row["tokens"][:, layer].long()
            cur = cur[(cur >= 0) & (cur < args.n_codes)]
            if cur.numel() > 0:
                counts.scatter_add_(0, cur.cpu(), torch.ones(cur.numel(), dtype=torch.float32))
        freq = (counts + smooth) / (counts.sum() + smooth * args.n_codes).clamp_min(1.0)
        weight = freq.pow(-power)
        active = counts > 0
        normalizer = weight[active].mean() if active.any() else weight.mean()
        weight = weight / normalizer.clamp_min(1e-6)
        if max_weight > 0:
            weight = weight.clamp(max=max_weight)
        weights.append(weight.to(args.device))
        layer_summary.append(
            {
                "layer": layer,
                "active_codes": int(active.sum().item()),
                "min_weight": float(weight[active].min().item()) if active.any() else float(weight.min().item()),
                "max_weight": float(weight[active].max().item()) if active.any() else float(weight.max().item()),
                "mean_weight": float(weight[active].mean().item()) if active.any() else float(weight.mean().item()),
            }
        )
    return weights, {"enabled": True, "power": power, "max_weight": max_weight, "smoothing": smooth, "layers": layer_summary}


def token_loss(
    logits: list[torch.Tensor],
    target: torch.Tensor,
    coarse_weight: float,
    train_pred_layers: int = 0,
    class_weights: list[torch.Tensor] | None = None,
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    losses = []
    acc = []
    limit = train_pred_layers if train_pred_layers > 0 else len(logits)
    for layer, cur in enumerate(logits[:limit]):
        weight = coarse_weight ** layer
        cls_weight = class_weights[layer] if class_weights is not None and layer < len(class_weights) else None
        loss = nn.functional.cross_entropy(
            cur.reshape(-1, cur.shape[-1]),
            target[:, :, layer].reshape(-1),
            ignore_index=-100,
            weight=cls_weight,
            label_smoothing=label_smoothing,
        )
        losses.append(weight * loss)
        mask = target[:, :, layer].ne(-100)
        if mask.any():
            pred = cur.argmax(dim=-1)
            acc.append(float(pred[mask].eq(target[:, :, layer][mask]).float().mean().detach().cpu()))
        else:
            acc.append(0.0)
    return torch.stack(losses).sum() / len(losses), {"layer_acc": acc, "layer_loss": [float(x.detach().cpu()) for x in losses], "trained_layers": limit}


def drop_condition_tokens(condition: torch.Tensor, prob: float) -> torch.Tensor:
    if prob <= 0:
        return condition
    aug = condition.clone()
    valid = aug.ne(PAD)
    mask = torch.rand(aug.shape, device=aug.device).lt(prob) & valid
    aug[mask] = UNK
    return aug


def consistency_loss(logits_a: list[torch.Tensor], logits_b: list[torch.Tensor], target: torch.Tensor, train_pred_layers: int = 0) -> torch.Tensor:
    limit = train_pred_layers if train_pred_layers > 0 else len(logits_a)
    losses = []
    for layer in range(limit):
        mask = target[:, :, layer].ne(-100)
        if not mask.any():
            continue
        logp_a = torch.log_softmax(logits_a[layer].float(), dim=-1)
        logp_b = torch.log_softmax(logits_b[layer].float(), dim=-1)
        prob_a = logp_a.exp().detach()
        prob_b = logp_b.exp().detach()
        kl_ab = nn.functional.kl_div(logp_a, prob_b, reduction="none").sum(dim=-1)
        kl_ba = nn.functional.kl_div(logp_b, prob_a, reduction="none").sum(dim=-1)
        losses.append((kl_ab[mask].mean() + kl_ba[mask].mean()) * 0.5)
    if not losses:
        return target.new_tensor(0.0, dtype=torch.float32)
    return torch.stack(losses).mean()


def val_metrics(model: RVQTokenNAT, loader: DataLoader, args: argparse.Namespace, class_weights: list[torch.Tensor] | None = None) -> dict[str, Any]:
    model.eval()
    total = 0.0
    count = 0
    layer_acc = None
    with torch.no_grad():
        for batch in loader:
            gloss = batch["condition"].to(args.device)
            tokens = batch["tokens"].to(args.device)
            logits = model(gloss, tokens.shape[1])
            loss, parts = token_loss(logits, tokens, args.coarse_weight, args.train_pred_layers, class_weights, args.label_smoothing)
            total += float(loss.item())
            count += 1
            cur_acc = parts["layer_acc"]
            layer_acc = cur_acc if layer_acc is None else [a + b for a, b in zip(layer_acc, cur_acc)]
    if layer_acc is not None:
        layer_acc = [x / max(count, 1) for x in layer_acc]
    return {"loss": total / max(count, 1), "layer_acc": layer_acc}


def build_transition_lm(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any] | None:
    if float(args.token_transition_lambda) <= 0:
        return None
    limit = int(args.decode_pred_layers or args.train_pred_layers or rows[0]["tokens"].shape[1])
    limit = min(limit, int(rows[0]["tokens"].shape[1]))
    smoothing = float(args.token_transition_smoothing)
    start = torch.full((limit, args.n_codes), smoothing, dtype=torch.float32)
    trans = torch.full((limit, args.n_codes, args.n_codes), smoothing, dtype=torch.float32)
    for row in rows:
        tokens = row["tokens"].long().clamp(0, args.n_codes - 1)
        for layer in range(limit):
            seq = tokens[:, layer]
            if seq.numel() == 0:
                continue
            start[layer, int(seq[0])] += 1.0
            if seq.numel() > 1:
                flat = seq[:-1] * args.n_codes + seq[1:]
                trans[layer].view(-1).scatter_add_(0, flat.cpu(), torch.ones_like(flat, dtype=torch.float32).cpu())
    log_start = (start / start.sum(dim=1, keepdim=True)).log().to(args.device)
    log_trans = (trans / trans.sum(dim=2, keepdim=True)).log().to(args.device)
    return {
        "log_start": log_start,
        "log_trans": log_trans,
        "layers": limit,
        "summary": {
            "enabled": True,
            "lambda": float(args.token_transition_lambda),
            "topk": int(args.token_transition_topk),
            "smoothing": smoothing,
            "layers": limit,
            "source_rows": len(rows),
        },
    }


def transition_viterbi(logits: torch.Tensor, layer: int, transition_lm: dict[str, Any], args: argparse.Namespace) -> torch.Tensor:
    emissions = torch.log_softmax(logits.float(), dim=-1)
    topk = min(int(args.token_transition_topk), emissions.shape[-1])
    cand_score, cand_id = emissions.topk(topk, dim=-1)
    lam = float(args.token_transition_lambda)
    log_start = transition_lm["log_start"][layer]
    log_trans = transition_lm["log_trans"][layer]
    dp = cand_score[0] + lam * log_start[cand_id[0]]
    backptr = []
    for pos in range(1, cand_id.shape[0]):
        prev_ids = cand_id[pos - 1]
        cur_ids = cand_id[pos]
        score = dp.unsqueeze(1) + lam * log_trans[prev_ids][:, cur_ids]
        best_score, best_idx = score.max(dim=0)
        dp = best_score + cand_score[pos]
        backptr.append(best_idx)
    choice = int(dp.argmax())
    path = [0] * cand_id.shape[0]
    path[-1] = int(cand_id[-1, choice])
    for pos in range(cand_id.shape[0] - 1, 0, -1):
        choice = int(backptr[pos - 1][choice])
        path[pos - 1] = int(cand_id[pos - 1, choice])
    return torch.tensor(path, dtype=torch.long)


def train_prior(args: argparse.Namespace, rows: list[dict[str, Any]], vocab: dict[str, int], out_dir: Path) -> Path:
    ckpt = out_dir / "checkpoints" / "rvq_prior_best.pt"
    if ckpt.exists() and not args.force:
        return ckpt
    split_seed = int(args.split_seed if args.split_seed is not None else args.seed)
    init_seed = int(args.init_seed if args.init_seed is not None else args.seed)
    rng = random.Random(split_seed)
    rows = list(rows)
    rng.shuffle(rows)
    val_n = min(max(1, int(len(rows) * args.val_ratio), 64), max(1, len(rows) - 1))
    train_rows, val_rows = rows[val_n:], rows[:val_n]
    split_summary = {
        "split_seed": split_seed,
        "init_seed": init_seed,
        "val_n": val_n,
        "train_n": len(train_rows),
        "val_sample_id_sha1": hashlib.sha1("\n".join(str(x.get("sample_id", "")) for x in val_rows).encode("utf-8")).hexdigest(),
    }
    original_train_n = len(train_rows)
    train_predicted_gloss, train_predicted_gloss_path = load_train_predicted_gloss(args)
    class_weights, class_weight_summary = build_token_class_weights(train_rows, args)
    train_rows = augment_train_rows(args, train_rows, vocab, train_predicted_gloss)
    train_loader = DataLoader(TokenDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(TokenDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    q_layers = int(rows[0]["tokens"].shape[1])
    max_tokens = max(int(x["tokens"].shape[0]) for x in rows) + 16
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    model = RVQTokenNAT(len(vocab), args.n_codes, q_layers, args.dim, args.heads, args.enc_layers, args.dec_layers, args.dropout, max_tokens).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    history = []
    deadline = time.time() + args.train_seconds
    step = 0
    while time.time() < deadline and step < args.max_steps:
        model.train()
        for batch in train_loader:
            step += 1
            gloss = batch["condition"].to(args.device)
            tokens = batch["tokens"].to(args.device)
            logits = model(gloss, tokens.shape[1])
            loss, parts = token_loss(logits, tokens, args.coarse_weight, args.train_pred_layers, class_weights, args.label_smoothing)
            cons_value = None
            if float(args.consistency_weight) > 0:
                aug_gloss = drop_condition_tokens(gloss, float(args.condition_dropout_prob))
                aug_logits = model(aug_gloss, tokens.shape[1])
                cons = consistency_loss(logits, aug_logits, tokens, args.train_pred_layers)
                cons_value = float(cons.detach().cpu())
                loss = loss + float(args.consistency_weight) * cons
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step == 1 or step % args.val_every == 0:
                val = val_metrics(model, val_loader, args, class_weights)
                row = {"step": step, "time": now_iso(), "train_loss": float(loss.detach().cpu()), "train_layer_acc": parts["layer_acc"], "grad_norm": float(torch.as_tensor(grad).cpu()), "val": val}
                if cons_value is not None:
                    row["consistency_loss"] = cons_value
                if step == 1:
                    row["train_condition_augmentation"] = {
                        "original_train_rows": original_train_n,
                        "effective_train_rows": len(train_rows),
                        "train_predicted_gloss_path": train_predicted_gloss_path,
                        "train_predicted_gloss_rows": len(train_predicted_gloss),
                        "train_predicted_gloss_copies": int(args.train_predicted_gloss_copies),
                        "train_predicted_gloss_rank": int(args.train_predicted_gloss_rank),
                        "gloss_noise_copies": int(args.gloss_noise_copies),
                        "drop_prob": float(args.gloss_noise_drop_prob),
                        "replace_prob": float(args.gloss_noise_replace_prob),
                        "swap_prob": float(args.gloss_noise_swap_prob),
                    }
                    row["training_objective"] = {
                        "label_smoothing": float(args.label_smoothing),
                        "token_class_weight": class_weight_summary,
                        "consistency_weight": float(args.consistency_weight),
                        "condition_dropout_prob": float(args.condition_dropout_prob),
                    }
                    row["split"] = split_summary
                history.append(row)
                write_json(out_dir / "logs" / "rvq_prior_train_history.json", history)
                if val["loss"] < best:
                    best = float(val["loss"])
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model": model.state_dict(), "vocab": vocab, "q_layers": q_layers, "max_tokens": max_tokens, "best_val": best, "step": step, "split": split_summary, "training_objective": {"label_smoothing": float(args.label_smoothing), "token_class_weight": class_weight_summary, "consistency_weight": float(args.consistency_weight), "condition_dropout_prob": float(args.condition_dropout_prob)}, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, ckpt)
            if time.time() >= deadline or step >= args.max_steps:
                break
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "vocab": vocab, "q_layers": q_layers, "max_tokens": max_tokens, "best_val": best, "step": step, "split": split_summary, "training_objective": {"label_smoothing": float(args.label_smoothing), "token_class_weight": class_weight_summary, "consistency_weight": float(args.consistency_weight), "condition_dropout_prob": float(args.condition_dropout_prob)}, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}, ckpt)
    return ckpt


def load_prior(args: argparse.Namespace, ckpt: Path) -> tuple[RVQTokenNAT, dict[str, int]]:
    data = load_torch(ckpt)
    vocab = data["vocab"]
    model = RVQTokenNAT(len(vocab), args.n_codes, int(data["q_layers"]), args.dim, args.heads, args.enc_layers, args.dec_layers, args.dropout, int(data["max_tokens"])).to(args.device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, vocab


def average_prior_logits(priors: list[RVQTokenNAT], cond_ids: torch.Tensor, token_len: int) -> list[torch.Tensor]:
    logits_per_model = [prior(cond_ids, token_len) for prior in priors]
    if len(logits_per_model) == 1:
        return logits_per_model[0]
    n_layers = len(logits_per_model[0])
    averaged = []
    for layer in range(n_layers):
        ref_shape = logits_per_model[0][layer].shape
        cur_logits = []
        for model_idx, logits in enumerate(logits_per_model):
            if len(logits) != n_layers:
                raise ValueError(f"ensemble prior {model_idx} has {len(logits)} layers, expected {n_layers}")
            if logits[layer].shape != ref_shape:
                raise ValueError(f"ensemble prior {model_idx} layer {layer} shape {tuple(logits[layer].shape)}, expected {tuple(ref_shape)}")
            cur_logits.append(logits[layer].float())
        averaged.append(torch.stack(cur_logits, dim=0).mean(dim=0))
    return averaged


def generate_split(args: argparse.Namespace, split: str, priors: list[RVQTokenNAT], vocab: dict[str, int], duration: dict[str, Any], out_dir: Path, transition_lm: dict[str, Any] | None = None) -> Path:
    if not priors:
        raise ValueError("at least one prior model is required")
    tok_model, norm, tok_data = load_model_from_ckpt(args.tokenizer_ckpt, args)
    codebooks = tok_data["posthoc_codebooks"].to(args.device)
    zero_codes = codebooks.pow(2).sum(dim=2).argmin(dim=1).cpu()
    rows = read_json(topk_path(args.project_root, split, args.config_id))
    items = list(rows.items())
    if split == "dev" and args.max_dev_samples > 0:
        if not args.skip_eval:
            raise ValueError("--max-dev-samples can only be used with --skip-eval")
        items = items[: args.max_dev_samples]
    pred = {}
    cand_rank = int(args.candidate_rank)
    with torch.no_grad():
        for sid, row in items:
            candidates = row.get("candidates", [])
            if not candidates:
                raise ValueError(f"{sid} has no predicted gloss candidates")
            if cand_rank >= len(candidates):
                raise ValueError(f"{sid} has {len(candidates)} candidates, cannot use rank {cand_rank}")
            gloss = candidates[cand_rank]["pred_gloss"]
            frame_len = predict_frame_len(gloss, duration, args.min_len, args.max_len)
            frame_len = int(max(args.min_len, min(args.max_len, round(frame_len * args.duration_scale))))
            token_len = len(make_starts(frame_len, args.window_size, args.stride))
            cond_ids = encode_condition_values(row.get("text", ""), gloss, vocab, args).unsqueeze(0).to(args.device)
            logits = average_prior_logits(priors, cond_ids, token_len)
            seqs = []
            for layer, cur in enumerate(logits):
                cur_logits = cur.squeeze(0)
                if transition_lm is not None and layer < int(transition_lm["layers"]):
                    seqs.append(transition_viterbi(cur_logits, layer, transition_lm, args).cpu())
                else:
                    seqs.append(cur_logits.argmax(dim=-1).cpu())
            tokens = torch.stack(seqs, dim=1)
            if 0 < args.decode_pred_layers < tokens.shape[1]:
                for layer in range(args.decode_pred_layers, tokens.shape[1]):
                    tokens[:, layer] = int(zero_codes[layer])
            pred[sid] = decode_tokens_to_pose(tok_model, tokens, norm, codebooks, frame_len, args)
    suffix = f"_{args.prediction_suffix}" if args.prediction_suffix else ""
    if cand_rank != 0:
        suffix = f"{suffix}_rank{cand_rank}"
    if args.decode_pred_layers > 0:
        suffix = f"{suffix}_k{args.decode_pred_layers}"
    if args.duration_scale != 1.0:
        suffix = f"{suffix}_dur{str(args.duration_scale).replace('.', 'p')}"
    if float(args.token_transition_lambda) > 0:
        suffix = f"{suffix}_tlm{str(args.token_transition_lambda).replace('.', 'p')}_top{args.token_transition_topk}"
    out = out_dir / "predictions" / split / f"rvq_prior_{args.config_id}{suffix}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    write_json(out.with_suffix(".validation.json"), validate_prediction_only(pred))
    return out


def validate_prediction_only(pred: dict[str, torch.Tensor]) -> dict[str, Any]:
    errors = []
    lengths = []
    for sid, pose in pred.items():
        if not isinstance(pose, torch.Tensor) or pose.dtype != torch.float32 or pose.ndim != 3 or tuple(pose.shape[1:]) != (178, 3) or pose.shape[0] <= 0 or not torch.isfinite(pose).all():
            errors.append({"sample_id": sid, "shape": tuple(pose.shape) if hasattr(pose, "shape") else None, "dtype": str(getattr(pose, "dtype", None))})
        else:
            lengths.append(int(pose.shape[0]))
    return {"ok": not errors, "num_errors": len(errors), "errors": errors[:50], "num_samples": len(pred), "length_mean": sum(lengths) / max(len(lengths), 1)}


def run_evaluator(args: argparse.Namespace, pred_path: Path, split: str, out_dir: Path) -> dict[str, Any]:
    if split == "test" and not args.include_test:
        raise RuntimeError("test evaluation 需要 --include-test")
    tag = pred_path.stem if split == "dev" else f"{split}_{pred_path.stem}"
    workspace = out_dir / "eval_workspace" / tag
    workspace.mkdir(parents=True, exist_ok=True)
    evaluator = args.project_root / "exps" / "task3" / "repos" / "SLRTP-Sign-Production-Evaluation" / "main.py"
    gt = args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / f"{split}.pt"
    bt = args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "backTranslation_PHIX_model"
    cmd = [str(args.eval_python), str(evaluator), str(pred_path.resolve()), str(gt.resolve()), str(bt.resolve()), "--tag", tag, "--fps", "25"] if args.eval_python else ["conda", "run", "-n", args.eval_env, "python", str(evaluator), str(pred_path.resolve()), str(gt.resolve()), str(bt.resolve()), "--tag", tag, "--fps", "25"]
    start = now_iso()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = now_iso()
    log = out_dir / "eval_results" / split / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8")
    src = workspace / "results" / f"{tag}.json"
    dst = out_dir / "eval_results" / split / f"{tag}.json"
    result = {"cmd": " ".join(cmd), "start": start, "end": end, "elapsed_sec": time.time() - t0, "returncode": proc.returncode, "log": str(log), "result_json": str(dst)}
    if proc.returncode == 0 and src.exists():
        shutil.copy2(src, dst)
        result.update({"status": "ok", "metrics": read_json(dst)})
    else:
        result.update({"status": "failed", "error_tail": proc.stdout[-4000:]})
        write_json(dst, result)
    return result


def render_report(out_dir: Path, summary: dict[str, Any]) -> None:
    split = summary["args"].get("eval_split", "dev")
    eval_result = summary.get("split_eval") or summary.get("dev_eval") or summary.get("test_eval") or {}
    metrics = eval_result.get("metrics", {})
    bleu = metrics.get("bleu", {}) if metrics else {}
    bleu4 = bleu.get("bleu4")
    if split == "test":
        verdict = "final test 已执行；不再根据 test 结果调参或二次提交。"
    elif isinstance(bleu4, (int, float)) and bleu4 > 10.877469287658991:
        verdict = "dev 已超过 current B-best 10.8775；需复验并冻结配置后才允许 final test。"
    else:
        verdict = "dev 未超过 current B-best 10.8775 前不跑 test。"
    lines = [
        "# RVQ prior 实验报告",
        "",
        "## 方法",
        "",
        "冻结 v9 post-hoc RVQ tokenizer，训练 condition-conditioned NAT 预测 RVQ token。prior target 只来自 train GT pose 经 tokenizer 编码，不使用 B-best/GUS/winner pose 训练。",
        "",
        "## Eval 结果",
        "",
        f"- split: {split}",
        f"- status: {eval_result.get('status', 'skipped')}",
        f"- condition_mode: {summary['args'].get('condition_mode')}",
        f"- BLEU4: {bleu.get('bleu4', 'NA')}",
        f"- WER: {metrics.get('wer', 'NA') if metrics else 'NA'}",
        "",
        "## 判定",
        "",
        verdict,
    ]
    suffix = f"_{summary['args'].get('prediction_suffix')}" if summary["args"].get("prediction_suffix") else ""
    if int(summary["args"].get("candidate_rank", 0) or 0) != 0:
        suffix = f"{suffix}_rank{summary['args']['candidate_rank']}"
    if int(summary["args"].get("decode_pred_layers", 0) or 0) > 0:
        suffix = f"{suffix}_k{summary['args']['decode_pred_layers']}"
    if float(summary["args"].get("duration_scale", 1.0) or 1.0) != 1.0:
        suffix = f"{suffix}_dur{str(summary['args']['duration_scale']).replace('.', 'p')}"
    if float(summary["args"].get("token_transition_lambda", 0.0) or 0.0) > 0:
        suffix = f"{suffix}_tlm{str(summary['args']['token_transition_lambda']).replace('.', 'p')}_top{summary['args']['token_transition_topk']}"
    report_name = "rvq_prior_report" if split == "dev" else f"rvq_prior_report_{split}"
    path = out_dir / "reports" / f"{report_name}{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary_name = "rvq_prior_summary" if split == "dev" else f"rvq_prior_summary_{split}"
    write_json(out_dir / "reports" / f"{summary_name}{suffix}.json", summary)


def run(args: argparse.Namespace) -> None:
    if args.eval_split == "test" and not args.skip_eval and not args.include_test:
        raise RuntimeError("test evaluation 需要显式传入 --include-test")
    if args.candidate_rank < 0:
        raise ValueError("--candidate-rank must be >= 0")
    if args.train_predicted_gloss_copies < 0:
        raise ValueError("--train-predicted-gloss-copies must be >= 0")
    if args.train_predicted_gloss_rank < 0:
        raise ValueError("--train-predicted-gloss-rank must be >= 0")
    if args.gloss_noise_copies < 0:
        raise ValueError("--gloss-noise-copies must be >= 0")
    if args.token_transition_lambda < 0:
        raise ValueError("--token-transition-lambda must be >= 0")
    if args.token_transition_topk <= 0:
        raise ValueError("--token-transition-topk must be > 0")
    if args.token_transition_smoothing <= 0:
        raise ValueError("--token-transition-smoothing must be > 0")
    if not 0.0 <= float(args.label_smoothing) < 1.0:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.token_class_weight_power < 0:
        raise ValueError("--token-class-weight-power must be >= 0")
    if args.token_class_weight_max < 0:
        raise ValueError("--token-class-weight-max must be >= 0")
    if args.token_class_weight_smoothing <= 0:
        raise ValueError("--token-class-weight-smoothing must be > 0")
    if args.consistency_weight < 0:
        raise ValueError("--consistency-weight must be >= 0")
    for name in ("gloss_noise_drop_prob", "gloss_noise_replace_prob", "gloss_noise_swap_prob"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if not 0.0 <= float(args.condition_dropout_prob) <= 1.0:
        raise ValueError("--condition-dropout-prob must be in [0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_torch(args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / "train.pt")
    vocab = build_vocab(train, args)
    rows, duration = build_token_rows(args, train, vocab, args.out_dir)
    write_json(args.out_dir / "tokenizer" / "gloss_vocab.json", vocab)
    transition_lm = build_transition_lm(rows, args)
    ckpt = args.prior_ckpt if args.prior_ckpt else train_prior(args, rows, vocab, args.out_dir)
    prior, vocab = load_prior(args, ckpt)
    priors = [prior]
    ensemble_ckpts = []
    for ensemble_ckpt in args.ensemble_prior_ckpts:
        extra_prior, extra_vocab = load_prior(args, ensemble_ckpt)
        if extra_vocab != vocab:
            raise ValueError(f"ensemble prior vocab mismatch: {ensemble_ckpt}")
        priors.append(extra_prior)
        ensemble_ckpts.append(str(ensemble_ckpt))
    pred = generate_split(args, args.eval_split, priors, vocab, duration, args.out_dir, transition_lm)
    split_eval = None if args.skip_eval else run_evaluator(args, pred, args.eval_split, args.out_dir)
    summary = {
        "created_at": now_iso(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": str(ckpt),
        "ensemble_checkpoints": [str(ckpt), *ensemble_ckpts],
        "prediction": str(pred),
        "split_prediction": str(pred),
        "split_eval": split_eval,
        "dev_eval": split_eval if args.eval_split == "dev" else None,
        "test_eval": split_eval if args.eval_split == "test" else None,
        "data_boundary": {
            "target": "train_gt_pose_encoded_by_frozen_v9_temporal_rvq",
            **condition_boundary(args),
            "eval_split": args.eval_split,
            "eval_condition": eval_condition_description(args, args.eval_split),
            "uses_bbest_or_gus_pose_for_training": False,
            "candidate_rank": int(args.candidate_rank),
            "uses_topk_predicted_gloss_for_condition_selection": True,
            "token_transition_lm": transition_lm["summary"] if transition_lm is not None else {"enabled": False},
            "prior_ensemble": {
                "enabled": len(priors) > 1,
                "num_models": len(priors),
                "aggregation": "logits_mean" if len(priors) > 1 else "single_model",
                "checkpoints": [str(ckpt), *ensemble_ckpts],
            },
            "train_condition_augmentation": {
                "train_predicted_gloss_json": str(args.train_predicted_gloss_json) if args.train_predicted_gloss_json else None,
                "train_predicted_gloss_copies": int(args.train_predicted_gloss_copies),
                "train_predicted_gloss_rank": int(args.train_predicted_gloss_rank),
                "gloss_noise_copies": int(args.gloss_noise_copies),
                "drop_prob": float(args.gloss_noise_drop_prob),
                "replace_prob": float(args.gloss_noise_replace_prob),
                "swap_prob": float(args.gloss_noise_swap_prob),
            },
            "uses_test_pt_for_training": False,
            "uses_test_pt_for_final_evaluation": bool(args.eval_split == "test" and not args.skip_eval),
        },
        "training_objective": {
            "label_smoothing": float(args.label_smoothing),
            "token_class_weight_power": float(args.token_class_weight_power),
            "token_class_weight_max": float(args.token_class_weight_max),
            "token_class_weight_smoothing": float(args.token_class_weight_smoothing),
            "consistency_weight": float(args.consistency_weight),
            "condition_dropout_prob": float(args.condition_dropout_prob),
        },
        "split_seed": int(args.split_seed if args.split_seed is not None else args.seed),
        "init_seed": int(args.init_seed if args.init_seed is not None else args.seed),
        "baseline": {
            "rvq_tokenizer_reconstruct_bbest_bleu4": 10.582364543182312,
            "rvq_prior_k3_duration_bleu4": 10.537193085461618,
            "current_b_best_dev_bleu4": 10.877469287658991,
        },
    }
    render_report(args.out_dir, summary)


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--out-dir", type=Path, default=default_project / "g2p_ddm_token_exps" / "outputs" / "rvq_prior_nat")
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--condition-mode", choices=["gloss", "text", "text_gloss"], default="gloss")
    parser.add_argument("--max-condition-len", type=int, default=128)
    parser.add_argument("--config-id", default="beam5_lp1p0_max100")
    parser.add_argument("--candidate-rank", type=int, default=0, help="which predicted gloss candidate rank to use from the top-k json")
    parser.add_argument("--eval-env", default="t2s-oracle")
    parser.add_argument("--eval-python", type=Path, default=None)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--split-seed", type=int, default=None, help="train/validation split seed; defaults to --seed for backward compatibility")
    parser.add_argument("--init-seed", type=int, default=None, help="model initialization seed; defaults to --seed for backward compatibility")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prior-ckpt", type=Path, default=None)
    parser.add_argument("--ensemble-prior-ckpts", type=Path, nargs="*", default=[], help="additional same-architecture prior checkpoints for logits-mean ensemble at inference")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--decode-pred-layers", type=int, default=0, help="if >0, predict only the first K RVQ layers and fill later layers with nearest-zero residual codes")
    parser.add_argument("--train-pred-layers", type=int, default=0, help="if >0, train CE only on the first K RVQ layers")
    parser.add_argument("--prediction-suffix", default="")
    parser.add_argument("--train-predicted-gloss-json", type=Path, default=None, help="optional train top-k/predicted gloss json for condition augmentation")
    parser.add_argument("--train-predicted-gloss-copies", type=int, default=0, help="number of train predicted-gloss condition copies to add to the train split only")
    parser.add_argument("--train-predicted-gloss-rank", type=int, default=0)
    parser.add_argument("--gloss-noise-copies", type=int, default=0, help="number of noisy-gloss condition copies to add to the train split only")
    parser.add_argument("--gloss-noise-drop-prob", type=float, default=0.0)
    parser.add_argument("--gloss-noise-replace-prob", type=float, default=0.0)
    parser.add_argument("--gloss-noise-swap-prob", type=float, default=0.0)
    parser.add_argument("--token-transition-lambda", type=float, default=0.0, help="if >0, add train-token transition LM score during top-k Viterbi decoding")
    parser.add_argument("--token-transition-topk", type=int, default=32, help="per-position emission candidates kept for transition Viterbi")
    parser.add_argument("--token-transition-smoothing", type=float, default=0.1)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--feature-mode", choices=["raw_pose", "coarse_velocity"], default="raw_pose")
    parser.add_argument("--pose-dct-components", type=int, default=2)
    parser.add_argument("--velocity-dct-components", type=int, default=2)
    parser.add_argument("--n-codes", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-quantizers", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coarse-weight", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--token-class-weight-power", type=float, default=0.0, help="if >0, apply inverse-frequency token class weights with this power")
    parser.add_argument("--token-class-weight-max", type=float, default=5.0)
    parser.add_argument("--token-class-weight-smoothing", type=float, default=1.0)
    parser.add_argument("--consistency-weight", type=float, default=0.0, help="if >0, add symmetric KL consistency loss between clean and condition-dropout forwards")
    parser.add_argument("--condition-dropout-prob", type=float, default=0.0, help="replace non-pad condition tokens with <unk> for the consistency branch")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--train-seconds", type=int, default=7200)
    parser.add_argument("--max-steps", type=int, default=50000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=280)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
