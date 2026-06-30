#!/usr/bin/env python3
"""aligned prior explicit duration/alignment K3 RVQ token prior.

本脚本保留已经有效的 frozen v9 whole-pose K3 RVQ motion token target，
但把 train-only duration 展开的 gloss timeline 显式作为 decoder query。

边界：
- token target 只来自 train GT pose 经 frozen tokenizer 编码；
- duration/alignment 只来自 train split 统计；
- dev 只用 predicted gloss 生成；
- 不使用 B-best/GUS/winner pose 训练、rerank 或 fallback；
- 不读取 test.pt；
- token 不是 gloss token，也不是单帧 pose token。
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from rvq_monotonic_prior_experiment import (
    allocate_duration_counts,
    collapse_check,
    token_usage_from_sequences,
)
from rvq_prior_experiment import (
    PAD,
    UNK,
    PositionalEncoding,
    build_token_rows,
    build_vocab,
    condition_boundary,
    decode_tokens_to_pose,
    encode_condition_values,
    load_model_from_ckpt,
    load_torch,
    make_starts,
    predict_frame_len,
    read_json,
    run_evaluator,
    split_gloss,
    token_loss,
    topk_path,
    validate_prediction_only,
    write_json,
)


K3_BASELINE_DEV_BLEU4 = 11.193247069579575
MONOTONIC_PRIOR_DEV_BLEU4 = 11.149383861057952
CROSS_PRIOR_INTERPOLATION_DEV_BLEU4 = 11.43461328313725
MIXTURE_PRIOR_DEV_BLEU4 = 11.483153718257475


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def weight_tag(weight: float) -> str:
    return str(weight).replace(".", "p")


def load_cache_vocab_duration(cache_dir: Path) -> tuple[dict[str, int], dict[str, Any]]:
    vocab_path = cache_dir / "gloss_vocab.json"
    duration_path = cache_dir / "duration_stats.json"
    if not vocab_path.exists() or not duration_path.exists():
        raise FileNotFoundError(f"missing token cache vocab/duration under {cache_dir}")
    return read_json(vocab_path), read_json(duration_path)


def normalize_duration_stats(duration: dict[str, Any], source: Path | str) -> dict[str, Any]:
    if "global" not in duration or "per_gloss" not in duration:
        raise ValueError(f"duration stats from {source} must contain global and per_gloss")
    if not isinstance(duration["per_gloss"], dict):
        raise ValueError(f"duration stats from {source} has non-dict per_gloss")
    global_value = float(duration["global"])
    if global_value <= 0:
        raise ValueError(f"duration stats from {source} has non-positive global value")
    per_gloss = {str(key): float(value) for key, value in duration["per_gloss"].items()}
    if any(value <= 0 for value in per_gloss.values()):
        raise ValueError(f"duration stats from {source} has non-positive per_gloss value")
    normalized = dict(duration)
    normalized["global"] = global_value
    normalized["per_gloss"] = per_gloss
    return normalized


def maybe_override_duration_stats(args: argparse.Namespace, duration: dict[str, Any]) -> tuple[dict[str, Any], str]:
    default_source = str(args.token_cache_dir / "duration_stats.json")
    if args.duration_stats_json is None:
        return normalize_duration_stats(duration, default_source), default_source
    if not args.duration_stats_json.exists():
        raise FileNotFoundError(f"duration stats override not found: {args.duration_stats_json}")
    loaded = normalize_duration_stats(read_json(args.duration_stats_json), args.duration_stats_json)
    return loaded, str(args.duration_stats_json)


def load_or_build_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any], str, str]:
    vocab, duration = load_cache_vocab_duration(args.token_cache_dir)
    cached = args.token_cache_dir / "train_rvq_tokens.pt"
    if cached.exists() and not args.rebuild_token_cache:
        rows = load_torch(cached)
        source = str(cached)
    else:
        train = load_torch(args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / "train.pt")
        local_vocab = build_vocab(train, args)
        if local_vocab != vocab:
            raise ValueError("local train vocab differs from token cache vocab")
        rows, duration = build_token_rows(args, train, vocab, args.out_dir)
        source = str(args.out_dir / "tokenizer" / "train_rvq_tokens.pt")
    if args.max_train_samples > 0:
        rows = rows[: args.max_train_samples]
    duration, duration_source = maybe_override_duration_stats(args, duration)
    return rows, vocab, duration, source, duration_source


def load_train_predicted_gloss(args: argparse.Namespace) -> tuple[dict[str, str], str | None]:
    copies = int(args.train_predicted_gloss_copies)
    if copies <= 0 and not args.val_predicted_gloss:
        return {}, None
    path = args.train_predicted_gloss_json or topk_path(args.project_root, "train", args.config_id)
    rows = read_json(path)
    rank = int(args.train_predicted_gloss_rank)
    pred: dict[str, str] = {}
    for sid, row in rows.items():
        candidates = row.get("candidates", [])
        if candidates:
            if rank >= len(candidates):
                raise ValueError(f"{sid} has {len(candidates)} train predicted gloss candidates, cannot use rank {rank}")
            pred[sid] = candidates[rank]["pred_gloss"]
        elif row.get("pred_gloss"):
            pred[sid] = row["pred_gloss"]
    return pred, str(path)


def load_generation_items(args: argparse.Namespace, split: str) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    source = str(getattr(args, "eval_condition_source", "predicted_gloss") or "predicted_gloss")
    if source == "predicted_gloss":
        path = topk_path(args.project_root, split, args.config_id)
        rows = read_json(path)
        return list(rows.items()), str(path)
    if source == "gt_gloss":
        split_path = args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / f"{split}.pt"
        rows = load_torch(split_path)
        items: list[tuple[str, dict[str, Any]]] = []
        for sid, row in rows.items():
            items.append((sid, {"text": row.get("text", ""), "gt_gloss": row.get("gloss", ""), "gloss": row.get("gloss", "")}))
        return items, str(split_path)
    raise ValueError(f"unknown --eval-condition-source: {source}")


def with_predicted_gloss_condition(row: dict[str, Any], pred_gloss: str, vocab: dict[str, int], args: argparse.Namespace, suffix: str) -> dict[str, Any]:
    item = dict(row)
    item["sample_id"] = f"{row.get('sample_id')}::{suffix}"
    item["gloss"] = pred_gloss
    item["condition_ids"] = encode_condition_values(row.get("text", ""), pred_gloss, vocab, args)
    item["gloss_ids"] = item["condition_ids"]
    item["condition_gloss"] = pred_gloss
    item["condition_augmented"] = "train_predicted_gloss"
    return item


def apply_train_predicted_gloss(
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    vocab: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_predicted_gloss, train_predicted_gloss_path = load_train_predicted_gloss(args)
    copies = int(args.train_predicted_gloss_copies)
    summary = {
        "train_predicted_gloss_json": train_predicted_gloss_path,
        "train_predicted_gloss_rows": len(train_predicted_gloss),
        "train_predicted_gloss_copies": copies,
        "train_predicted_gloss_rank": int(args.train_predicted_gloss_rank),
        "val_predicted_gloss": bool(args.val_predicted_gloss),
        "base_train_n": len(train_rows),
        "base_val_n": len(val_rows),
    }
    if copies <= 0 and not args.val_predicted_gloss:
        summary["train_n_after_aug"] = len(train_rows)
        summary["val_n_after_aug"] = len(val_rows)
        return train_rows, val_rows, summary
    if not train_predicted_gloss:
        raise ValueError("train predicted gloss augmentation requested but no rows were loaded")

    augmented_train = list(train_rows)
    train_missing = 0
    for row in train_rows:
        sid = row.get("sample_id")
        pred_gloss = train_predicted_gloss.get(sid)
        if pred_gloss is None:
            train_missing += 1
            continue
        for copy_idx in range(copies):
            augmented_train.append(with_predicted_gloss_condition(row, pred_gloss, vocab, args, f"pred_r{args.train_predicted_gloss_rank}_c{copy_idx}"))

    if args.val_predicted_gloss:
        augmented_val = []
        val_missing = 0
        for row in val_rows:
            sid = row.get("sample_id")
            pred_gloss = train_predicted_gloss.get(sid)
            if pred_gloss is None:
                val_missing += 1
                augmented_val.append(row)
                continue
            augmented_val.append(with_predicted_gloss_condition(row, pred_gloss, vocab, args, f"val_pred_r{args.train_predicted_gloss_rank}"))
    else:
        augmented_val = val_rows
        val_missing = 0

    summary.update(
        {
            "train_missing_predicted_gloss": train_missing,
            "val_missing_predicted_gloss": val_missing,
            "train_n_after_aug": len(augmented_train),
            "val_n_after_aug": len(augmented_val),
            "leakage_guard": "split_before_duplicate",
        }
    )
    return augmented_train, augmented_val, summary


def apply_alignment_mode(timeline: torch.Tensor, align: torch.Tensor, args: argparse.Namespace | None) -> tuple[torch.Tensor, torch.Tensor]:
    mode = str(getattr(args, "alignment_mode", "full") or "full")
    if mode == "full":
        return timeline, align
    if mode == "none":
        return torch.full_like(timeline, UNK), torch.zeros_like(align)
    raise ValueError(f"unknown --alignment-mode: {mode}")


def expand_alignment_inputs(gloss: str, token_len: int, vocab: dict[str, int], duration: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    if token_len <= 0:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 4, dtype=torch.float32)
    gloss_tokens = split_gloss(gloss)
    if not gloss_tokens:
        ids = torch.full((token_len,), UNK, dtype=torch.long)
        pos = torch.linspace(0.0, 1.0, token_len)
        feats = torch.stack([pos, torch.zeros_like(pos), torch.zeros_like(pos), torch.ones_like(pos)], dim=1)
        return ids, feats

    token_ids = [vocab.get(tok, UNK) for tok in gloss_tokens]
    counts = allocate_duration_counts(gloss_tokens, token_len, duration)
    timeline: list[int] = []
    features: list[list[float]] = []
    cursor = 0
    denom_seg = max(len(gloss_tokens) - 1, 1)
    denom_abs = max(token_len - 1, 1)
    for seg_idx, (tok_id, count) in enumerate(zip(token_ids, counts)):
        if count <= 0:
            continue
        denom_local = max(count - 1, 1)
        for local_idx in range(count):
            abs_idx = cursor + local_idx
            timeline.append(tok_id)
            features.append(
                [
                    abs_idx / denom_abs,
                    seg_idx / denom_seg,
                    local_idx / denom_local,
                    count / max(token_len, 1),
                ]
            )
        cursor += count
    if len(timeline) < token_len:
        last = token_ids[-1]
        while len(timeline) < token_len:
            abs_idx = len(timeline)
            timeline.append(last)
            features.append([abs_idx / denom_abs, 1.0, 1.0, 1.0 / max(token_len, 1)])
    return torch.tensor(timeline[:token_len], dtype=torch.long), torch.tensor(features[:token_len], dtype=torch.float32)


class AlignedDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def make_collate(vocab: dict[str, int], duration: dict[str, Any], args: argparse.Namespace | None = None):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        condition = pad_sequence([x.get("condition_ids", x["gloss_ids"]) for x in batch], batch_first=True, padding_value=PAD)
        max_t = max(int(x["tokens"].shape[0]) for x in batch)
        q = int(batch[0]["tokens"].shape[1])
        timeline = torch.full((len(batch), max_t), PAD, dtype=torch.long)
        align = torch.zeros(len(batch), max_t, 4, dtype=torch.float32)
        tokens = torch.full((len(batch), max_t, q), -100, dtype=torch.long)
        token_lens = torch.tensor([int(x["tokens"].shape[0]) for x in batch], dtype=torch.long)
        for idx, row in enumerate(batch):
            t_len = int(row["tokens"].shape[0])
            cur_timeline, cur_align = expand_alignment_inputs(row.get("gloss", ""), t_len, vocab, duration)
            cur_timeline, cur_align = apply_alignment_mode(cur_timeline, cur_align, args)
            timeline[idx, :t_len] = cur_timeline
            align[idx, :t_len] = cur_align
            tokens[idx, :t_len] = row["tokens"]
        return {"condition": condition, "timeline": timeline, "align": align, "tokens": tokens, "token_lens": token_lens}

    return collate


def split_train_val_rows(args: argparse.Namespace, rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split_seed = int(args.split_seed if args.split_seed is not None else args.seed)
    init_seed = int(args.init_seed if args.init_seed is not None else args.seed)
    rows = list(rows)
    if len(rows) < 2:
        raise ValueError("aligned prior train_prior requires at least two rows")
    val_n = max(1, int(len(rows) * args.val_ratio))
    if len(rows) >= 256:
        val_n = max(val_n, 64)
    val_n = min(val_n, len(rows) - 1)
    bin_summary = None
    if args.split_mode == "length_hash":
        bins: list[list[dict[str, Any]]] = [[] for _ in range(max(1, int(args.split_length_bins)))]
        by_len = sorted(rows, key=lambda x: int(x["tokens"].shape[0]))
        for rank, row in enumerate(by_len):
            bin_idx = min(int(rank * len(bins) / max(len(by_len), 1)), len(bins) - 1)
            bins[bin_idx].append(row)
        val_rows = []
        train_rows = []
        bin_summary = []
        remaining_val = val_n
        for bin_idx, cur_bin in enumerate(bins):
            cur_bin = sorted(cur_bin, key=lambda x: hashlib.sha1(str(x.get("sample_id", "")).encode("utf-8")).hexdigest())
            if not cur_bin:
                bin_summary.append({"bin": bin_idx, "rows": 0, "val_n": 0, "len_min": None, "len_max": None})
                continue
            target = int(round(len(cur_bin) * args.val_ratio))
            target = max(1, target) if len(cur_bin) > 1 else 0
            target = min(target, len(cur_bin) - 1)
            target = min(target, remaining_val)
            val_rows.extend(cur_bin[:target])
            train_rows.extend(cur_bin[target:])
            remaining_val -= target
            lengths = [int(x["tokens"].shape[0]) for x in cur_bin]
            bin_summary.append({"bin": bin_idx, "rows": len(cur_bin), "val_n": target, "len_min": min(lengths), "len_max": max(lengths)})
        if len(val_rows) < val_n:
            candidates = sorted(train_rows, key=lambda x: hashlib.sha1(str(x.get("sample_id", "")).encode("utf-8")).hexdigest())
            needed = min(val_n - len(val_rows), len(candidates) - 1)
            move_ids = {id(x) for x in candidates[:needed]}
            val_rows.extend(candidates[:needed])
            train_rows = [x for x in train_rows if id(x) not in move_ids]
    elif args.split_mode == "hash":
        rows.sort(key=lambda x: hashlib.sha1(str(x.get("sample_id", "")).encode("utf-8")).hexdigest())
        val_rows, train_rows = rows[:val_n], rows[val_n:]
    else:
        rng = random.Random(split_seed)
        rng.shuffle(rows)
        val_rows, train_rows = rows[:val_n], rows[val_n:]
    summary = {
        "split_mode": args.split_mode,
        "split_seed": split_seed,
        "init_seed": init_seed,
        "val_n": val_n,
        "train_n": len(train_rows),
        "val_sample_id_sha1": hashlib.sha1("\n".join(str(x.get("sample_id", "")) for x in val_rows).encode("utf-8")).hexdigest(),
    }
    if bin_summary is not None:
        summary["length_bins"] = bin_summary
    return train_rows, val_rows, summary


class AlignedK3Prior(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codes: int,
        pred_layers: int,
        dim: int,
        heads: int,
        enc_layers: int,
        dec_layers: int,
        dropout: float,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.n_codes = n_codes
        self.pred_layers = pred_layers
        self.condition_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.timeline_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.align_proj = nn.Linear(4, dim)
        self.pos = PositionalEncoding(dim, max_tokens + 256)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(dim, heads, dim * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.heads = nn.ModuleList([nn.Linear(dim, n_codes) for _ in range(pred_layers)])

    def forward(self, condition: torch.Tensor, timeline: torch.Tensor, align: torch.Tensor) -> list[torch.Tensor]:
        condition_pad = condition.eq(PAD)
        timeline_pad = timeline.eq(PAD)
        memory = self.encoder(self.pos(self.condition_emb(condition)), src_key_padding_mask=condition_pad)
        query = self.timeline_emb(timeline) + self.align_proj(align.float())
        decoded = self.decoder(
            self.pos(query),
            memory,
            tgt_key_padding_mask=timeline_pad,
            memory_key_padding_mask=condition_pad,
        )
        return [head(decoded) for head in self.heads]


def val_metrics(
    model: AlignedK3Prior,
    loader: DataLoader,
    args: argparse.Namespace,
    decoded_aux: dict[str, Any] | None = None,
    include_aux: bool = False,
) -> dict[str, Any]:
    model.eval()
    total = 0.0
    token_total = 0.0
    count = 0
    layer_acc = None
    with torch.no_grad():
        for batch in loader:
            condition = batch["condition"].to(args.device)
            timeline = batch["timeline"].to(args.device)
            align = batch["align"].to(args.device)
            tokens = batch["tokens"].to(args.device)
            logits = model(condition, timeline, align)
            token_ce, parts = token_loss(logits, tokens, args.coarse_weight, args.pred_layers)
            loss = token_ce
            if include_aux:
                if float(args.decoded_aux_weight) > 0.0:
                    aux_loss, _aux_parts = decoded_window_aux_loss(logits, tokens, decoded_aux, args)
                    loss = loss + float(args.decoded_aux_weight) * aux_loss
                if float(args.decoded_overlap_aux_weight) > 0.0:
                    overlap_loss, _overlap_parts = decoded_overlap_aux_loss(logits, tokens, decoded_aux, args)
                    loss = loss + float(args.decoded_overlap_aux_weight) * overlap_loss
                if float(args.latent_delta_aux_weight) > 0.0 or float(args.latent_energy_aux_weight) > 0.0:
                    latent_loss, _latent_parts = latent_delta_aux_loss(logits, tokens, decoded_aux, args)
                    loss = loss + latent_loss
                if float(getattr(args, "stable_token_kl_weight", 0.0)) > 0.0:
                    stable_loss, _stable_parts = stable_token_kl_loss(logits, tokens, args)
                    loss = loss + float(args.stable_token_kl_weight) * stable_loss
            total += float(loss.item())
            token_total += float(token_ce.item())
            count += 1
            cur_acc = parts["layer_acc"]
            layer_acc = cur_acc if layer_acc is None else [a + b for a, b in zip(layer_acc, cur_acc)]
    if layer_acc is not None:
        layer_acc = [x / max(count, 1) for x in layer_acc]
    return {
        "loss": total / max(count, 1),
        "token_loss": token_total / max(count, 1),
        "layer_acc": layer_acc,
        "selection_loss": "composite_aux" if include_aux else "token_ce",
    }


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def cpu_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def update_ema_state(ema_state: dict[str, torch.Tensor] | None, model: nn.Module, decay: float) -> dict[str, torch.Tensor]:
    cur = model.state_dict()
    if ema_state is None:
        return clone_state_dict(cur)
    with torch.no_grad():
        for name, value in cur.items():
            if name not in ema_state:
                ema_state[name] = value.detach().clone()
            elif torch.is_floating_point(value):
                ema_state[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)
            else:
                ema_state[name].copy_(value.detach())
    return ema_state


def evaluate_ema_state(
    model: AlignedK3Prior,
    ema_state: dict[str, torch.Tensor],
    loader: DataLoader,
    args: argparse.Namespace,
    decoded_aux: dict[str, Any] | None = None,
    include_aux: bool = False,
) -> dict[str, Any]:
    raw_state = clone_state_dict(model.state_dict())
    model.load_state_dict(ema_state)
    ema_val = val_metrics(model, loader, args, decoded_aux, include_aux)
    model.load_state_dict(raw_state)
    return ema_val


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)


def load_matching_state_dict(module: nn.Module, source_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Load only parameters whose names and shapes match the target module."""
    target_state = module.state_dict()
    loaded: list[str] = []
    skipped: list[dict[str, Any]] = []
    for name, value in source_state.items():
        if name not in target_state:
            skipped.append({"name": name, "reason": "missing_in_target"})
            continue
        if tuple(value.shape) != tuple(target_state[name].shape):
            skipped.append(
                {
                    "name": name,
                    "reason": "shape_mismatch",
                    "source_shape": list(value.shape),
                    "target_shape": list(target_state[name].shape),
                }
            )
            continue
        target_state[name].copy_(value)
        loaded.append(name)
    module.load_state_dict(target_state, strict=True)
    return {"loaded": loaded, "skipped": skipped, "num_loaded": len(loaded), "num_skipped": len(skipped)}


def build_decoded_aux(args: argparse.Namespace) -> dict[str, Any] | None:
    if (
        float(args.decoded_aux_weight) <= 0.0
        and float(getattr(args, "decoded_overlap_aux_weight", 0.0)) <= 0.0
        and float(getattr(args, "latent_delta_aux_weight", 0.0)) <= 0.0
        and float(getattr(args, "latent_energy_aux_weight", 0.0)) <= 0.0
    ):
        return None
    tok_model, _norm, tok_data = load_model_from_ckpt(args.tokenizer_ckpt, args)
    freeze_module(tok_model)
    codebooks = tok_data["posthoc_codebooks"].to(args.device)
    zero_codes = codebooks.pow(2).sum(dim=2).argmin(dim=1).to(args.device)
    return {"model": tok_model, "codebooks": codebooks, "zero_codes": zero_codes}


def decoded_windows_for_indices(
    logits: list[torch.Tensor],
    tokens: torch.Tensor,
    batch_idx: torch.Tensor,
    token_idx: torch.Tensor,
    aux: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    codebooks = aux["codebooks"]
    zero_codes = aux["zero_codes"]
    pred_q = logits[0].new_zeros((batch_idx.shape[0], codebooks.shape[-1]))
    target_q = logits[0].new_zeros((batch_idx.shape[0], codebooks.shape[-1]))
    temperature = max(float(args.decoded_aux_temperature), 1e-6)
    for layer in range(codebooks.shape[0]):
        if layer < int(args.pred_layers):
            cur_logits = logits[layer][batch_idx, token_idx].float()
            probs = torch.softmax(cur_logits / temperature, dim=-1)
            pred_q = pred_q + probs @ codebooks[layer].to(pred_q.dtype)
            target_ids = tokens[batch_idx, token_idx, layer].clamp(0, codebooks.shape[1] - 1).to(codebooks.device)
            target_q = target_q + codebooks[layer, target_ids].to(target_q.dtype)
        else:
            zero = codebooks[layer, zero_codes[layer]].to(pred_q.dtype)
            pred_q = pred_q + zero.unsqueeze(0)
            target_q = target_q + zero.unsqueeze(0)
    decoder = aux["model"].decoder
    pred_win = decoder(pred_q).reshape(-1, args.window_size, 178, 3)
    with torch.no_grad():
        target_win = decoder(target_q).reshape(-1, args.window_size, 178, 3)
    return pred_win, target_win


def decoded_window_aux_loss(
    logits: list[torch.Tensor],
    tokens: torch.Tensor,
    aux: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if aux is None:
        return tokens.new_tensor(0.0, dtype=torch.float32), {}
    valid = tokens[:, :, 0].ne(-100).nonzero(as_tuple=False)
    if valid.numel() == 0:
        return logits[0].new_tensor(0.0), {"decoded_aux_windows": 0.0}
    max_windows = int(args.decoded_aux_max_windows_per_batch)
    if max_windows > 0 and valid.shape[0] > max_windows:
        perm = torch.randperm(valid.shape[0], device=valid.device)[:max_windows]
        valid = valid[perm]
    batch_idx = valid[:, 0]
    token_idx = valid[:, 1]
    pred_win, target_win = decoded_windows_for_indices(logits, tokens, batch_idx, token_idx, aux, args)
    if args.decoded_aux_loss == "l1":
        recon = F.l1_loss(pred_win, target_win)
    else:
        recon = F.smooth_l1_loss(pred_win, target_win)
    if pred_win.shape[1] > 1:
        pred_vel = pred_win[:, 1:] - pred_win[:, :-1]
        target_vel = target_win[:, 1:] - target_win[:, :-1]
        if args.decoded_aux_loss == "l1":
            vel = F.l1_loss(pred_vel, target_vel)
        else:
            vel = F.smooth_l1_loss(pred_vel, target_vel)
    else:
        vel = recon.new_tensor(0.0)
    loss = recon + float(args.decoded_aux_velocity_weight) * vel
    return loss, {
        "decoded_aux_windows": float(valid.shape[0]),
        "decoded_aux_recon": float(recon.detach().cpu()),
        "decoded_aux_velocity": float(vel.detach().cpu()),
        "decoded_aux_total": float(loss.detach().cpu()),
    }


def decoded_overlap_aux_loss(
    logits: list[torch.Tensor],
    tokens: torch.Tensor,
    aux: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if aux is None or float(args.decoded_overlap_aux_weight) <= 0.0:
        return logits[0].new_tensor(0.0), {}
    window_size = int(args.window_size)
    stride = int(args.stride)
    overlap = window_size - stride
    if overlap <= 0:
        return logits[0].new_tensor(0.0), {"decoded_overlap_pairs": 0.0}
    valid_pairs = tokens[:, :-1, 0].ne(-100) & tokens[:, 1:, 0].ne(-100)
    pair_idx = valid_pairs.nonzero(as_tuple=False)
    if pair_idx.numel() == 0:
        return logits[0].new_tensor(0.0), {"decoded_overlap_pairs": 0.0}
    max_pairs = int(args.decoded_overlap_aux_max_pairs_per_batch)
    if max_pairs > 0 and pair_idx.shape[0] > max_pairs:
        perm = torch.randperm(pair_idx.shape[0], device=pair_idx.device)[:max_pairs]
        pair_idx = pair_idx[perm]
    batch_idx = pair_idx[:, 0]
    left_idx = pair_idx[:, 1]
    right_idx = left_idx + 1
    left_pred, left_target = decoded_windows_for_indices(logits, tokens, batch_idx, left_idx, aux, args)
    right_pred, right_target = decoded_windows_for_indices(logits, tokens, batch_idx, right_idx, aux, args)
    pred_delta = left_pred[:, stride:] - right_pred[:, :overlap]
    target_delta = left_target[:, stride:] - right_target[:, :overlap]
    if args.decoded_aux_loss == "l1":
        consistency = nn.functional.l1_loss(pred_delta, target_delta)
    else:
        consistency = nn.functional.smooth_l1_loss(pred_delta, target_delta)
    return consistency, {
        "decoded_overlap_pairs": float(pair_idx.shape[0]),
        "decoded_overlap_consistency": float(consistency.detach().cpu()),
        "decoded_overlap_total": float(consistency.detach().cpu()),
    }


def latent_vectors_for_indices(
    logits: list[torch.Tensor],
    tokens: torch.Tensor,
    batch_idx: torch.Tensor,
    token_idx: torch.Tensor,
    aux: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    codebooks = aux["codebooks"]
    pred_q = logits[0].new_zeros((batch_idx.shape[0], codebooks.shape[-1]))
    target_q = logits[0].new_zeros((batch_idx.shape[0], codebooks.shape[-1]))
    temperature = max(float(args.decoded_aux_temperature), 1e-6)
    for layer in range(min(int(args.pred_layers), codebooks.shape[0])):
        cur_logits = logits[layer][batch_idx, token_idx].float()
        probs = torch.softmax(cur_logits / temperature, dim=-1)
        pred_q = pred_q + probs @ codebooks[layer].to(pred_q.dtype)
        target_ids = tokens[batch_idx, token_idx, layer].clamp(0, codebooks.shape[1] - 1).to(codebooks.device)
        target_q = target_q + codebooks[layer, target_ids].to(target_q.dtype)
    return pred_q, target_q


def latent_delta_aux_loss(
    logits: list[torch.Tensor],
    tokens: torch.Tensor,
    aux: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    if aux is None:
        return logits[0].new_tensor(0.0), {}
    valid_pairs = tokens[:, :-1, 0].ne(-100) & tokens[:, 1:, 0].ne(-100)
    pair_idx = valid_pairs.nonzero(as_tuple=False)
    if pair_idx.numel() == 0:
        return logits[0].new_tensor(0.0), {"latent_delta_pairs": 0.0}
    max_pairs = int(args.latent_delta_max_pairs_per_batch)
    if max_pairs > 0 and pair_idx.shape[0] > max_pairs:
        perm = torch.randperm(pair_idx.shape[0], device=pair_idx.device)[:max_pairs]
        pair_idx = pair_idx[perm]
    batch_idx = pair_idx[:, 0]
    left_idx = pair_idx[:, 1]
    right_idx = left_idx + 1
    left_pred, left_target = latent_vectors_for_indices(logits, tokens, batch_idx, left_idx, aux, args)
    right_pred, right_target = latent_vectors_for_indices(logits, tokens, batch_idx, right_idx, aux, args)
    pred_delta = right_pred - left_pred
    target_delta = right_target - left_target
    if args.decoded_aux_loss == "l1":
        delta_loss = F.l1_loss(pred_delta, target_delta)
    else:
        delta_loss = F.smooth_l1_loss(pred_delta, target_delta)
    pred_energy = pred_delta.pow(2).mean(dim=1).sqrt()
    target_energy = target_delta.pow(2).mean(dim=1).sqrt()
    energy_match = F.smooth_l1_loss(pred_energy, target_energy)
    floor = target_energy.detach() * float(args.latent_energy_floor_ratio)
    energy_floor = F.relu(floor - pred_energy).pow(2).mean()
    energy_loss = energy_match + energy_floor
    total = float(args.latent_delta_aux_weight) * delta_loss + float(args.latent_energy_aux_weight) * energy_loss
    return total, {
        "latent_delta_pairs": float(pair_idx.shape[0]),
        "latent_delta": float(delta_loss.detach().cpu()),
        "latent_energy": float(energy_loss.detach().cpu()),
        "latent_pred_energy": float(pred_energy.mean().detach().cpu()),
        "latent_target_energy": float(target_energy.mean().detach().cpu()),
        "latent_aux_total": float(total.detach().cpu()),
    }


def teacher_kl_loss(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    tokens: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    temperature = max(float(args.teacher_kl_temperature), 1e-6)
    losses = []
    parts = []
    shared_layers = min(int(args.pred_layers), len(student_logits), len(teacher_logits))
    for layer in range(shared_layers):
        valid = tokens[:, :, layer].ne(-100)
        if valid.sum() == 0:
            continue
        student = student_logits[layer][valid].float() / temperature
        teacher = teacher_logits[layer][valid].float() / temperature
        cur = F.kl_div(
            F.log_softmax(student, dim=-1),
            F.softmax(teacher, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature)
        losses.append(cur)
        parts.append(float(cur.detach().cpu()))
    if not losses:
        return student_logits[0].new_tensor(0.0), {"teacher_kl_layers": [], "teacher_kl_shared_layers": float(shared_layers)}
    loss = torch.stack(losses).mean()
    return loss, {"teacher_kl": float(loss.detach().cpu()), "teacher_kl_layers": parts, "teacher_kl_shared_layers": float(shared_layers)}


def stable_token_kl_loss(
    logits: list[torch.Tensor],
    tokens: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep adjacent prediction distributions stable only where GT tokens are stable."""
    layer = int(args.stable_token_kl_layer)
    if layer < 0 or layer >= int(args.pred_layers):
        raise ValueError("--stable-token-kl-layer must be within predicted layers")
    valid_pairs = tokens[:, :-1, layer].ne(-100) & tokens[:, 1:, layer].ne(-100)
    stable_pairs = valid_pairs & tokens[:, :-1, layer].eq(tokens[:, 1:, layer])
    pair_idx = stable_pairs.nonzero(as_tuple=False)
    if pair_idx.numel() == 0:
        return logits[0].new_tensor(0.0), {"stable_token_kl_pairs": 0.0, "stable_token_kl_layer": float(layer)}
    max_pairs = int(args.stable_token_kl_max_pairs_per_batch)
    if max_pairs > 0 and pair_idx.shape[0] > max_pairs:
        perm = torch.randperm(pair_idx.shape[0], device=pair_idx.device)[:max_pairs]
        pair_idx = pair_idx[perm]
    batch_idx = pair_idx[:, 0]
    left_idx = pair_idx[:, 1]
    right_idx = left_idx + 1
    temperature = max(float(args.stable_token_kl_temperature), 1e-6)
    left = logits[layer][batch_idx, left_idx].float() / temperature
    right = logits[layer][batch_idx, right_idx].float() / temperature
    left_to_right = F.kl_div(
        F.log_softmax(left, dim=-1),
        F.softmax(right.detach(), dim=-1),
        reduction="batchmean",
    )
    right_to_left = F.kl_div(
        F.log_softmax(right, dim=-1),
        F.softmax(left.detach(), dim=-1),
        reduction="batchmean",
    )
    loss = 0.5 * (left_to_right + right_to_left) * (temperature * temperature)
    return loss, {
        "stable_token_kl": float(loss.detach().cpu()),
        "stable_token_kl_pairs": float(pair_idx.shape[0]),
        "stable_token_kl_layer": float(layer),
    }


def train_prior(args: argparse.Namespace, rows: list[dict[str, Any]], vocab: dict[str, int], duration: dict[str, Any]) -> Path:
    ckpt = args.prior_ckpt if args.prior_ckpt else args.out_dir / "checkpoints" / "rvq_aligned_prior_best.pt"
    if args.prior_ckpt:
        if not ckpt.exists():
            raise FileNotFoundError(f"frozen prior checkpoint not found: {ckpt}")
        return ckpt
    if ckpt.exists() and not args.force:
        return ckpt
    init_seed = int(args.init_seed if args.init_seed is not None else args.seed)
    train_rows, val_rows, split_summary = split_train_val_rows(args, rows)
    train_rows, val_rows, condition_aug_summary = apply_train_predicted_gloss(args, train_rows, val_rows, vocab)
    split_summary["train_condition_augmentation"] = condition_aug_summary
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    collate = make_collate(vocab, duration, args)
    train_loader = DataLoader(AlignedDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(AlignedDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    max_tokens = max(int(x["tokens"].shape[0]) for x in rows) + 16
    max_tokens = max(max_tokens, int(args.max_tokens))
    model = AlignedK3Prior(len(vocab), args.n_codes, args.pred_layers, args.dim, args.heads, args.enc_layers, args.dec_layers, args.dropout, max_tokens).to(args.device)
    init_prior_meta = None
    if args.init_prior_ckpt is not None:
        if not args.init_prior_ckpt.exists():
            raise FileNotFoundError(f"init prior checkpoint not found: {args.init_prior_ckpt}")
        init_data = load_torch(args.init_prior_ckpt)
        if init_data.get("vocab") != vocab:
            raise ValueError("init prior vocab differs from current token cache vocab")
        init_pred_layers = int(init_data.get("pred_layers", args.pred_layers))
        init_load_summary = {"mode": "strict"}
        if init_pred_layers != int(args.pred_layers):
            if not bool(getattr(args, "allow_init_pred_layer_mismatch", False)):
                raise ValueError("init prior pred_layers differs from current --pred-layers")
            init_load_summary = {"mode": "matching_only", **load_matching_state_dict(model, init_data["model"])}
        else:
            model.load_state_dict(init_data["model"], strict=True)
        init_prior_meta = {
            "path": str(args.init_prior_ckpt),
            "step": int(init_data.get("step", -1)),
            "best_val": init_data.get("best_val"),
            "decoded_aux": init_data.get("decoded_aux"),
            "ema": init_data.get("ema"),
            "source_pred_layers": init_pred_layers,
            "target_pred_layers": int(args.pred_layers),
            "load_summary": init_load_summary,
        }
    decoded_aux = build_decoded_aux(args)
    teacher_prior = None
    teacher_prior_meta = None
    freeze_train_meta = None
    if args.teacher_prior_ckpt is not None:
        if not args.teacher_prior_ckpt.exists():
            raise FileNotFoundError(f"teacher prior checkpoint not found: {args.teacher_prior_ckpt}")
        teacher_prior, teacher_vocab, teacher_data = load_prior(args, args.teacher_prior_ckpt)
        if teacher_vocab != vocab:
            raise ValueError("teacher prior vocab differs from current token cache vocab")
        teacher_pred_layers = int(teacher_data.get("pred_layers", args.pred_layers))
        if teacher_pred_layers != int(args.pred_layers) and not bool(getattr(args, "allow_teacher_pred_layer_mismatch", False)):
            raise ValueError("teacher prior pred_layers differs from current --pred-layers")
        freeze_module(teacher_prior)
        teacher_prior_meta = {
            "path": str(args.teacher_prior_ckpt),
            "step": int(teacher_data.get("step", -1)),
            "best_val": teacher_data.get("best_val"),
            "decoded_aux": teacher_data.get("decoded_aux"),
            "ema": teacher_data.get("ema"),
            "source_pred_layers": teacher_pred_layers,
            "target_pred_layers": int(args.pred_layers),
            "shared_kl_layers": min(teacher_pred_layers, int(args.pred_layers)),
            "weight": float(args.teacher_kl_weight),
            "temperature": float(args.teacher_kl_temperature),
        }
    if bool(getattr(args, "freeze_non_new_heads", False)):
        if init_prior_meta is None:
            raise ValueError("--freeze-non-new-heads requires --init-prior-ckpt")
        start_layer = int(getattr(args, "new_head_start_layer", -1))
        if start_layer < 0:
            start_layer = int(init_prior_meta["source_pred_layers"])
        if start_layer < 0 or start_layer >= len(model.heads):
            raise ValueError("--new-head-start-layer must point to at least one predicted head")
        for param in model.parameters():
            param.requires_grad_(False)
        trainable_heads = []
        for layer in range(start_layer, len(model.heads)):
            for param in model.heads[layer].parameters():
                param.requires_grad_(True)
            trainable_heads.append(layer)
        freeze_train_meta = {
            "enabled": True,
            "start_layer": start_layer,
            "trainable_heads": trainable_heads,
            "source_pred_layers": int(init_prior_meta["source_pred_layers"]),
            "target_pred_layers": int(args.pred_layers),
        }
    else:
        freeze_train_meta = {"enabled": False}
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("no trainable parameters remain after freezing")
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    history = []
    deadline = time.time() + args.train_seconds
    step = 0
    ema_state = None
    ema_enabled = float(args.ema_decay) > 0.0
    while time.time() < deadline and step < args.max_steps:
        model.train()
        for batch in train_loader:
            step += 1
            condition = batch["condition"].to(args.device)
            timeline = batch["timeline"].to(args.device)
            align = batch["align"].to(args.device)
            tokens = batch["tokens"].to(args.device)
            logits = model(condition, timeline, align)
            token_ce, parts = token_loss(logits, tokens, args.coarse_weight, args.pred_layers)
            if float(args.decoded_aux_weight) > 0.0:
                aux_loss, aux_parts = decoded_window_aux_loss(logits, tokens, decoded_aux, args)
            else:
                aux_loss, aux_parts = logits[0].new_tensor(0.0), {}
            overlap_loss, overlap_parts = decoded_overlap_aux_loss(logits, tokens, decoded_aux, args)
            if float(args.latent_delta_aux_weight) > 0.0 or float(args.latent_energy_aux_weight) > 0.0:
                latent_loss, latent_parts = latent_delta_aux_loss(logits, tokens, decoded_aux, args)
            else:
                latent_loss, latent_parts = logits[0].new_tensor(0.0), {}
            if teacher_prior is not None and float(args.teacher_kl_weight) > 0.0:
                with torch.no_grad():
                    teacher_logits = teacher_prior(condition, timeline, align)
                teacher_loss, teacher_parts = teacher_kl_loss(logits, teacher_logits, tokens, args)
            else:
                teacher_loss, teacher_parts = logits[0].new_tensor(0.0), {}
            if float(args.stable_token_kl_weight) > 0.0:
                stable_loss, stable_parts = stable_token_kl_loss(logits, tokens, args)
            else:
                stable_loss, stable_parts = logits[0].new_tensor(0.0), {}
            loss = (
                token_ce
                + float(args.decoded_aux_weight) * aux_loss
                + float(args.decoded_overlap_aux_weight) * overlap_loss
                + latent_loss
                + float(args.teacher_kl_weight) * teacher_loss
                + float(args.stable_token_kl_weight) * stable_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite aligned prior loss at step {step}: {loss.item()}")
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if ema_enabled and step >= int(args.ema_warmup_steps):
                ema_state = update_ema_state(ema_state, model, float(args.ema_decay))
            if step == 1 or step % args.val_every == 0:
                raw_val = val_metrics(model, val_loader, args, decoded_aux, bool(args.val_include_aux))
                ema_val = evaluate_ema_state(model, ema_state, val_loader, args, decoded_aux, bool(args.val_include_aux)) if ema_state is not None else None
                val = ema_val if ema_val is not None else raw_val
                row = {
                    "step": step,
                    "time": now_iso(),
                    "train_loss": float(loss.detach().cpu()),
                    "train_token_ce": float(token_ce.detach().cpu()),
                    "train_decoded_aux": aux_parts,
                    "train_decoded_overlap_aux": overlap_parts,
                    "train_latent_aux": latent_parts,
                    "train_teacher_kl": teacher_parts,
                    "train_stable_token_kl": stable_parts,
                    "train_layer_acc": parts["layer_acc"],
                    "grad_norm": float(torch.as_tensor(grad).cpu()),
                    "val": val,
                    "raw_val": raw_val,
                    "ema_val": ema_val,
                    "ema": {
                        "enabled": ema_enabled,
                        "decay": float(args.ema_decay),
                        "warmup_steps": int(args.ema_warmup_steps),
                        "selection": "ema_val" if ema_val is not None else "raw_val",
                    },
                }
                if step == 1:
                    row["split"] = split_summary
                    row["method"] = "condition_encoder_plus_duration_alignment_query"
                    row["init_prior"] = init_prior_meta
                    row["teacher_prior"] = teacher_prior_meta
                    row["freeze_train"] = freeze_train_meta
                history.append(row)
                write_json(args.out_dir / "logs" / "rvq_aligned_prior_history.json", history)
                if float(val["loss"]) < best:
                    best = float(val["loss"])
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": cpu_state_dict(ema_state) if ema_val is not None and ema_state is not None else cpu_state_dict(model.state_dict()),
                            "vocab": vocab,
                            "pred_layers": int(args.pred_layers),
                            "max_tokens": int(max_tokens),
                            "best_val": best,
                            "step": step,
                            "split": split_summary,
                            "ema": {
                                "enabled": ema_enabled,
                                "decay": float(args.ema_decay),
                                "warmup_steps": int(args.ema_warmup_steps),
                                "selected": bool(ema_val is not None),
                            },
                            "decoded_aux": {
                                "enabled": decoded_aux is not None,
                                "weight": float(args.decoded_aux_weight),
                                "temperature": float(args.decoded_aux_temperature),
                                "velocity_weight": float(args.decoded_aux_velocity_weight),
                                "max_windows_per_batch": int(args.decoded_aux_max_windows_per_batch),
                                "loss": str(args.decoded_aux_loss),
                                "overlap_weight": float(args.decoded_overlap_aux_weight),
                                "overlap_max_pairs_per_batch": int(args.decoded_overlap_aux_max_pairs_per_batch),
                                "latent_delta_weight": float(args.latent_delta_aux_weight),
                                "latent_energy_weight": float(args.latent_energy_aux_weight),
                                "latent_delta_max_pairs_per_batch": int(args.latent_delta_max_pairs_per_batch),
                                "latent_energy_floor_ratio": float(args.latent_energy_floor_ratio),
                                "val_include_aux": bool(args.val_include_aux),
                            },
                            "stable_token_kl": {
                                "weight": float(args.stable_token_kl_weight),
                                "temperature": float(args.stable_token_kl_temperature),
                                "layer": int(args.stable_token_kl_layer),
                                "max_pairs_per_batch": int(args.stable_token_kl_max_pairs_per_batch),
                            },
                            "init_prior": init_prior_meta,
                            "teacher_prior": teacher_prior_meta,
                            "freeze_train": freeze_train_meta,
                            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        },
                        ckpt,
                    )
            if time.time() >= deadline or step >= args.max_steps:
                break
    if not ckpt.exists():
        raw_val = val_metrics(model, val_loader, args, decoded_aux, bool(args.val_include_aux))
        ema_val = evaluate_ema_state(model, ema_state, val_loader, args, decoded_aux, bool(args.val_include_aux)) if ema_state is not None else None
        val = ema_val if ema_val is not None else raw_val
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": cpu_state_dict(ema_state) if ema_val is not None and ema_state is not None else cpu_state_dict(model.state_dict()),
                "vocab": vocab,
                "pred_layers": int(args.pred_layers),
                "max_tokens": int(max_tokens),
                "best_val": float(val["loss"]),
                "step": step,
                "split": split_summary,
                "ema": {
                    "enabled": ema_enabled,
                    "decay": float(args.ema_decay),
                    "warmup_steps": int(args.ema_warmup_steps),
                    "selected": bool(ema_val is not None),
                },
                "decoded_aux": {
                    "enabled": decoded_aux is not None,
                    "weight": float(args.decoded_aux_weight),
                    "temperature": float(args.decoded_aux_temperature),
                        "velocity_weight": float(args.decoded_aux_velocity_weight),
                        "max_windows_per_batch": int(args.decoded_aux_max_windows_per_batch),
                        "loss": str(args.decoded_aux_loss),
                        "overlap_weight": float(args.decoded_overlap_aux_weight),
                        "overlap_max_pairs_per_batch": int(args.decoded_overlap_aux_max_pairs_per_batch),
                        "latent_delta_weight": float(args.latent_delta_aux_weight),
                        "latent_energy_weight": float(args.latent_energy_aux_weight),
                        "latent_delta_max_pairs_per_batch": int(args.latent_delta_max_pairs_per_batch),
                        "latent_energy_floor_ratio": float(args.latent_energy_floor_ratio),
                        "val_include_aux": bool(args.val_include_aux),
                    },
                "stable_token_kl": {
                    "weight": float(args.stable_token_kl_weight),
                    "temperature": float(args.stable_token_kl_temperature),
                    "layer": int(args.stable_token_kl_layer),
                    "max_pairs_per_batch": int(args.stable_token_kl_max_pairs_per_batch),
                },
                "init_prior": init_prior_meta,
                "teacher_prior": teacher_prior_meta,
                "freeze_train": freeze_train_meta,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            ckpt,
        )
    return ckpt


def load_prior(args: argparse.Namespace, ckpt: Path) -> tuple[AlignedK3Prior, dict[str, int], dict[str, Any]]:
    data = load_torch(ckpt)
    vocab = data["vocab"]
    model = AlignedK3Prior(
        len(vocab),
        args.n_codes,
        int(data.get("pred_layers", args.pred_layers)),
        args.dim,
        args.heads,
        args.enc_layers,
        args.dec_layers,
        args.dropout,
        int(data.get("max_tokens", args.max_tokens)),
    ).to(args.device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, vocab, data


def generate_split(
    args: argparse.Namespace,
    split: str,
    model: AlignedK3Prior,
    vocab: dict[str, int],
    duration: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    tok_model, norm, tok_data = load_model_from_ckpt(args.tokenizer_ckpt, args)
    codebooks = tok_data["posthoc_codebooks"].to(args.device)
    zero_codes = codebooks.pow(2).sum(dim=2).argmin(dim=1).cpu()
    items, generation_source = load_generation_items(args, split)
    condition_source = str(getattr(args, "eval_condition_source", "predicted_gloss") or "predicted_gloss")
    if split == "dev" and args.max_dev_samples > 0:
        if not args.skip_eval:
            raise ValueError("--max-dev-samples can only be used with --skip-eval")
        items = items[: args.max_dev_samples]
    pred = {}
    generated_tokens = []
    cand_rank = int(args.candidate_rank)
    with torch.no_grad():
        for sid, row in items:
            if condition_source == "gt_gloss":
                gloss = row.get("gt_gloss") or row.get("gloss", "")
            else:
                candidates = row.get("candidates", [])
                if not candidates:
                    raise ValueError(f"{sid} has no predicted gloss candidates")
                if cand_rank >= len(candidates):
                    raise ValueError(f"{sid} has {len(candidates)} candidates, cannot use rank {cand_rank}")
                gloss = candidates[cand_rank]["pred_gloss"]
            frame_len = predict_frame_len(gloss, duration, args.min_len, args.max_len)
            frame_len = int(max(args.min_len, min(args.max_len, round(frame_len * args.duration_scale))))
            token_len = len(make_starts(frame_len, args.window_size, args.stride))
            condition = encode_condition_values(row.get("text", ""), gloss, vocab, args).unsqueeze(0).to(args.device)
            timeline, align = expand_alignment_inputs(gloss, token_len, vocab, duration)
            timeline, align = apply_alignment_mode(timeline, align, args)
            logits = model(condition, timeline.unsqueeze(0).to(args.device), align.unsqueeze(0).to(args.device))
            tokens = torch.empty((token_len, codebooks.shape[0]), dtype=torch.long)
            for layer in range(codebooks.shape[0]):
                tokens[:, layer] = int(zero_codes[layer])
            for layer, cur in enumerate(logits):
                tokens[:, layer] = cur.squeeze(0).argmax(dim=-1).cpu()
            generated_tokens.append(tokens[:, : args.pred_layers].clone())
            pred[sid] = decode_tokens_to_pose(tok_model, tokens, norm, codebooks, frame_len, args)
    suffix = f"_{args.prediction_suffix}" if args.prediction_suffix else ""
    if condition_source == "gt_gloss":
        suffix = f"{suffix}_gtgloss"
    elif cand_rank != 0:
        suffix = f"{suffix}_rank{cand_rank}"
    suffix = f"{suffix}_k{args.pred_layers}_dur{weight_tag(args.duration_scale)}"
    out = args.out_dir / "predictions" / split / f"rvq_aligned_prior_{args.config_id}{suffix}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    write_json(out.with_suffix(".validation.json"), validate_prediction_only(pred))
    write_json(out.with_suffix(".generation.json"), {"split": split, "condition_source": condition_source, "source": generation_source})
    usage = token_usage_from_sequences(generated_tokens, args.n_codes, args.pred_layers)
    write_json(out.with_suffix(".token_usage.json"), usage)
    return out, usage


def gate_decision(split_eval: dict[str, Any] | None, collapse: dict[str, Any], split: str) -> dict[str, Any]:
    if split_eval is None:
        return {"status": "smoke", "decision": "eval skipped; smoke only.", "bleu4": None}
    metrics = split_eval.get("metrics") or {}
    bleu4 = (metrics.get("bleu") or {}).get("bleu4")
    if not isinstance(bleu4, (int, float)):
        return {"status": "no_metric", "decision": "eval finished but BLEU4 missing; stop.", "bleu4": None}
    bleu4 = float(bleu4)
    if split == "test":
        if not collapse.get("ok", False):
            return {"status": "final_test_token_collapse", "decision": "final test completed; token usage collapsed; record only, no tuning.", "bleu4": bleu4}
        return {"status": "final_test", "decision": "final test completed for frozen candidate; record only, no tuning on test.", "bleu4": bleu4}
    if not collapse.get("ok", False):
        return {"status": "token_collapse", "decision": "generated token usage collapsed; stop, no test.", "bleu4": None}
    if bleu4 <= K3_BASELINE_DEV_BLEU4:
        return {"status": "failed", "decision": f"dev BLEU4 {bleu4} <= K3 baseline {K3_BASELINE_DEV_BLEU4}; stop.", "bleu4": bleu4}
    if bleu4 <= CROSS_PRIOR_INTERPOLATION_DEV_BLEU4:
        return {"status": "below_cross_prior_interpolation", "decision": f"dev BLEU4 {bleu4} <= cross-prior interpolation {CROSS_PRIOR_INTERPOLATION_DEV_BLEU4}; record only, no test.", "bleu4": bleu4}
    if bleu4 <= MIXTURE_PRIOR_DEV_BLEU4:
        return {"status": "below_current_best", "decision": f"dev BLEU4 {bleu4} <= mixture prior {MIXTURE_PRIOR_DEV_BLEU4}; record only, no test.", "bleu4": bleu4}
    if bleu4 < 11.55:
        return {"status": "small_positive", "decision": f"dev BLEU4 {bleu4} > mixture prior but < 11.55; one regeneration discussion only, no test.", "bleu4": bleu4}
    return {"status": "positive", "decision": f"dev BLEU4 {bleu4} >= 11.55; independent regeneration required before any test.", "bleu4": bleu4}


def render_report(args: argparse.Namespace, summary: dict[str, Any]) -> None:
    metrics = ((summary.get("split_eval") or {}).get("metrics") or {})
    bleu = metrics.get("bleu", {}) if metrics else {}
    result_title = "Test 结果" if args.eval_split == "test" else "Dev 结果"
    test_boundary = "- final-test 模式只在显式 `--include-test` 且加载冻结 `--prior-ckpt` 时读取 test.pt；不用于训练或调参。" if args.eval_split == "test" else "- 不读取 test.pt。"
    lines = [
        "# aligned prior aligned K3 prior 报告",
        "",
        "## 方法",
        "",
        "冻结 v9 temporal RVQ tokenizer，target 为 train GT pose encode 后的 K3 learned motion token。prior 使用 predicted gloss condition encoder，并用 train-only duration stats 展开的 gloss timeline + phase features 作为 decoder query。",
        f"decoded_aux: weight={float(args.decoded_aux_weight)}, source=train GT motion tokens decoded through frozen tokenizer, no BT training/selection.",
        "",
        f"## {result_title}",
        "",
        f"- status: {(summary.get('split_eval') or {}).get('status', 'skipped')}",
        f"- BLEU4: {bleu.get('bleu4', 'NA')}",
        f"- WER: {metrics.get('wer', 'NA') if metrics else 'NA'}",
        f"- DTW-MJE: {metrics.get('dtw_mje', 'NA') if metrics else 'NA'}",
        f"- gate: {summary['gate']['decision']}",
        f"- collapse_ok: {summary.get('prediction_token_usage', {}).get('collapse_check', {}).get('ok', 'NA')}",
        "",
        "## 边界",
        "",
        test_boundary,
        "- 不使用 official BT 参与训练、candidate selection 或 rerank；official evaluator 只用于最终 dev/test 指标。",
        "- 不使用 B-best/GUS/winner pose 训练、rerank 或 fallback。",
        "- token 是 frozen v9 whole-pose K3 RVQ learned motion token，不是 gloss token，也不是单帧 pose token。",
    ]
    path = args.out_dir / "reports" / "rvq_aligned_prior_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    if args.eval_split == "test":
        if not args.include_test:
            raise RuntimeError("读取 test.pt 必须显式传入 --include-test")
        if args.prior_ckpt is None:
            raise RuntimeError("final test 只能加载冻结 --prior-ckpt，禁止边训练边 test")
        if args.skip_eval:
            raise RuntimeError("final test 不允许 --skip-eval")
    if args.pred_layers <= 0:
        raise ValueError("--pred-layers must be > 0")
    if args.prior_ckpt is not None and args.init_prior_ckpt is not None:
        raise ValueError("--prior-ckpt and --init-prior-ckpt are mutually exclusive")
    if args.teacher_kl_weight < 0:
        raise ValueError("--teacher-kl-weight must be >= 0")
    if args.teacher_kl_weight > 0 and args.teacher_prior_ckpt is None:
        raise ValueError("--teacher-prior-ckpt is required when --teacher-kl-weight > 0")
    if args.teacher_kl_temperature <= 0:
        raise ValueError("--teacher-kl-temperature must be > 0")
    if args.stable_token_kl_weight < 0:
        raise ValueError("--stable-token-kl-weight must be >= 0")
    if args.stable_token_kl_temperature <= 0:
        raise ValueError("--stable-token-kl-temperature must be > 0")
    if args.stable_token_kl_layer < 0:
        raise ValueError("--stable-token-kl-layer must be >= 0")
    if args.stable_token_kl_max_pairs_per_batch < 0:
        raise ValueError("--stable-token-kl-max-pairs-per-batch must be >= 0")
    if args.candidate_rank < 0:
        raise ValueError("--candidate-rank must be >= 0")
    if args.train_predicted_gloss_copies < 0:
        raise ValueError("--train-predicted-gloss-copies must be >= 0")
    if args.train_predicted_gloss_rank < 0:
        raise ValueError("--train-predicted-gloss-rank must be >= 0")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.ema_warmup_steps < 0:
        raise ValueError("--ema-warmup-steps must be >= 0")
    if args.decoded_aux_weight < 0:
        raise ValueError("--decoded-aux-weight must be >= 0")
    if args.decoded_overlap_aux_weight < 0:
        raise ValueError("--decoded-overlap-aux-weight must be >= 0")
    if args.latent_delta_aux_weight < 0:
        raise ValueError("--latent-delta-aux-weight must be >= 0")
    if args.latent_energy_aux_weight < 0:
        raise ValueError("--latent-energy-aux-weight must be >= 0")
    if args.latent_delta_max_pairs_per_batch < 0:
        raise ValueError("--latent-delta-max-pairs-per-batch must be >= 0")
    if args.latent_energy_floor_ratio < 0:
        raise ValueError("--latent-energy-floor-ratio must be >= 0")
    if args.decoded_aux_temperature <= 0:
        raise ValueError("--decoded-aux-temperature must be > 0")
    if args.decoded_aux_max_windows_per_batch < 0:
        raise ValueError("--decoded-aux-max-windows-per-batch must be >= 0")
    if args.decoded_overlap_aux_max_pairs_per_batch < 0:
        raise ValueError("--decoded-overlap-aux-max-pairs-per-batch must be >= 0")
    if args.decode_overlap_floor < 0 or args.decode_overlap_floor > 1:
        raise ValueError("--decode-overlap-floor must be in [0, 1]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rows, vocab, duration, row_source, duration_source = load_or_build_rows(args)
    if args.pred_layers > int(rows[0]["tokens"].shape[1]):
        raise ValueError(f"--pred-layers {args.pred_layers} exceeds tokenizer layers {rows[0]['tokens'].shape[1]}")
    train_usage = token_usage_from_sequences([row["tokens"][:, : args.pred_layers] for row in rows], args.n_codes, args.pred_layers)
    ckpt = train_prior(args, rows, vocab, duration)
    model, vocab, ckpt_data = load_prior(args, ckpt)
    if args.train_only_no_generate:
        summary = {
            "created_at": now_iso(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "train_rows_source": row_source,
            "duration_stats_source": duration_source,
            "checkpoint": str(ckpt),
            "checkpoint_meta": {
                "step": int(ckpt_data.get("step", -1)),
                "best_val": ckpt_data.get("best_val"),
                "split": ckpt_data.get("split"),
                "ema": ckpt_data.get("ema"),
                "decoded_aux": ckpt_data.get("decoded_aux"),
            },
            "prediction": None,
            "split_eval": None,
            "gate": {"status": "train_only_no_generate", "decision": "trained and selected by train inner-val token loss only; no dev/test generation or evaluator."},
            "prediction_token_usage": {"train": train_usage, "generated": None, "collapse_check": None},
            "data_boundary": {
                "target": "frozen_v9_whole_pose_rvq_k3_learned_motion_token",
                **condition_boundary(args),
                "duration_alignment": "train_only_duration_stats_expanded_to_timeline_with_phase_features",
                "duration_stats_source": duration_source,
                "architecture": "condition_encoder_plus_alignment_query_decoder",
                "ema_stabilization": bool(float(args.ema_decay) > 0.0),
                "decoded_auxiliary": {
                    "enabled": bool(float(args.decoded_aux_weight) > 0.0),
                    "source": "train_gt_motion_tokens_decoded_through_frozen_tokenizer",
                    "uses_official_bt_for_training_or_selection": False,
                    "weight": float(args.decoded_aux_weight),
                    "temperature": float(args.decoded_aux_temperature),
                    "velocity_weight": float(args.decoded_aux_velocity_weight),
                    "max_windows_per_batch": int(args.decoded_aux_max_windows_per_batch),
                    "overlap_enabled": bool(float(args.decoded_overlap_aux_weight) > 0.0),
                    "overlap_weight": float(args.decoded_overlap_aux_weight),
                    "overlap_max_pairs_per_batch": int(args.decoded_overlap_aux_max_pairs_per_batch),
                    "latent_delta_enabled": bool(float(args.latent_delta_aux_weight) > 0.0),
                    "latent_delta_weight": float(args.latent_delta_aux_weight),
                    "latent_energy_weight": float(args.latent_energy_aux_weight),
                    "latent_delta_max_pairs_per_batch": int(args.latent_delta_max_pairs_per_batch),
                    "latent_energy_floor_ratio": float(args.latent_energy_floor_ratio),
                    "val_include_aux": bool(args.val_include_aux),
                },
                "teacher_kl": {
                    "enabled": bool(float(args.teacher_kl_weight) > 0.0),
                    "source": "frozen clean legacy coarse prior-style prior logits; no BT/dev/test/winner signal",
                    "weight": float(args.teacher_kl_weight),
                    "temperature": float(args.teacher_kl_temperature),
                },
                "uses_bbest_or_gus_pose_for_training": False,
                "uses_bbest_or_gus_pose_for_rerank_or_fallback": False,
                "uses_official_bt_for_training_or_selection": False,
                "uses_test_pt_for_training": False,
                "uses_test_pt_for_final_evaluation": False,
                "uses_train_predicted_gloss_for_condition_augmentation": bool(args.train_predicted_gloss_copies > 0 or args.val_predicted_gloss),
                "token_is_gloss": False,
                "token_is_single_pose_frame": False,
            },
        }
        write_json(args.out_dir / "reports" / "rvq_aligned_prior_summary.json", summary)
        return
    pred, pred_usage = generate_split(args, args.eval_split, model, vocab, duration)
    split_eval = None if args.skip_eval else run_evaluator(args, pred, args.eval_split, args.out_dir)
    check = collapse_check(train_usage, pred_usage, args)
    decision = gate_decision(split_eval, check, args.eval_split)
    summary = {
        "created_at": now_iso(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_rows_source": row_source,
        "duration_stats_source": duration_source,
        "checkpoint": str(ckpt),
        "checkpoint_meta": {
            "step": int(ckpt_data.get("step", -1)),
            "best_val": ckpt_data.get("best_val"),
            "split": ckpt_data.get("split"),
            "ema": ckpt_data.get("ema"),
            "decoded_aux": ckpt_data.get("decoded_aux"),
        },
        "prediction": str(pred),
        "split_eval": split_eval,
        "dev_eval": split_eval if args.eval_split == "dev" else None,
        "test_eval": split_eval if args.eval_split == "test" else None,
        "prediction_token_usage": {
            "train": train_usage,
            "generated": pred_usage,
            "collapse_check": check,
        },
        "gate": decision,
        "baselines": {
            "k3_baseline_dev_bleu4": K3_BASELINE_DEV_BLEU4,
            "monotonic_prior_dev_bleu4": MONOTONIC_PRIOR_DEV_BLEU4,
            "cross_prior_interpolation_cross_prior_dev_bleu4": CROSS_PRIOR_INTERPOLATION_DEV_BLEU4,
            "mixture_prior_current_best_dev_bleu4": MIXTURE_PRIOR_DEV_BLEU4,
        },
        "data_boundary": {
            "target": "frozen_v9_whole_pose_rvq_k3_learned_motion_token",
            **condition_boundary(args),
            "duration_alignment": "train_only_duration_stats_expanded_to_timeline_with_phase_features",
            "duration_stats_source": duration_source,
            "architecture": "condition_encoder_plus_alignment_query_decoder",
            "ema_stabilization": bool(float(args.ema_decay) > 0.0),
            "decoded_auxiliary": {
                "enabled": bool(float(args.decoded_aux_weight) > 0.0),
                "source": "train_gt_motion_tokens_decoded_through_frozen_tokenizer",
                "uses_official_bt_for_training_or_selection": False,
                "weight": float(args.decoded_aux_weight),
                "temperature": float(args.decoded_aux_temperature),
                "velocity_weight": float(args.decoded_aux_velocity_weight),
                "max_windows_per_batch": int(args.decoded_aux_max_windows_per_batch),
                "overlap_enabled": bool(float(args.decoded_overlap_aux_weight) > 0.0),
                "overlap_weight": float(args.decoded_overlap_aux_weight),
                "overlap_max_pairs_per_batch": int(args.decoded_overlap_aux_max_pairs_per_batch),
                "latent_delta_enabled": bool(float(args.latent_delta_aux_weight) > 0.0),
                "latent_delta_weight": float(args.latent_delta_aux_weight),
                "latent_energy_weight": float(args.latent_energy_aux_weight),
                "latent_delta_max_pairs_per_batch": int(args.latent_delta_max_pairs_per_batch),
                "latent_energy_floor_ratio": float(args.latent_energy_floor_ratio),
                "val_include_aux": bool(args.val_include_aux),
            },
            "teacher_kl": {
                "enabled": bool(float(args.teacher_kl_weight) > 0.0),
                "source": "frozen clean legacy coarse prior-style prior logits; no BT/dev/test/winner signal",
                "weight": float(args.teacher_kl_weight),
                "temperature": float(args.teacher_kl_temperature),
            },
            "uses_bbest_or_gus_pose_for_training": False,
            "uses_bbest_or_gus_pose_for_rerank_or_fallback": False,
            "uses_official_bt_for_training_or_selection": False,
            "uses_test_pt_for_training": False,
            "uses_test_pt_for_final_evaluation": bool(args.eval_split == "test" and not args.skip_eval),
            "uses_train_predicted_gloss_for_condition_augmentation": bool(args.train_predicted_gloss_copies > 0 or args.val_predicted_gloss),
            "token_is_gloss": False,
            "token_is_single_pose_frame": False,
        },
    }
    write_json(args.out_dir / "reports" / "rvq_aligned_prior_summary.json", summary)
    render_report(args, summary)


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[1]
    default_root = default_project / "g2p_ddm_token_exps"
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--out-dir", type=Path, default=default_root / "outputs" / "rvq_aligned_prior_aligned_prior")
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    parser.add_argument("--duration-stats-json", type=Path, default=None)
    parser.add_argument("--condition-mode", choices=["gloss"], default="gloss")
    parser.add_argument("--alignment-mode", choices=["full", "none"], default="full")
    parser.add_argument("--config-id", default="beam5_lp1p0_max100")
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--eval-condition-source", choices=["predicted_gloss", "gt_gloss"], default="predicted_gloss")
    parser.add_argument("--train-predicted-gloss-json", type=Path, default=None)
    parser.add_argument("--train-predicted-gloss-copies", type=int, default=0)
    parser.add_argument("--train-predicted-gloss-rank", type=int, default=0)
    parser.add_argument("--val-predicted-gloss", action="store_true")
    parser.add_argument("--eval-env", default="t2s-oracle")
    parser.add_argument("--eval-python", type=Path, default=None)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--split-mode", choices=["random", "hash", "length_hash"], default="random")
    parser.add_argument("--split-length-bins", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prior-ckpt", type=Path, default=None)
    parser.add_argument("--init-prior-ckpt", type=Path, default=None)
    parser.add_argument("--teacher-prior-ckpt", type=Path, default=None)
    parser.add_argument("--allow-init-pred-layer-mismatch", action="store_true")
    parser.add_argument("--allow-teacher-pred-layer-mismatch", action="store_true")
    parser.add_argument("--freeze-non-new-heads", action="store_true")
    parser.add_argument("--new-head-start-layer", type=int, default=-1)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--train-only-no-generate", action="store_true")
    parser.add_argument("--prediction-suffix", default="aligned_prior_aligned_prior_v1")
    parser.add_argument("--rebuild-token-cache", action="store_true")
    parser.add_argument("--duration-scale", type=float, default=1.35)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-codes", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-quantizers", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--feature-mode", default="raw_pose")
    parser.add_argument("--pose-dct-components", type=int, default=2)
    parser.add_argument("--velocity-dct-components", type=int, default=2)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--decode-overlap-window", choices=["uniform", "triangular", "hann"], default="uniform")
    parser.add_argument("--decode-overlap-floor", type=float, default=0.25)
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
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--ema-warmup-steps", type=int, default=0)
    parser.add_argument("--coarse-weight", type=float, default=0.9)
    parser.add_argument("--decoded-aux-weight", type=float, default=0.0)
    parser.add_argument("--decoded-aux-temperature", type=float, default=1.0)
    parser.add_argument("--decoded-aux-velocity-weight", type=float, default=0.5)
    parser.add_argument("--decoded-aux-max-windows-per-batch", type=int, default=256)
    parser.add_argument("--decoded-aux-loss", choices=["smooth_l1", "l1"], default="smooth_l1")
    parser.add_argument("--decoded-overlap-aux-weight", type=float, default=0.0)
    parser.add_argument("--decoded-overlap-aux-max-pairs-per-batch", type=int, default=256)
    parser.add_argument("--latent-delta-aux-weight", type=float, default=0.0)
    parser.add_argument("--latent-energy-aux-weight", type=float, default=0.0)
    parser.add_argument("--latent-delta-max-pairs-per-batch", type=int, default=256)
    parser.add_argument("--latent-energy-floor-ratio", type=float, default=0.75)
    parser.add_argument("--val-include-aux", action="store_true")
    parser.add_argument("--teacher-kl-weight", type=float, default=0.0)
    parser.add_argument("--teacher-kl-temperature", type=float, default=1.0)
    parser.add_argument("--stable-token-kl-weight", type=float, default=0.0)
    parser.add_argument("--stable-token-kl-temperature", type=float, default=1.0)
    parser.add_argument("--stable-token-kl-layer", type=int, default=0)
    parser.add_argument("--stable-token-kl-max-pairs-per-batch", type=int, default=256)
    parser.add_argument("--pred-layers", type=int, default=3)
    parser.add_argument("--train-seconds", type=int, default=32400)
    parser.add_argument("--max-steps", type=int, default=12000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=280)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--collapse-active-frac", type=float, default=0.3)
    parser.add_argument("--collapse-top1-ratio", type=float, default=0.5)
    parser.add_argument("--collapse-entropy-frac", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
