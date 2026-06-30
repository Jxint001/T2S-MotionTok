#!/usr/bin/env python3
"""Monotonic gloss-timeline K3 RVQ token prior.

本脚本复用 frozen v9 whole-pose temporal RVQ tokenizer，把 train GT pose
编码成 learned K3 motion token。与 K3 baseline 的 learned-query NAT 不同，本脚本先用
train-only duration 统计把 gloss token 显式展开到 token timeline，再做逐位置
K3 token classification。

数据边界：
- token target 只来自 train GT pose 经 frozen tokenizer 编码；
- duration/alignment 只来自 train split 统计；
- dev/test pose 只由 official evaluator 读取；
- 不使用 B-best/GUS/winner pose 训练、rerank 或 fallback；
- 默认不允许读取 test.pt。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rvq_prior_experiment import (
    PAD,
    UNK,
    PositionalEncoding,
    build_token_rows,
    build_vocab,
    condition_boundary,
    decode_tokens_to_pose,
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


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def allocate_duration_counts(gloss_tokens: list[str], token_len: int, duration: dict[str, Any]) -> list[int]:
    if token_len <= 0:
        return []
    if not gloss_tokens:
        return [token_len]
    n = len(gloss_tokens)
    if token_len < n:
        counts = [0] * n
        for idx in range(token_len):
            counts[idx] = 1
        return counts
    global_dur = float(duration.get("global", 12.0) or 12.0)
    per_gloss = duration.get("per_gloss", {}) or {}
    weights = [max(float(per_gloss.get(tok, global_dur)), 1e-3) for tok in gloss_tokens]
    total = sum(weights)
    raw = [w / total * token_len for w in weights]
    counts = [max(1, int(math.floor(x))) for x in raw]
    diff = token_len - sum(counts)
    frac_order = sorted(range(n), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    if diff > 0:
        for k in range(diff):
            counts[frac_order[k % n]] += 1
    elif diff < 0:
        removable = sorted(range(n), key=lambda i: raw[i] - math.floor(raw[i]))
        need = -diff
        for idx in removable:
            if need <= 0:
                break
            take = min(counts[idx] - 1, need)
            counts[idx] -= take
            need -= take
    if sum(counts) != token_len:
        counts[-1] += token_len - sum(counts)
    return counts


def expand_gloss_timeline(gloss: str, token_len: int, vocab: dict[str, int], duration: dict[str, Any]) -> torch.Tensor:
    gloss_tokens = split_gloss(gloss)
    if not gloss_tokens:
        return torch.full((token_len,), UNK, dtype=torch.long)
    ids = [vocab.get(tok, UNK) for tok in gloss_tokens]
    counts = allocate_duration_counts(gloss_tokens, token_len, duration)
    timeline: list[int] = []
    for tok_id, count in zip(ids, counts):
        if count > 0:
            timeline.extend([tok_id] * count)
    if len(timeline) < token_len:
        timeline.extend([ids[-1]] * (token_len - len(timeline)))
    return torch.tensor(timeline[:token_len], dtype=torch.long)


class MonotonicTokenDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def make_collate(vocab: dict[str, int], duration: dict[str, Any], args: argparse.Namespace):
    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_t = max(int(x["tokens"].shape[0]) for x in batch)
        q = int(batch[0]["tokens"].shape[1])
        timeline = torch.full((len(batch), max_t), PAD, dtype=torch.long)
        tokens = torch.full((len(batch), max_t, q), -100, dtype=torch.long)
        frame_lens = torch.tensor([int(x["frame_len"]) for x in batch], dtype=torch.long)
        token_lens = torch.tensor([int(x["tokens"].shape[0]) for x in batch], dtype=torch.long)
        for i, row in enumerate(batch):
            t_len = int(row["tokens"].shape[0])
            timeline[i, :t_len] = expand_gloss_timeline(row.get("gloss", ""), t_len, vocab, duration)
            tokens[i, :t_len] = row["tokens"]
        return {"timeline": timeline, "tokens": tokens, "frame_lens": frame_lens, "token_lens": token_lens}

    return collate


class MonotonicK3Prior(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_codes: int,
        pred_layers: int,
        dim: int,
        heads: int,
        layers: int,
        dropout: float,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.n_codes = n_codes
        self.pred_layers = pred_layers
        self.gloss_emb = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = PositionalEncoding(dim, max_tokens + 256)
        enc_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)
        self.heads = nn.ModuleList([nn.Linear(dim, n_codes) for _ in range(pred_layers)])

    def forward(self, timeline: torch.Tensor) -> list[torch.Tensor]:
        pad = timeline.eq(PAD)
        x = self.pos(self.gloss_emb(timeline))
        h = self.encoder(x, src_key_padding_mask=pad)
        return [head(h) for head in self.heads]


def val_metrics(model: MonotonicK3Prior, loader: DataLoader, args: argparse.Namespace) -> dict[str, Any]:
    model.eval()
    total = 0.0
    count = 0
    layer_acc = None
    with torch.no_grad():
        for batch in loader:
            timeline = batch["timeline"].to(args.device)
            tokens = batch["tokens"].to(args.device)
            logits = model(timeline)
            loss, parts = token_loss(logits, tokens, args.coarse_weight, args.pred_layers)
            total += float(loss.item())
            count += 1
            cur_acc = parts["layer_acc"]
            layer_acc = cur_acc if layer_acc is None else [a + b for a, b in zip(layer_acc, cur_acc)]
    if layer_acc is not None:
        layer_acc = [x / max(count, 1) for x in layer_acc]
    return {"loss": total / max(count, 1), "layer_acc": layer_acc}


def token_usage_from_sequences(seqs: list[torch.Tensor], n_codes: int, pred_layers: int) -> dict[str, Any]:
    layers = []
    total_tokens = 0
    if seqs:
        total_tokens = sum(int(x.shape[0]) for x in seqs)
    for layer in range(pred_layers):
        if seqs:
            cur = torch.cat([x[:, layer].reshape(-1).cpu() for x in seqs if x.numel() > 0], dim=0)
        else:
            cur = torch.empty(0, dtype=torch.long)
        if cur.numel() == 0:
            layers.append({"layer": layer, "active_codes": 0, "entropy": 0.0, "perplexity": 0.0, "top1_ratio": 0.0})
            continue
        counts = torch.bincount(cur.clamp(0, n_codes - 1), minlength=n_codes).float()
        probs = counts / counts.sum().clamp_min(1.0)
        nonzero = probs > 0
        entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
        layers.append(
            {
                "layer": layer,
                "active_codes": int(nonzero.sum().item()),
                "entropy": float(entropy.item()),
                "perplexity": float(entropy.exp().item()),
                "top1_ratio": float(probs.max().item()),
            }
        )
    return {"num_sequences": len(seqs), "num_tokens": total_tokens, "n_codes": n_codes, "pred_layers": pred_layers, "layers": layers}


def collapse_check(train_usage: dict[str, Any], pred_usage: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    issues = []
    train_layers = train_usage.get("layers", [])
    pred_layers = pred_usage.get("layers", [])
    for train, pred in zip(train_layers, pred_layers):
        min_active = float(args.collapse_active_frac) * max(int(train.get("active_codes", 0)), 1)
        min_entropy = float(args.collapse_entropy_frac) * max(float(train.get("entropy", 0.0)), 1e-6)
        if int(pred.get("active_codes", 0)) < min_active:
            issues.append({"layer": pred.get("layer"), "reason": "active_code_collapse", "train_active_codes": train.get("active_codes"), "generated_active_codes": pred.get("active_codes"), "threshold": min_active})
        if float(pred.get("top1_ratio", 0.0)) > float(args.collapse_top1_ratio):
            issues.append({"layer": pred.get("layer"), "reason": "top1_ratio_high", "top1_ratio": pred.get("top1_ratio"), "threshold": float(args.collapse_top1_ratio)})
        if float(pred.get("entropy", 0.0)) < min_entropy:
            issues.append({"layer": pred.get("layer"), "reason": "entropy_low", "train_entropy": train.get("entropy"), "generated_entropy": pred.get("entropy"), "threshold": min_entropy})
    return {
        "ok": not issues,
        "num_issues": len(issues),
        "thresholds": {
            "active_frac_of_train": float(args.collapse_active_frac),
            "max_top1_ratio": float(args.collapse_top1_ratio),
            "min_entropy_frac_of_train": float(args.collapse_entropy_frac),
        },
        "issues": issues,
    }


def train_prior(args: argparse.Namespace, rows: list[dict[str, Any]], vocab: dict[str, int], duration: dict[str, Any], out_dir: Path) -> Path:
    ckpt = out_dir / "checkpoints" / "rvq_monotonic_prior_best.pt"
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
    collate = make_collate(vocab, duration, args)
    train_loader = DataLoader(MonotonicTokenDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(MonotonicTokenDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    max_tokens = max(int(x["tokens"].shape[0]) for x in rows) + 16
    torch.manual_seed(init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(init_seed)
    model = MonotonicK3Prior(len(vocab), args.n_codes, args.pred_layers, args.dim, args.heads, args.layers, args.dropout, max_tokens).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    history = []
    deadline = time.time() + args.train_seconds
    step = 0
    while time.time() < deadline and step < args.max_steps:
        model.train()
        for batch in train_loader:
            step += 1
            timeline = batch["timeline"].to(args.device)
            tokens = batch["tokens"].to(args.device)
            logits = model(timeline)
            loss, parts = token_loss(logits, tokens, args.coarse_weight, args.pred_layers)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite monotonic prior loss at step {step}: {loss.item()}")
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            if step == 1 or step % args.val_every == 0:
                val = val_metrics(model, val_loader, args)
                row = {
                    "step": step,
                    "time": now_iso(),
                    "train_loss": float(loss.detach().cpu()),
                    "train_layer_acc": parts["layer_acc"],
                    "grad_norm": float(torch.as_tensor(grad).cpu()),
                    "val": val,
                }
                if step == 1:
                    row["split"] = split_summary
                    row["method"] = "duration_expanded_gloss_timeline_tagger"
                history.append(row)
                write_json(out_dir / "logs" / "rvq_monotonic_prior_history.json", history)
                if float(val["loss"]) < best:
                    best = float(val["loss"])
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "vocab": vocab,
                            "pred_layers": int(args.pred_layers),
                            "max_tokens": int(max_tokens),
                            "best_val": best,
                            "step": step,
                            "split": split_summary,
                            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                        },
                        ckpt,
                    )
            if time.time() >= deadline or step >= args.max_steps:
                break
    if not ckpt.exists():
        val = val_metrics(model, val_loader, args)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "vocab": vocab,
                "pred_layers": int(args.pred_layers),
                "max_tokens": int(max_tokens),
                "best_val": float(val["loss"]),
                "step": step,
                "split": split_summary,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            },
            ckpt,
        )
    return ckpt


def load_prior(args: argparse.Namespace, ckpt: Path) -> tuple[MonotonicK3Prior, dict[str, int], dict[str, Any]]:
    data = load_torch(ckpt)
    vocab = data["vocab"]
    model = MonotonicK3Prior(
        len(vocab),
        args.n_codes,
        int(data.get("pred_layers", args.pred_layers)),
        args.dim,
        args.heads,
        args.layers,
        args.dropout,
        int(data.get("max_tokens", args.max_tokens)),
    ).to(args.device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, vocab, data


def generate_split(
    args: argparse.Namespace,
    split: str,
    model: MonotonicK3Prior,
    vocab: dict[str, int],
    duration: dict[str, Any],
    out_dir: Path,
) -> tuple[Path, dict[str, Any]]:
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
    generated_tokens = []
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
            timeline = expand_gloss_timeline(gloss, token_len, vocab, duration).unsqueeze(0).to(args.device)
            logits = model(timeline)
            tokens = torch.empty((token_len, codebooks.shape[0]), dtype=torch.long)
            for layer in range(codebooks.shape[0]):
                tokens[:, layer] = int(zero_codes[layer])
            for layer, cur in enumerate(logits):
                tokens[:, layer] = cur.squeeze(0).argmax(dim=-1).cpu()
            generated_tokens.append(tokens[:, : args.pred_layers].clone())
            pred[sid] = decode_tokens_to_pose(tok_model, tokens, norm, codebooks, frame_len, args)
    suffix = f"_{args.prediction_suffix}" if args.prediction_suffix else ""
    if cand_rank != 0:
        suffix = f"{suffix}_rank{cand_rank}"
    suffix = f"{suffix}_k{args.pred_layers}"
    if args.duration_scale != 1.0:
        suffix = f"{suffix}_dur{str(args.duration_scale).replace('.', 'p')}"
    out = out_dir / "predictions" / split / f"rvq_monotonic_prior_{args.config_id}{suffix}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    write_json(out.with_suffix(".validation.json"), validate_prediction_only(pred))
    usage = token_usage_from_sequences(generated_tokens, args.n_codes, args.pred_layers)
    write_json(out.with_suffix(".token_usage.json"), usage)
    return out, usage


def render_report(out_dir: Path, summary: dict[str, Any]) -> None:
    split = summary["args"].get("eval_split", "dev")
    eval_result = summary.get("split_eval") or {}
    metrics = eval_result.get("metrics", {}) if eval_result else {}
    bleu = metrics.get("bleu", {}) if metrics else {}
    lines = [
        "# RVQ monotonic prior 实验报告",
        "",
        "## 方法",
        "",
        "冻结 v9 temporal RVQ tokenizer，target 为 train GT pose encode 后的 K3 learned motion token。prior 使用 train-only duration stats 将 gloss 展开到 token timeline，再逐位置预测 K3 token。",
        "",
        "## Eval 结果",
        "",
        f"- split: {split}",
        f"- status: {eval_result.get('status', 'skipped') if eval_result else 'skipped'}",
        f"- BLEU4: {bleu.get('bleu4', 'NA')}",
        f"- WER: {metrics.get('wer', 'NA') if metrics else 'NA'}",
        f"- DTW-MJE: {metrics.get('dtw_mje', 'NA') if metrics else 'NA'}",
        f"- collapse_ok: {summary.get('prediction_token_usage', {}).get('collapse_check', {}).get('ok', 'NA')}",
        "",
        "## 判定",
        "",
        "dev 未明确超过 K3 baseline `11.1932` 前不跑 test。",
    ]
    suffix = f"_{summary['args'].get('prediction_suffix')}" if summary["args"].get("prediction_suffix") else ""
    suffix = f"{suffix}_k{summary['args'].get('pred_layers')}"
    if float(summary["args"].get("duration_scale", 1.0) or 1.0) != 1.0:
        suffix = f"{suffix}_dur{str(summary['args']['duration_scale']).replace('.', 'p')}"
    path = out_dir / "reports" / f"rvq_monotonic_prior_report{suffix}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "reports" / f"rvq_monotonic_prior_summary{suffix}.json", summary)


def run(args: argparse.Namespace) -> None:
    if args.eval_split == "test" and not args.skip_eval and not args.include_test:
        raise RuntimeError("test evaluation 需要显式传入 --include-test")
    if args.eval_split == "test":
        raise RuntimeError("monotonic prior gate 禁止读取 test.pt；如需 final test 必须新建冻结候选脚本")
    if args.pred_layers <= 0:
        raise ValueError("--pred-layers must be > 0")
    if args.candidate_rank < 0:
        raise ValueError("--candidate-rank must be >= 0")
    for name in ("collapse_active_frac", "collapse_top1_ratio", "collapse_entropy_frac"):
        value = float(getattr(args, name))
        if value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_torch(args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / "train.pt")
    vocab = build_vocab(train, args)
    rows, duration = build_token_rows(args, train, vocab, args.out_dir)
    if args.pred_layers > int(rows[0]["tokens"].shape[1]):
        raise ValueError(f"--pred-layers {args.pred_layers} exceeds tokenizer layers {rows[0]['tokens'].shape[1]}")
    train_usage = token_usage_from_sequences([row["tokens"][:, : args.pred_layers] for row in rows], args.n_codes, args.pred_layers)
    ckpt = args.prior_ckpt if args.prior_ckpt else train_prior(args, rows, vocab, duration, args.out_dir)
    model, vocab, ckpt_data = load_prior(args, ckpt)
    pred, pred_usage = generate_split(args, args.eval_split, model, vocab, duration, args.out_dir)
    split_eval = None if args.skip_eval else run_evaluator(args, pred, args.eval_split, args.out_dir)
    check = collapse_check(train_usage, pred_usage, args)
    summary = {
        "created_at": now_iso(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": str(ckpt),
        "checkpoint_meta": {
            "step": int(ckpt_data.get("step", -1)),
            "best_val": ckpt_data.get("best_val"),
            "split": ckpt_data.get("split"),
        },
        "prediction": str(pred),
        "split_eval": split_eval,
        "dev_eval": split_eval if args.eval_split == "dev" else None,
        "data_boundary": {
            "target": "train_gt_pose_encoded_by_frozen_v9_temporal_rvq_k3",
            **condition_boundary(args),
            "duration_alignment": "train_only_gloss_duration_stats_expanded_to_token_timeline",
            "uses_bbest_or_gus_pose_for_training": False,
            "uses_bbest_or_gus_pose_for_rerank_or_fallback": False,
            "uses_test_pt_for_training": False,
            "uses_test_pt_for_final_evaluation": False,
            "token_is_gloss": False,
            "token_is_single_pose_frame": False,
        },
        "prediction_token_usage": {
            "train": train_usage,
            "generated": pred_usage,
            "collapse_check": check,
        },
        "baseline": {
            "k3_baseline_dev_bleu4": 11.193247069579575,
            "current_b_best_dev_bleu4": 10.877469287658991,
            "k3_baseline_final_test_bleu4": 10.919712662019919,
        },
    }
    render_report(args.out_dir, summary)


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--out-dir", type=Path, default=default_project / "g2p_ddm_token_exps" / "outputs" / "rvq_monotonic_prior")
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--condition-mode", choices=["gloss"], default="gloss")
    parser.add_argument("--config-id", default="beam5_lp1p0_max100")
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--eval-env", default="t2s-oracle")
    parser.add_argument("--eval-python", type=Path, default=None)
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prior-ckpt", type=Path, default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--prediction-suffix", default="")
    parser.add_argument("--duration-scale", type=float, default=1.35)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--n-codes", type=int, default=1024)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-quantizers", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--coarse-weight", type=float, default=0.9)
    parser.add_argument("--pred-layers", type=int, default=3)
    parser.add_argument("--train-seconds", type=int, default=7200)
    parser.add_argument("--max-steps", type=int, default=5000)
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
