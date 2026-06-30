#!/usr/bin/env python3
"""Hybrid decode: coarse K3 tokens plus detail K4 token."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from rvq_aligned_prior_experiment import (
    apply_alignment_mode,
    encode_condition_values,
    expand_alignment_inputs,
    load_cache_vocab_duration,
    load_generation_items,
    load_prior,
    maybe_override_duration_stats,
    now_iso,
    weight_tag,
)
from rvq_monotonic_prior_experiment import collapse_check, token_usage_from_sequences
from rvq_prior_experiment import (
    decode_tokens_to_pose,
    load_model_from_ckpt,
    make_starts,
    predict_frame_len,
    read_json,
    run_evaluator,
    topk_path,
    validate_prediction_only,
    write_json,
)


def generate_hybrid(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    vocab, duration = load_cache_vocab_duration(args.token_cache_dir)
    duration, duration_source = maybe_override_duration_stats(args, duration)
    coarse_prior, coarse_vocab, coarse_meta = load_prior(args, args.coarse_ckpt)
    detail_prior, detail_vocab, detail_meta = load_prior(args, args.detail_ckpt)
    if coarse_vocab != vocab or detail_vocab != vocab:
        raise ValueError("checkpoint vocab differs from token cache vocab")
    if int(coarse_meta.get("pred_layers", 0)) < 3:
        raise ValueError("coarse checkpoint must predict at least 3 layers")
    if int(detail_meta.get("pred_layers", 0)) < 4:
        raise ValueError("detail checkpoint must predict at least 4 layers")
    coarse_prior.eval()
    detail_prior.eval()

    tok_model, norm, tok_data = load_model_from_ckpt(args.tokenizer_ckpt, args)
    codebooks = tok_data["posthoc_codebooks"].to(args.device)
    zero_codes = codebooks.pow(2).sum(dim=2).argmin(dim=1).cpu()

    items, generation_source = load_generation_items(args, args.eval_split)
    condition_source = str(getattr(args, "eval_condition_source", "predicted_gloss") or "predicted_gloss")
    if args.max_eval_samples > 0:
        items = items[: args.max_eval_samples]

    pred: dict[str, torch.Tensor] = {}
    generated_tokens: list[torch.Tensor] = []
    with torch.no_grad():
        for sid, row in items:
            if condition_source == "gt_gloss":
                gloss = row.get("gt_gloss") or row.get("gloss", "")
            else:
                candidates = row.get("candidates", [])
                if not candidates:
                    raise ValueError(f"{sid} has no predicted gloss candidates")
                rank = int(args.candidate_rank)
                if rank >= len(candidates) and bool(getattr(args, "candidate_rank_fallback_zero", False)):
                    rank = 0
                if rank >= len(candidates):
                    raise ValueError(f"{sid} has {len(candidates)} candidates, cannot use rank {args.candidate_rank}")
                gloss = candidates[rank]["pred_gloss"]
            frame_len = predict_frame_len(gloss, duration, args.min_len, args.max_len)
            frame_len = int(max(args.min_len, min(args.max_len, round(frame_len * args.duration_scale))))
            token_len = len(make_starts(frame_len, args.window_size, args.stride))
            condition = encode_condition_values(row.get("text", ""), gloss, vocab, args).unsqueeze(0).to(args.device)
            timeline, align = expand_alignment_inputs(gloss, token_len, vocab, duration)
            timeline, align = apply_alignment_mode(timeline, align, args)
            timeline = timeline.unsqueeze(0).to(args.device)
            align = align.unsqueeze(0).to(args.device)
            coarse_logits = coarse_prior(condition, timeline, align)
            detail_logits = detail_prior(condition, timeline, align)
            tokens = torch.empty((token_len, codebooks.shape[0]), dtype=torch.long)
            for layer in range(codebooks.shape[0]):
                tokens[:, layer] = int(zero_codes[layer])
            for layer in range(3):
                tokens[:, layer] = coarse_logits[layer].squeeze(0).argmax(dim=-1).cpu()
            tokens[:, 3] = detail_logits[3].squeeze(0).argmax(dim=-1).cpu()
            generated_tokens.append(tokens[:, :4].clone())
            pred[sid] = decode_tokens_to_pose(tok_model, tokens, norm, codebooks, frame_len, args)

    suffix = f"_{args.prediction_suffix}" if args.prediction_suffix else ""
    if condition_source == "gt_gloss":
        suffix = f"{suffix}_gtgloss"
    suffix = f"{suffix}_k3plus1_dur{weight_tag(args.duration_scale)}"
    out = args.out_dir / "predictions" / args.eval_split / f"hybrid_rvq_{args.config_id}{suffix}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pred, out)
    write_json(out.with_suffix(".validation.json"), validate_prediction_only(pred))
    usage = token_usage_from_sequences(generated_tokens, args.n_codes, 4)
    write_json(out.with_suffix(".token_usage.json"), usage)
    summary = {
        "created_at": now_iso(),
        "prediction": str(out),
        "duration_stats_source": duration_source,
        "generation_source": generation_source,
        "condition_source": condition_source,
        "coarse_ckpt": str(args.coarse_ckpt),
        "detail_ckpt": str(args.detail_ckpt),
        "coarse_meta": {"step": int(coarse_meta.get("step", -1)), "pred_layers": int(coarse_meta.get("pred_layers", -1)), "best_val": coarse_meta.get("best_val")},
        "detail_meta": {"step": int(detail_meta.get("step", -1)), "pred_layers": int(detail_meta.get("pred_layers", -1)), "best_val": detail_meta.get("best_val")},
        "prediction_token_usage": {"generated": usage},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "data_boundary": {
            "uses_train_gt_pose_tokens_only_for_training_sources": True,
            "uses_dev_pose_for_training_or_selection": False,
            "uses_bbest_gus_winner_pose": False,
            "uses_retrieval": False,
            "uses_test_pt": bool(args.eval_split == "test"),
            "hybrid": "layers_0_2_from_coarse_ckpt_layer_3_from_detail_ckpt_zero_fill_remaining_layers",
        },
    }
    write_json(args.out_dir / "reports" / "hybrid_rvq_summary.json", summary)
    return out, usage


def run(args: argparse.Namespace) -> None:
    if args.eval_split == "test" and not args.include_test:
        raise RuntimeError("test evaluation requires --include-test")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred, usage = generate_hybrid(args)
    split_eval = None if args.skip_eval else run_evaluator(args, pred, args.eval_split, args.out_dir)
    train_rows = torch.load(args.token_cache_dir / "train_rvq_tokens.pt", map_location="cpu")
    train_usage = token_usage_from_sequences([row["tokens"][:, :4] for row in train_rows], args.n_codes, 4)
    check = collapse_check(train_usage, usage, args)
    summary_path = args.out_dir / "reports" / "hybrid_rvq_summary.json"
    summary = read_json(summary_path)
    summary.update({"split_eval": split_eval, "prediction_token_usage": {"train": train_usage, "generated": usage, "collapse_check": check}})
    write_json(summary_path, summary)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--out-dir", type=Path, default=root / "outputs" / "hybrid_rvq_k4")
    parser.add_argument("--tokenizer-ckpt", type=Path, required=True)
    parser.add_argument("--token-cache-dir", type=Path, required=True)
    parser.add_argument("--duration-stats-json", type=Path, default=None)
    parser.add_argument("--coarse-ckpt", type=Path, required=True)
    parser.add_argument("--detail-ckpt", type=Path, required=True)
    parser.add_argument("--config-id", default="beam1_lp0p8_max100")
    parser.add_argument("--candidate-rank", type=int, default=0)
    parser.add_argument("--eval-condition-source", choices=["predicted_gloss", "gt_gloss"], default="predicted_gloss")
    parser.add_argument("--candidate-rank-fallback-zero", action="store_true")
    parser.add_argument("--eval-python", type=Path, default=None)
    parser.add_argument("--eval-env", default="t2s-oracle")
    parser.add_argument("--eval-split", choices=["dev", "test"], default="dev")
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--prediction-suffix", default="dev")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--pred-layers", type=int, default=4)
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--decode-overlap-window", choices=["uniform", "triangular", "hann"], default="uniform")
    parser.add_argument("--decode-overlap-floor", type=float, default=0.25)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-len", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=280)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--collapse-active-frac", type=float, default=0.3)
    parser.add_argument("--collapse-top1-ratio", type=float, default=0.5)
    parser.add_argument("--collapse-entropy-frac", type=float, default=0.5)
    parser.add_argument("--condition-mode", choices=["gloss"], default="gloss")
    parser.add_argument("--alignment-mode", choices=["full", "none"], default="full")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
