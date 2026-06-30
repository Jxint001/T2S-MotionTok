#!/usr/bin/env python3
"""Learnable temporal RVQ tokenizer gate.

本脚本只验证 tokenizer 本身是否能保住 pose 信息，不训练 text/gloss -> token prior。

数据边界：
- encoder/decoder/RVQ codebook 只用 train.pt 的 pose window 训练；
- normalization 只从 train.pt 统计；
- dev pose 只由 official evaluator 读取；
- test.pt 不读取。
"""

from __future__ import annotations

import argparse
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


BBEST_REL = Path(
    "exps/task3/outputs/task3_followup_abc/predictions/dev/"
    "B4_t2g_oov_duration_gus_a0p5_b0p2_c0p1_beam5_lp1p0_max100_rel_gus_predgloss.pt"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def to_pose(x: Any) -> torch.Tensor:
    pose = torch.as_tensor(x, dtype=torch.float32)
    if pose.ndim != 3 or tuple(pose.shape[1:]) != (178, 3):
        raise ValueError(f"bad pose shape: {tuple(pose.shape)}")
    return pose


def compute_pose_norm(samples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    total = torch.zeros(178 * 3)
    total_sq = torch.zeros(178 * 3)
    count = 0
    for pose in samples:
        flat = pose.reshape(-1, 178 * 3)
        total += flat.sum(0)
        total_sq += (flat * flat).sum(0)
        count += flat.shape[0]
    mean = total / max(count, 1)
    var = total_sq / max(count, 1) - mean * mean
    return {"mean": mean, "std": var.clamp_min(1e-6).sqrt()}


def normalize_pose(pose: torch.Tensor, norm: dict[str, torch.Tensor]) -> torch.Tensor:
    flat = (pose.reshape(-1, 178 * 3) - norm["mean"]) / norm["std"]
    return flat.reshape_as(pose)


def denormalize_pose(pose: torch.Tensor, norm: dict[str, torch.Tensor]) -> torch.Tensor:
    flat = pose.reshape(-1, 178 * 3) * norm["std"] + norm["mean"]
    return flat.reshape_as(pose).to(torch.float32)


def make_starts(length: int, window_size: int, stride: int) -> list[int]:
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    last = length - window_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_window(pose: torch.Tensor, start: int, window_size: int) -> torch.Tensor:
    end = start + window_size
    if end <= pose.shape[0]:
        return pose[start:end]
    tail = pose[-1:].expand(end - pose.shape[0], -1, -1)
    return torch.cat([pose[start:], tail], dim=0)


def dct_basis(length: int, components: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    if components <= 0 or components > length:
        raise ValueError(f"bad DCT components {components} for length {length}")
    t = torch.arange(length, device=device, dtype=dtype)
    k = torch.arange(components, device=device, dtype=dtype).unsqueeze(1)
    basis = torch.cos(math.pi / length * (t + 0.5).unsqueeze(0) * k)
    basis[0] *= math.sqrt(1.0 / length)
    if components > 1:
        basis[1:] *= math.sqrt(2.0 / length)
    return basis


def feature_mode(args: argparse.Namespace) -> str:
    return str(getattr(args, "feature_mode", "raw_pose") or "raw_pose")


def tokenizer_input_dim(args: argparse.Namespace) -> int:
    if feature_mode(args) == "raw_pose":
        return int(args.window_size) * 178 * 3
    if feature_mode(args) == "coarse_velocity":
        pose_components = int(getattr(args, "pose_dct_components", 2))
        velocity_components = int(getattr(args, "velocity_dct_components", 2))
        return (pose_components + velocity_components) * 178 * 3
    raise ValueError(f"unknown feature_mode: {feature_mode(args)}")


def window_features(window: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    mode = feature_mode(args)
    if window.ndim != 4 or window.shape[2:] != (178, 3):
        raise ValueError(f"bad window shape: {tuple(window.shape)}")
    if mode == "raw_pose":
        return window.reshape(window.shape[0], -1)
    if mode != "coarse_velocity":
        raise ValueError(f"unknown feature_mode: {mode}")
    pose_components = int(getattr(args, "pose_dct_components", 2))
    velocity_components = int(getattr(args, "velocity_dct_components", 2))
    if pose_components <= 0 or pose_components > window.shape[1]:
        raise ValueError(f"bad --pose-dct-components {pose_components}")
    if window.shape[1] < 2:
        raise ValueError("coarse_velocity feature_mode requires window_size >= 2")
    if velocity_components <= 0 or velocity_components > window.shape[1] - 1:
        raise ValueError(f"bad --velocity-dct-components {velocity_components}")
    pose_basis = dct_basis(window.shape[1], pose_components, window.device, window.dtype)
    pose_coeff = torch.einsum("kw,bwjc->bkjc", pose_basis, window)
    velocity = window[:, 1:] - window[:, :-1]
    velocity_basis = dct_basis(velocity.shape[1], velocity_components, window.device, window.dtype)
    velocity_coeff = torch.einsum("kw,bwjc->bkjc", velocity_basis, velocity)
    return torch.cat([pose_coeff.reshape(window.shape[0], -1), velocity_coeff.reshape(window.shape[0], -1)], dim=1)


def tokenizer_input_dim_from_cfg(cfg: dict[str, Any], args: argparse.Namespace) -> int:
    if "input_dim" in cfg:
        return int(cfg["input_dim"])
    ns = argparse.Namespace(
        window_size=int(cfg.get("window_size", args.window_size)),
        feature_mode=cfg.get("feature_mode", getattr(args, "feature_mode", "raw_pose")),
        pose_dct_components=int(cfg.get("pose_dct_components", getattr(args, "pose_dct_components", 2))),
        velocity_dct_components=int(cfg.get("velocity_dct_components", getattr(args, "velocity_dct_components", 2))),
    )
    return tokenizer_input_dim(ns)


def split_train_ids(ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = list(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    val_n = max(64, int(len(ids) * val_ratio))
    return ids[val_n:], ids[:val_n]


class PoseWindowDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        poses: dict[str, torch.Tensor],
        norm: dict[str, torch.Tensor],
        window_size: int,
        stride: int,
        max_windows: int | None,
        seed: int,
    ) -> None:
        self.poses = {sid: normalize_pose(poses[sid], norm) for sid in sample_ids}
        self.window_size = window_size
        index: list[tuple[str, int]] = []
        for sid in sample_ids:
            for start in make_starts(poses[sid].shape[0], window_size, stride):
                index.append((sid, start))
        if max_windows is not None and len(index) > max_windows:
            rng = random.Random(seed)
            index = rng.sample(index, max_windows)
        self.index = index

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sid, start = self.index[idx]
        return extract_window(self.poses[sid], start, self.window_size)


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, n_quantizers: int, n_codes: int, dim: int, beta: float) -> None:
        super().__init__()
        self.n_quantizers = n_quantizers
        self.n_codes = n_codes
        self.beta = beta
        self.codebooks = nn.ModuleList([nn.Embedding(n_codes, dim) for _ in range(n_quantizers)])
        for emb in self.codebooks:
            nn.init.uniform_(emb.weight, -1.0 / n_codes, 1.0 / n_codes)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = z
        quantized = torch.zeros_like(z)
        losses = []
        all_indices = []
        for emb in self.codebooks:
            weight = emb.weight
            dist = (
                residual.pow(2).sum(dim=1, keepdim=True)
                + weight.pow(2).sum(dim=1).unsqueeze(0)
                - 2 * residual @ weight.t()
            )
            idx = dist.argmin(dim=1)
            q = emb(idx)
            losses.append(F.mse_loss(q, residual.detach()) + self.beta * F.mse_loss(residual, q.detach()))
            quantized = quantized + q
            residual = residual - q.detach()
            all_indices.append(idx)
        quantized_st = z + (quantized - z).detach()
        return quantized_st, torch.stack(losses).sum(), torch.stack(all_indices, dim=1)


class RVQTokenizer(nn.Module):
    def __init__(
        self,
        window_size: int,
        latent_dim: int,
        hidden_dim: int,
        n_codes: int,
        n_quantizers: int,
        beta: float,
        dropout: float,
        input_dim: int | None = None,
        feature_mode_name: str = "raw_pose",
        pose_dct_components: int = 2,
        velocity_dct_components: int = 2,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        input_dim = input_dim or window_size * 178 * 3
        self.input_dim = input_dim
        self.output_dim = window_size * 178 * 3
        self.feature_args = argparse.Namespace(
            window_size=window_size,
            feature_mode=feature_mode_name,
            pose_dct_components=pose_dct_components,
            velocity_dct_components=velocity_dct_components,
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.quantizer = ResidualVectorQuantizer(n_quantizers, n_codes, latent_dim, beta)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.output_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = window_features(x, self.feature_args)
        z = self.encoder(flat)
        q, vq_loss, indices = self.quantizer(z)
        recon = self.decoder(q).reshape_as(x)
        return recon, vq_loss, indices

    def encode_decode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recon, _, indices = self.forward(x)
        return recon, indices


def point_weights(device: torch.device | str) -> torch.Tensor:
    weights = torch.ones(178, dtype=torch.float32)
    weights[:8] = 2.0
    weights[8:50] = 3.0
    weights[50:] = 0.5
    weights = weights / weights.mean()
    return weights.view(1, 1, 178, 1).to(device)


def recon_loss(pred: torch.Tensor, target: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    weights = point_weights(pred.device)
    abs_err = (pred - target).abs()
    l1 = (abs_err * weights).mean()
    mse = ((pred - target).pow(2) * weights).mean()
    if pred.shape[1] > 1:
        pred_vel = pred[:, 1:] - pred[:, :-1]
        target_vel = target[:, 1:] - target[:, :-1]
        vel = ((pred_vel - target_vel).abs() * weights).mean()
    else:
        vel = pred.new_tensor(0.0)
    loss = l1 + args.mse_weight * mse + args.velocity_weight * vel
    return loss, {"l1": float(l1.detach().cpu()), "mse": float(mse.detach().cpu()), "velocity": float(vel.detach().cpu())}


def codebook_health(indices: torch.Tensor, n_codes: int) -> list[dict[str, float | int]]:
    health = []
    for q in range(indices.shape[1]):
        counts = torch.bincount(indices[:, q].cpu(), minlength=n_codes).float()
        total = counts.sum().clamp_min(1)
        probs = counts / total
        nonzero = probs > 0
        entropy = -(probs[nonzero] * probs[nonzero].log()).sum()
        health.append(
            {
                "layer": q,
                "active_codes": int((counts > 0).sum().item()),
                "perplexity": float(entropy.exp().item()),
                "dead_ratio": float((counts == 0).float().mean().item()),
            }
        )
    return health


def ae_forward(model: RVQTokenizer, x: torch.Tensor) -> torch.Tensor:
    flat = window_features(x, model.feature_args)
    z = model.encoder(flat)
    return model.decoder(z).reshape_as(x)


def posthoc_quantize(z: torch.Tensor, codebooks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    residual = z
    quantized = torch.zeros_like(z)
    all_indices = []
    for centers in codebooks:
        centers = centers.to(z.device)
        dist = (
            residual.pow(2).sum(dim=1, keepdim=True)
            + centers.pow(2).sum(dim=1).unsqueeze(0)
            - 2 * residual @ centers.t()
        )
        idx = dist.argmin(dim=1)
        q = centers[idx]
        quantized = quantized + q
        residual = residual - q
        all_indices.append(idx)
    return quantized, torch.stack(all_indices, dim=1)


def model_decode_batch(
    model: RVQTokenizer,
    x: torch.Tensor,
    args: argparse.Namespace,
    codebooks: torch.Tensor | None = None,
    ae_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if ae_only:
        return ae_forward(model, x), x.new_tensor(0.0), None
    if codebooks is None:
        recon, vq_loss, indices = model(x)
        return recon, vq_loss, indices
    flat = window_features(x, model.feature_args)
    z = model.encoder(flat)
    q, indices = posthoc_quantize(z, codebooks.to(x.device))
    recon = model.decoder(q).reshape_as(x)
    vq_loss = F.mse_loss(q, z)
    return recon, vq_loss, indices


def train_kmeans(x: torch.Tensor, n_codes: int, iters: int, batch_size: int, device: str, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    if x.shape[0] < n_codes:
        raise ValueError(f"not enough vectors for kmeans: {x.shape[0]} < {n_codes}")
    centers = x[torch.randperm(x.shape[0], generator=gen)[:n_codes]].to(device)
    x_dev = x.to(device)
    for _ in range(iters):
        sums = torch.zeros_like(centers)
        counts = torch.zeros(n_codes, device=device)
        for start in range(0, x_dev.shape[0], batch_size):
            batch = x_dev[start : start + batch_size]
            labels = torch.cdist(batch, centers).argmin(dim=1)
            sums.index_add_(0, labels, batch)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float32))
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)
    return centers.cpu()


def nearest_codes(x: torch.Tensor, centers: torch.Tensor, device: str, batch_size: int) -> torch.Tensor:
    labels = []
    centers = centers.to(device)
    for start in range(0, x.shape[0], batch_size):
        batch = x[start : start + batch_size].to(device)
        labels.append(torch.cdist(batch, centers).argmin(dim=1).cpu())
    return torch.cat(labels, dim=0)


def collect_latents(model: RVQTokenizer, loader: DataLoader, args: argparse.Namespace, max_windows: int) -> torch.Tensor:
    model.eval()
    chunks = []
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(args.device)
            z = model.encoder(window_features(batch, model.feature_args)).cpu()
            chunks.append(z)
            count += z.shape[0]
            if count >= max_windows:
                break
    return torch.cat(chunks, dim=0)[:max_windows]


def build_posthoc_codebooks(model: RVQTokenizer, loader: DataLoader, args: argparse.Namespace, out_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    latents = collect_latents(model, loader, args, args.max_codebook_windows)
    residual = latents
    codebooks = []
    stats = []
    for q in range(args.n_quantizers):
        centers = train_kmeans(residual, args.n_codes, args.kmeans_iters, args.kmeans_batch_size, args.device, args.seed + 1000 + q)
        labels = nearest_codes(residual, centers, args.device, args.kmeans_batch_size)
        before = float((residual * residual).mean().item())
        residual = residual - centers[labels]
        after = float((residual * residual).mean().item())
        codebooks.append(centers)
        stats.append(
            {
                "layer": q,
                "residual_mse_before": before,
                "residual_mse_after": after,
                "active_codes": int(torch.unique(labels).numel()),
            }
        )
    codebook = torch.stack(codebooks, dim=0)
    write_json(out_dir / "tokenizer" / "posthoc_rvq_codebook_stats.json", {"num_latents": int(latents.shape[0]), "layers": stats})
    return codebook, {"num_latents": int(latents.shape[0]), "layers": stats}


def evaluate_windows(
    model: RVQTokenizer,
    loader: DataLoader,
    args: argparse.Namespace,
    codebooks: torch.Tensor | None = None,
    ae_only: bool = False,
) -> dict[str, Any]:
    model.eval()
    totals = {"loss": 0.0, "l1": 0.0, "mse": 0.0, "velocity": 0.0, "vq": 0.0}
    count = 0
    batches = 0
    all_indices = []
    with torch.no_grad():
        for batch in loader:
            batches += 1
            batch = batch.to(args.device)
            recon, vq_loss, indices = model_decode_batch(model, batch, args, codebooks, ae_only)
            base_loss, parts = recon_loss(recon, batch, args)
            total_loss = base_loss + args.vq_weight * vq_loss
            bsz = batch.shape[0]
            totals["loss"] += float(total_loss.item()) * bsz
            totals["vq"] += float(vq_loss.item()) * bsz
            for key, value in parts.items():
                totals[key] += value * bsz
            count += bsz
            if indices is not None:
                all_indices.append(indices.cpu())
            if args.val_batches and batches >= args.val_batches:
                break
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["num_windows"] = count
    if all_indices:
        metrics["codebook"] = codebook_health(torch.cat(all_indices, dim=0), args.n_codes)
    return metrics


def save_checkpoint(
    path: Path,
    model: RVQTokenizer,
    norm: dict[str, torch.Tensor],
    args: argparse.Namespace,
    step: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "norm": norm,
        "step": step,
        "metrics": metrics,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_model_from_ckpt(path: Path, args: argparse.Namespace) -> tuple[RVQTokenizer, dict[str, torch.Tensor], dict[str, Any]]:
    data = load_torch(path)
    cfg = data.get("args", {})
    model = RVQTokenizer(
        window_size=int(cfg.get("window_size", args.window_size)),
        latent_dim=int(cfg.get("latent_dim", args.latent_dim)),
        hidden_dim=int(cfg.get("hidden_dim", args.hidden_dim)),
        n_codes=int(cfg.get("n_codes", args.n_codes)),
        n_quantizers=int(cfg.get("n_quantizers", args.n_quantizers)),
        beta=float(cfg.get("beta", args.beta)),
        dropout=float(cfg.get("dropout", args.dropout)),
        input_dim=tokenizer_input_dim_from_cfg(cfg, args),
        feature_mode_name=str(cfg.get("feature_mode", getattr(args, "feature_mode", "raw_pose"))),
        pose_dct_components=int(cfg.get("pose_dct_components", getattr(args, "pose_dct_components", 2))),
        velocity_dct_components=int(cfg.get("velocity_dct_components", getattr(args, "velocity_dct_components", 2))),
    ).to(args.device)
    model.load_state_dict(data["model"])
    model.eval()
    return model, data["norm"], data


def train_tokenizer(args: argparse.Namespace, train: dict[str, dict[str, Any]], out_dir: Path) -> Path:
    ckpt = out_dir / "checkpoints" / "rvq_tokenizer_best.pt"
    if ckpt.exists() and not args.force:
        return ckpt

    ids = sorted(train)
    train_ids, val_ids = split_train_ids(ids, args.val_ratio, args.seed)
    poses = {sid: to_pose(train[sid]["poses_3d"]) for sid in ids}
    norm = compute_pose_norm(list(poses.values()))
    train_set = PoseWindowDataset(train_ids, poses, norm, args.window_size, args.stride, args.max_train_windows, args.seed)
    val_set = PoseWindowDataset(val_ids, poses, norm, args.window_size, args.stride, args.max_val_windows, args.seed + 1)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = RVQTokenizer(
        args.window_size,
        args.latent_dim,
        args.hidden_dim,
        args.n_codes,
        args.n_quantizers,
        args.beta,
        args.dropout,
        input_dim=tokenizer_input_dim(args),
        feature_mode_name=feature_mode(args),
        pose_dct_components=int(args.pose_dct_components),
        velocity_dct_components=int(args.velocity_dct_components),
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best = float("inf")
    history = []
    deadline = time.time() + args.train_seconds
    step = 0
    if args.posthoc_rvq:
        while time.time() < deadline and step < args.max_steps:
            model.train()
            for batch in train_loader:
                step += 1
                batch = batch.to(args.device)
                recon = ae_forward(model, batch)
                loss, parts = recon_loss(recon, batch, args)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite AE loss at step {step}: {loss.item()}")
                opt.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

                if step == 1 or step % args.val_every == 0:
                    val = evaluate_windows(model, val_loader, args, ae_only=True)
                    row = {
                        "step": step,
                        "time": now_iso(),
                        "mode": "posthoc_rvq_ae",
                        "train_loss": float(loss.detach().cpu()),
                        "train_l1": parts["l1"],
                        "train_mse": parts["mse"],
                        "train_velocity": parts["velocity"],
                        "train_vq": 0.0,
                        "grad_norm": float(torch.as_tensor(grad).cpu()),
                        "val": val,
                    }
                    history.append(row)
                    write_json(out_dir / "logs" / "rvq_train_history.json", history)
                    save_checkpoint(out_dir / "checkpoints" / "rvq_tokenizer_latest.pt", model, norm, args, step, val, {"mode": "posthoc_rvq_ae"})
                    if float(val["loss"]) < best:
                        best = float(val["loss"])
                        save_checkpoint(ckpt, model, norm, args, step, val, {"mode": "posthoc_rvq_ae"})
                if time.time() >= deadline or step >= args.max_steps:
                    break
        if ckpt.exists():
            data = load_torch(ckpt)
            model.load_state_dict(data["model"])
            step = int(data.get("step", step))
        else:
            val = evaluate_windows(model, val_loader, args, ae_only=True)
            save_checkpoint(ckpt, model, norm, args, step, val, {"mode": "posthoc_rvq_ae"})
        codebooks, codebook_stats = build_posthoc_codebooks(model, train_loader, args, out_dir)
        quant_val = evaluate_windows(model, val_loader, args, codebooks=codebooks)
        save_checkpoint(
            ckpt,
            model,
            norm,
            args,
            step,
            quant_val,
            {"mode": "posthoc_rvq", "posthoc_codebooks": codebooks.cpu(), "posthoc_stats": codebook_stats},
        )
        save_checkpoint(
            out_dir / "checkpoints" / "rvq_tokenizer_latest.pt",
            model,
            norm,
            args,
            step,
            quant_val,
            {"mode": "posthoc_rvq", "posthoc_codebooks": codebooks.cpu(), "posthoc_stats": codebook_stats},
        )
        write_json(
            out_dir / "tokenizer" / "rvq_tokenizer_dataset.json",
            {
                "train_samples": len(train_ids),
                "val_samples": len(val_ids),
                "train_windows": len(train_set),
                "val_windows": len(val_set),
                "window_size": args.window_size,
                "stride": args.stride,
                "feature_mode": feature_mode(args),
                "pose_dct_components": int(args.pose_dct_components),
                "velocity_dct_components": int(args.velocity_dct_components),
                "input_dim": tokenizer_input_dim(args),
                "mode": "posthoc_rvq",
            },
        )
        return ckpt

    while time.time() < deadline and step < args.max_steps:
        model.train()
        for batch in train_loader:
            step += 1
            batch = batch.to(args.device)
            recon, vq_loss, _ = model(batch)
            base_loss, parts = recon_loss(recon, batch, args)
            loss = base_loss + args.vq_weight * vq_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite tokenizer loss at step {step}: {loss.item()}")
            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            if step == 1 or step % args.val_every == 0:
                val = evaluate_windows(model, val_loader, args)
                row = {
                    "step": step,
                    "time": now_iso(),
                    "train_loss": float(loss.detach().cpu()),
                    "train_l1": parts["l1"],
                    "train_mse": parts["mse"],
                    "train_velocity": parts["velocity"],
                    "train_vq": float(vq_loss.detach().cpu()),
                    "grad_norm": float(torch.as_tensor(grad).cpu()),
                    "val": val,
                }
                history.append(row)
                write_json(out_dir / "logs" / "rvq_train_history.json", history)
                save_checkpoint(out_dir / "checkpoints" / "rvq_tokenizer_latest.pt", model, norm, args, step, val)
                if float(val["loss"]) < best:
                    best = float(val["loss"])
                    save_checkpoint(ckpt, model, norm, args, step, val)
            if time.time() >= deadline or step >= args.max_steps:
                break
    if not ckpt.exists():
        val = evaluate_windows(model, val_loader, args)
        save_checkpoint(ckpt, model, norm, args, step, val)
    write_json(
        out_dir / "tokenizer" / "rvq_tokenizer_dataset.json",
        {
            "train_samples": len(train_ids),
            "val_samples": len(val_ids),
            "train_windows": len(train_set),
            "val_windows": len(val_set),
            "window_size": args.window_size,
            "stride": args.stride,
            "feature_mode": feature_mode(args),
            "pose_dct_components": int(args.pose_dct_components),
            "velocity_dct_components": int(args.velocity_dct_components),
            "input_dim": tokenizer_input_dim(args),
        },
    )
    return ckpt


def reconstruct_pose(model: RVQTokenizer, pose: torch.Tensor, norm: dict[str, torch.Tensor], args: argparse.Namespace, ckpt_data: dict[str, Any]) -> torch.Tensor:
    model.eval()
    orig_len = pose.shape[0]
    norm_pose = normalize_pose(pose, norm)
    starts = make_starts(orig_len, args.window_size, args.stride)
    recon_sum = torch.zeros(max(orig_len, starts[-1] + args.window_size), 178, 3)
    counts = torch.zeros(recon_sum.shape[0], 1, 1)
    windows = [extract_window(norm_pose, start, args.window_size) for start in starts]
    codebooks = ckpt_data.get("posthoc_codebooks")
    if codebooks is not None:
        codebooks = codebooks.to(args.device)
    with torch.no_grad():
        for offset in range(0, len(windows), args.reconstruct_batch_size):
            batch = torch.stack(windows[offset : offset + args.reconstruct_batch_size], dim=0).to(args.device)
            pred, _, _ = model_decode_batch(model, batch, args, codebooks=codebooks)
            pred = pred.cpu()
            for local_idx, window in enumerate(pred):
                start = starts[offset + local_idx]
                recon_sum[start : start + args.window_size] += window
                counts[start : start + args.window_size] += 1
    recon = recon_sum / counts.clamp_min(1.0)
    recon = recon[:orig_len]
    return denormalize_pose(recon, norm)


def validate_prediction_only(pred: dict[str, torch.Tensor]) -> dict[str, Any]:
    errors = []
    lengths = []
    for sid, pose in pred.items():
        if not isinstance(pose, torch.Tensor) or pose.dtype != torch.float32 or pose.ndim != 3 or tuple(pose.shape[1:]) != (178, 3) or pose.shape[0] <= 0 or not torch.isfinite(pose).all():
            errors.append({"sample_id": sid, "shape": tuple(pose.shape) if hasattr(pose, "shape") else None, "dtype": str(getattr(pose, "dtype", None))})
        else:
            lengths.append(int(pose.shape[0]))
    return {
        "ok": not errors,
        "num_errors": len(errors),
        "errors": errors[:50],
        "num_samples": len(pred),
        "length_min": min(lengths) if lengths else None,
        "length_max": max(lengths) if lengths else None,
        "length_mean": sum(lengths) / max(len(lengths), 1),
    }


def reconstruct_prediction(args: argparse.Namespace, ckpt: Path, out_dir: Path) -> Path:
    source = args.bbest_pred
    if source is None:
        source = args.project_root / BBEST_REL
    if not source.exists():
        raise FileNotFoundError(f"B-best prediction not found: {source}")
    model, norm, ckpt_data = load_model_from_ckpt(ckpt, args)
    source_pred = load_torch(source)
    items = list(source_pred.items())
    if args.max_reconstruct_samples > 0:
        if not args.skip_eval:
            raise ValueError("--max-reconstruct-samples can only be used with --skip-eval")
        items = items[: args.max_reconstruct_samples]
    recon_pred = {}
    for sid, pose in items:
        recon_pred[sid] = reconstruct_pose(model, to_pose(pose), norm, args, ckpt_data)
    out = out_dir / "predictions" / "dev" / "rvq_tokenizer_reconstruct_bbest.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(recon_pred, out)
    validation = validate_prediction_only(recon_pred)
    write_json(out.with_suffix(".validation.json"), validation)
    if not validation["ok"]:
        raise RuntimeError(f"bad reconstructed prediction: {out}")
    return out


def run_evaluator(args: argparse.Namespace, pred_path: Path, out_dir: Path) -> dict[str, Any]:
    tag = pred_path.stem
    workspace = out_dir / "eval_workspace" / tag
    workspace.mkdir(parents=True, exist_ok=True)
    evaluator = args.project_root / "exps" / "task3" / "repos" / "SLRTP-Sign-Production-Evaluation" / "main.py"
    gt = args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / "dev.pt"
    bt = args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "backTranslation_PHIX_model"
    if args.eval_python:
        cmd = [str(args.eval_python), str(evaluator), str(pred_path.resolve()), str(gt.resolve()), str(bt.resolve()), "--tag", tag, "--fps", "25"]
    else:
        cmd = ["conda", "run", "-n", args.eval_env, "python", str(evaluator), str(pred_path.resolve()), str(gt.resolve()), str(bt.resolve()), "--tag", tag, "--fps", "25"]
    start = now_iso()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=workspace, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    end = now_iso()
    log = out_dir / "eval_results" / "dev" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8")
    src = workspace / "results" / f"{tag}.json"
    dst = out_dir / "eval_results" / "dev" / f"{tag}.json"
    result = {"cmd": " ".join(cmd), "start": start, "end": end, "elapsed_sec": time.time() - t0, "returncode": proc.returncode, "log": str(log), "result_json": str(dst)}
    if proc.returncode == 0 and src.exists():
        shutil.copy2(src, dst)
        result.update({"status": "ok", "metrics": read_json(dst)})
    else:
        result.update({"status": "failed", "error_tail": proc.stdout[-4000:]})
        write_json(dst, result)
    write_json(out_dir / "eval_results" / "dev" / f"{tag}.run.json", result)
    return result


def render_report(out_dir: Path, summary: dict[str, Any]) -> None:
    eval_result = summary.get("bbest_reconstruction_eval") or {}
    metrics = eval_result.get("metrics", {}) if eval_result else {}
    bleu = metrics.get("bleu", {}) if metrics else {}
    bleu4 = bleu.get("bleu4", "NA")
    status = eval_result.get("status", "skipped")
    gate = summary.get("gate", {})
    lines = [
        "# RVQ tokenizer gate 报告",
        "",
        "## 方法边界",
        "",
        "本实验只训练 learnable temporal RVQ tokenizer，验证离散 token 是否能保住 pose 信息；它不是完整 text/gloss -> pose 方法，也不是 winner/GUS 包装。",
        "",
        "## B-best 重建诊断",
        "",
        f"- status: {status}",
        f"- BLEU4: {bleu4}",
        f"- gate_threshold_bleu4: {gate.get('threshold_bleu4', 'NA')}",
        f"- pass_gate: {gate.get('pass_gate', 'NA')}",
        "",
        "## 决策",
        "",
        "只有 B-best prediction 经 tokenizer encode/decode 后仍保留足够 evaluator 信号，才进入 token prior。否则继续训练 text/gloss -> token 没有意义。",
    ]
    path = out_dir / "reports" / "rvq_tokenizer_gate_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out_dir / "reports" / "rvq_tokenizer_gate_summary.json", summary)


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = load_torch(args.project_root / "data" / "SLRTP-Sign-Production-Evaluation-Data" / "data" / "train.pt")
    ckpt = train_tokenizer(args, train, args.out_dir)
    pred = reconstruct_prediction(args, ckpt, args.out_dir)
    dev_eval = None if args.skip_eval else run_evaluator(args, pred, args.out_dir)
    bleu4 = None
    if dev_eval and dev_eval.get("status") == "ok":
        bleu4 = dev_eval.get("metrics", {}).get("bleu", {}).get("bleu4")
    gate = {"threshold_bleu4": args.gate_bleu4, "pass_gate": bool(bleu4 is not None and bleu4 >= args.gate_bleu4)}
    summary = {
        "created_at": now_iso(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint": str(ckpt),
        "reconstructed_prediction": str(pred),
        "bbest_reconstruction_eval": dev_eval,
        "gate": gate,
        "baseline": {"current_b_best_dev_bleu4": 10.877469287658991, "phase1_window_token_reconstruct_bbest_bleu4": 4.990574851828193},
    }
    render_report(args.out_dir, summary)


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--out-dir", type=Path, default=default_project / "g2p_ddm_token_exps" / "outputs" / "rvq_tokenizer_gate")
    parser.add_argument("--bbest-pred", type=Path, default=None)
    parser.add_argument("--eval-env", default="t2s-oracle")
    parser.add_argument("--eval-python", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--posthoc-rvq", action="store_true", help="train continuous AE first, then fit residual k-means codebooks on latents")
    parser.add_argument("--feature-mode", choices=["raw_pose", "coarse_velocity"], default="raw_pose")
    parser.add_argument("--pose-dct-components", type=int, default=2, help="low-frequency pose DCT components for coarse_velocity feature mode")
    parser.add_argument("--velocity-dct-components", type=int, default=2, help="low-frequency velocity DCT components for coarse_velocity feature mode")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--n-codes", type=int, default=512)
    parser.add_argument("--n-quantizers", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--vq-weight", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reconstruct-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.1)
    parser.add_argument("--velocity-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--train-seconds", type=int, default=14400)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=32)
    parser.add_argument("--max-train-windows", type=int, default=160000)
    parser.add_argument("--max-val-windows", type=int, default=12000)
    parser.add_argument("--max-codebook-windows", type=int, default=80000)
    parser.add_argument("--kmeans-iters", type=int, default=15)
    parser.add_argument("--kmeans-batch-size", type=int, default=4096)
    parser.add_argument("--max-reconstruct-samples", type=int, default=0)
    parser.add_argument("--gate-bleu4", type=float, default=8.5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
