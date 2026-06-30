#!/usr/bin/env python3
"""Shared helpers for T2S-MotionTok evidence scripts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

METRIC_COLUMNS = [
    "bleu1",
    "bleu2",
    "bleu3",
    "bleu4",
    "chrf",
    "rouge",
    "wer",
    "dtw_mje",
    "total_distance",
    "avg_duration",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metric_block(data: dict[str, Any]) -> dict[str, Any]:
    if "bleu" in data or any(k in data for k in METRIC_COLUMNS):
        return data
    if isinstance(data.get("metrics"), dict):
        return data["metrics"]
    for key in ("split_eval", "test_eval", "dev_eval"):
        block = data.get(key)
        if isinstance(block, dict) and isinstance(block.get("metrics"), dict):
            return block["metrics"]
    return data


def extract_metrics(path: Path) -> dict[str, float | None]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    block = _metric_block(data)
    bleu = block.get("bleu", {}) if isinstance(block.get("bleu"), dict) else {}
    out: dict[str, float | None] = {}
    for name in ("bleu1", "bleu2", "bleu3", "bleu4"):
        out[name] = as_float(block.get(name, bleu.get(name)))
    out["chrf"] = as_float(block.get("chrf"))
    out["rouge"] = as_float(block.get("rouge"))
    out["wer"] = as_float(block.get("wer"))
    out["dtw_mje"] = as_float(block.get("dtw_mje", block.get("dtw")))
    out["total_distance"] = as_float(block.get("total_distance", block.get("distance")))
    out["avg_duration"] = as_float(block.get("avg_duration", block.get("avg_dur")))
    return out


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def write_markdown(path: Path, rows: list[dict[str, Any]], columns: list[str], title: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def metric_row(name: str, metrics: dict[str, Any], **extra: Any) -> dict[str, Any]:
    row = {"name": name}
    row.update(extra)
    for col in METRIC_COLUMNS:
        row[col] = metrics.get(col)
    return row


def wants_format(format_value: str, name: str) -> bool:
    tokens = []
    for raw in format_value.split(','):
        raw = raw.strip().lower()
        if raw == 'both':
            tokens.extend(['csv', 'md'])
        elif raw:
            tokens.append(raw)
    allowed = {'csv', 'md'}
    unknown = sorted(set(tokens) - allowed)
    if unknown:
        raise ValueError(f"unknown output format(s): {', '.join(unknown)}")
    return name in tokens
