#!/usr/bin/env python3
"""Build the test-split component ablation table."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import METRIC_COLUMNS, ensure_out_dir, extract_metrics, metric_row, wants_format, write_csv, write_markdown

VARIANTS = [
    ("full", "Full", "full_json"),
    ("wo_alignment", "w/o alignment features", "wo_alignment_json"),
    ("wo_duration", "w/o duration scaling", "wo_duration_json"),
    ("wo_detail", "w/o detail layer", "wo_detail_json"),
    ("wo_beam", "w/o beam search", "wo_beam_json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-json", type=Path, required=True)
    parser.add_argument("--wo-alignment-json", type=Path, required=True)
    parser.add_argument("--wo-duration-json", type=Path, required=True)
    parser.add_argument("--wo-detail-json", type=Path, required=True)
    parser.add_argument("--wo-beam-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evidence"))
    parser.add_argument("--format", default="csv,md", help="Comma-separated outputs: csv, md, or csv,md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    full_metrics = extract_metrics(args.full_json)
    full_bleu4 = full_metrics.get("bleu4")
    rows = []
    for key, name, attr in VARIANTS:
        metrics = extract_metrics(getattr(args, attr))
        delta = None
        if full_bleu4 is not None and metrics.get("bleu4") is not None:
            delta = metrics["bleu4"] - full_bleu4
        rows.append(metric_row(name, metrics, key=key, delta_bleu4_vs_full=delta))
    columns = ["key", "name", "delta_bleu4_vs_full", *METRIC_COLUMNS]
    out_dir = ensure_out_dir(args.out_dir)
    if wants_format(args.format, "csv"):
        write_csv(out_dir / "test_ablation.csv", rows, columns)
    if wants_format(args.format, "md"):
        write_markdown(out_dir / "test_ablation.md", rows, columns, "Component Ablation on Test")


if __name__ == "__main__":
    main()
