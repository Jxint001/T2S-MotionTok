#!/usr/bin/env python3
"""Build the official main-test result table for T2S-MotionTok."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import METRIC_COLUMNS, as_float, ensure_out_dir, extract_metrics, metric_row, read_csv_rows, wants_format, write_csv, write_markdown

# Reference numbers copied from the PHOENIX14T SLRTP official challenge table / project record.
# Team 2 and Team 3 are kept as partial built-ins because only BLEU-4 and WER are part of
# the required claim set. Pass --reference-csv to provide full official rows.
BUILTIN_REFERENCES = [
    {
        "name": "Team 1 / retrieval winner",
        "bleu1": 34.85,
        "bleu2": 21.96,
        "bleu3": 15.65,
        "bleu4": 12.06,
        "chrf": 36.83,
        "rouge": 36.59,
        "wer": 93.49,
        "dtw_mje": 0.0448,
        "total_distance": 1.631,
        "avg_duration": 1.438,
    },
    {"name": "Team 2", "bleu4": 2.05, "wer": 147.85},
    {"name": "Team 3", "bleu4": 9.59, "wer": 88.88},
]


def load_reference_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(row) for row in BUILTIN_REFERENCES]
    rows: list[dict[str, Any]] = []
    for raw in read_csv_rows(path):
        name = raw.get("name") or raw.get("method") or raw.get("team")
        if not name:
            raise ValueError(f"reference row missing name/method/team in {path}")
        row: dict[str, Any] = {"name": name}
        for col in METRIC_COLUMNS:
            row[col] = as_float(raw.get(col))
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours-json", type=Path, required=True, help="Evaluator JSON for the final model on test.")
    parser.add_argument("--gt-self-json", type=Path, required=True, help="Evaluator JSON for GT self-evaluation on test.")
    parser.add_argument("--reference-csv", type=Path, default=None, help="Optional official reference CSV with name/method and metric columns.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evidence"))
    parser.add_argument("--format", default="csv,md", help="Comma-separated outputs: csv, md, or csv,md.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = [
        metric_row("GT self", extract_metrics(args.gt_self_json)),
        *load_reference_rows(args.reference_csv),
        metric_row("T2S-MotionTok", extract_metrics(args.ours_json)),
    ]
    columns = ["name", *METRIC_COLUMNS]
    out_dir = ensure_out_dir(args.out_dir)
    if wants_format(args.format, "csv"):
        write_csv(out_dir / "main_results.csv", rows, columns)
    if wants_format(args.format, "md"):
        write_markdown(out_dir / "main_results.md", rows, columns, "Official Main Test Evaluation")


if __name__ == "__main__":
    main()
