#!/usr/bin/env python3
"""Build a dev duration-sweep table and optional plot."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import METRIC_COLUMNS, ensure_out_dir, extract_metrics, metric_row, wants_format, write_csv, write_markdown


def parse_run(value: str) -> tuple[float, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("expected SCALE:PATH_TO_JSON")
    scale, path = value.split(":", 1)
    try:
        scale_f = float(scale)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration scale: {scale}") from exc
    return scale_f, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="Duration sweep entry as SCALE:PATH_TO_EVAL_JSON. Repeat for multiple scales.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/evidence"))
    parser.add_argument("--format", default="csv,md", help="Comma-separated outputs: csv, md, or csv,md.")
    parser.add_argument("--plot", action="store_true", help="Also write duration_sweep_plot.png.")
    return parser.parse_args()


def write_plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    scales = [float(r["duration_scale"]) for r in rows]
    metrics = [
        ("BLEU-4", "bleu4"),
        ("WER", "wer"),
        ("DTW", "dtw_mje"),
        ("AvgDuration", "avg_duration"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), dpi=160)
    for ax, (title, key) in zip(axes.flat, metrics):
        values = [r.get(key) for r in rows]
        ax.plot(scales, values, marker="o", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("duration scale")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = []
    for scale, path in sorted(args.run, key=lambda x: x[0]):
        metrics = extract_metrics(path)
        rows.append(metric_row(f"duration {scale:g}", metrics, duration_scale=scale))
    columns = ["duration_scale", "name", *METRIC_COLUMNS]
    out_dir = ensure_out_dir(args.out_dir)
    if wants_format(args.format, "csv"):
        write_csv(out_dir / "duration_sweep.csv", rows, columns)
    if wants_format(args.format, "md"):
        write_markdown(out_dir / "duration_sweep.md", rows, columns, "Duration Calibration Sweep on Dev")
    if args.plot:
        write_plot(rows, out_dir / "duration_sweep_plot.png")


if __name__ == "__main__":
    main()
