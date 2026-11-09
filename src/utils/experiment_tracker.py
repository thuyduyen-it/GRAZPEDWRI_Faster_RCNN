"""Experiment Tracking & Benchmark Summary Manager (Phase 6).

Maintains outputs/metrics/experiments_summary.csv, records quantitative results across
ablation studies, and produces formatted tables for research reports and academic publications.
"""

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """Manages experiment summary metrics logging and comparative reporting."""

    HEADERS = [
        "exp_id",
        "variant",
        "architecture",
        "parameters_m",
        "image_resolution",
        "augmentation",
        "optimizer",
        "val_map_50",
        "val_map_50_95",
        "val_f1",
        "val_precision",
        "val_recall",
        "latency_ms",
        "status",
    ]

    def __init__(self, csv_path: str = "outputs/metrics/experiments_summary.csv"):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV file with headers if it doesn't already exist."""
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADERS)
                writer.writeheader()

    def log_result(self, record: Dict[str, Any]) -> None:
        """Log or update an experiment variant record in the CSV summary table.

        Args:
            record: Dictionary containing fields matching HEADERS.
        """
        rows = self.get_summary()
        exp_id = record.get("exp_id")
        variant = record.get("variant", "default")

        # Check if record already exists and update, otherwise append
        updated = False
        new_rows = []
        for r in rows:
            if r.get("exp_id") == exp_id and r.get("variant") == variant:
                new_rows.append({k: record.get(k, r.get(k, "")) for k in self.HEADERS})
                updated = True
            else:
                new_rows.append(r)

        if not updated:
            new_rows.append({k: record.get(k, "") for k in self.HEADERS})

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADERS)
            writer.writeheader()
            for row in new_rows:
                writer.writerow(row)

    def get_summary(self) -> List[Dict[str, Any]]:
        """Read all experiment summary rows."""
        if not self.csv_path.exists():
            return []

        rows = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
        return rows

    def format_markdown_table(self) -> str:
        """Generate a formatted GitHub markdown table of all recorded experiments."""
        rows = self.get_summary()
        if not rows:
            return "*(No experiments recorded yet)*"

        lines = [
            "| Exp ID | Variant | Architecture | Params (M) | Image Res | Augmentation | Optimizer | Val mAP@0.5 | Val mAP@0.5:0.95 | Val F1 | Latency (ms) | Status |",
            "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---|",
        ]
        for r in rows:
            lines.append(
                f"| **{r.get('exp_id')}** "
                f"| {r.get('variant')} "
                f"| {r.get('architecture')} "
                f"| {r.get('parameters_m')} "
                f"| {r.get('image_resolution')} "
                f"| {r.get('augmentation')} "
                f"| {r.get('optimizer')} "
                f"| {r.get('val_map_50')} "
                f"| {r.get('val_map_50_95')} "
                f"| {r.get('val_f1')} "
                f"| {r.get('latency_ms')} "
                f"| {r.get('status')} |"
            )
        return "\n".join(lines)
