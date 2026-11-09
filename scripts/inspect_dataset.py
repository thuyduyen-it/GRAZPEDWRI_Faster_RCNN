"""Comprehensive dataset inspection and audit CLI script (Phase 1).

Executes all 12 dataset health checks, validates patient-level split isolation,
verifies positive/negative class balance, profiles image bit depth, generates
visual sample overlays, and outputs structured reports.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.audit import DatasetAuditor
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and Inspect GRAZPEDWRI-DX Dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override dataset root path (or set GRAZPEDWRI_DATA_ROOT)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logger("InspectDatasetCLI")
    logger.info(f"Initializing DatasetAuditor with config: {args.config}")

    auditor = DatasetAuditor(
        config_path=args.config,
        data_root_override=args.data_root,
    )

    results = auditor.run_full_audit()
    auditor.save_audit_report(results)

    logger.info("------------------------------------------------------------")
    logger.info(f"FINAL AUDIT VERDICT: [{results['summary_status']}]")
    logger.info("------------------------------------------------------------")
    for check_name, info in results["checks"].items():
        logger.info(f"  * {check_name}: {info['status']}")

    return 0 if results["summary_status"] in ["PASS", "WARNING"] else 1


if __name__ == "__main__":
    sys.exit(main())
