"""Pipeline 02: Biomedical data quality and privacy governance."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biomed_lake.governance.anon import run_privacy_assessment
from biomed_lake.governance.gx import (
    load_clinical_data,
    load_governance_config,
    run_quality_assessment,
)


def _configure_stdout() -> None:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the clinical data contract and assess k/l privacy guarantees."
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "governance.yaml"),
        help="Path to governance YAML configuration.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "delta", "csv"],
        default="auto",
        help="Input source. 'auto' prefers Delta Lake and falls back to CSV.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit code 1 when either quality or privacy thresholds fail.",
    )
    return parser


def run_pipeline(config_path: str, source: str = "auto", strict: bool = False) -> int:
    _configure_stdout()
    started = time.perf_counter()
    print("=" * 80)
    print("PIPELINE 02: CLINICAL DATA GOVERNANCE & PRIVACY ASSESSMENT")
    print("=" * 80)

    config = load_governance_config(config_path)
    dataset = load_clinical_data(PROJECT_ROOT, config, preferred_source=source)
    print(f"[INPUT] {dataset.source_kind.upper()}: {dataset.source_path}")
    print(f"[INPUT] {len(dataset.dataframe):,} rows x {len(dataset.dataframe.columns)} columns")

    print("\n[1/2] Running Great Expectations biomedical data contract...")
    quality = run_quality_assessment(dataset, config, PROJECT_ROOT)
    quality_status = "PASS" if quality.success else "FAIL"
    print(
        f"[{quality_status}] Quality: {quality.successful_expectations}/"
        f"{quality.evaluated_expectations} expectations passed "
        f"({quality.success_percent:.2f}%)."
    )
    print(f"[REPORT] Data Docs: {quality.data_docs_index}")
    for finding in quality.failed_expectations:
        columns = finding["column"] or (
            f"{finding['column_A']} >= {finding['column_B']}"
        )
        print(
            f"  - {finding['expectation_type']} [{columns}]: "
            f"{finding['unexpected_count']} unexpected values "
            f"({finding['unexpected_percent']:.3f}%)"
        )

    print("\n[2/2] Running pyCANON k-anonymity and l-diversity assessment...")
    privacy = run_privacy_assessment(dataset.dataframe, config, PROJECT_ROOT)
    privacy_status = "PASS" if privacy.success else "FAIL"
    print(
        f"[BASELINE] Exact-age view: k={privacy.baseline_k}, "
        f"minimum l={privacy.baseline_minimum_l}"
    )
    print(
        f"[{privacy_status}] Generalized release view: k={privacy.k}, "
        f"minimum l={privacy.minimum_l}"
    )
    print(f"[REPORT] Privacy certificate: {privacy.html_path}")
    print(f"[REPORT] Machine-readable evidence: {privacy.json_path}")

    elapsed = time.perf_counter() - started
    overall_success = quality.success and privacy.success
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED in {elapsed:.2f}s")
    print(f"Overall governance gate: {'PASS' if overall_success else 'FAIL'}")
    if not quality.success:
        print("Quality findings are preserved in Data Docs; pipeline execution itself succeeded.")
    print("=" * 80)
    return 1 if strict and not overall_success else 0


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    raise SystemExit(run_pipeline(arguments.config, arguments.source, arguments.strict))
