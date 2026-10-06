"""Mathematical privacy assessment using pyCANON.

This module evaluates both the unmodified quasi-identifier view and a declared
release view with age generalization. It does not claim legal HIPAA
certification; the generated certificate records the threat model and limits.
"""

from __future__ import annotations

import hashlib
import html
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
from pycanon import anonymity


@dataclass(frozen=True)
class PrivacyAssessment:
    """Result of the configured k-anonymity and l-diversity checks."""

    success: bool
    k: int
    minimum_l: int
    baseline_k: int
    baseline_minimum_l: int
    json_path: Path
    html_path: Path


def dataframe_fingerprint(dataframe: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 fingerprint for certificate traceability."""

    normalized = dataframe.copy()
    for column in normalized.columns:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(repr)
    row_hashes = pd.util.hash_pandas_object(normalized, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()


def _privacy_frame(
    dataframe: pd.DataFrame,
    privacy_config: Mapping[str, Any],
) -> pd.DataFrame:
    """Remove direct IDs and generalize exact age for the declared release view."""

    frame = dataframe.drop(
        columns=[
            column
            for column in privacy_config["direct_identifiers"]
            if column in dataframe.columns
        ]
    ).copy()
    frame["age_band"] = pd.cut(
        frame["age"],
        bins=privacy_config["age_bins"],
        labels=privacy_config["age_labels"],
        include_lowest=True,
    ).astype("string")
    if frame["age_band"].isna().any():
        bad_ages = sorted(frame.loc[frame["age_band"].isna(), "age"].unique().tolist())
        raise ValueError(f"Age generalization does not cover values: {bad_ages}")
    frame = frame.drop(columns=["age"])
    return frame


def _validate_columns(dataframe: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing {label} columns required for privacy assessment: {missing}")


def _measure(
    dataframe: pd.DataFrame,
    quasi_identifiers: list[str],
    sensitive_attributes: list[str],
) -> tuple[int, dict[str, int]]:
    """Calculate k and l per sensitive attribute with pyCANON."""

    _validate_columns(dataframe, quasi_identifiers, "quasi-identifier")
    _validate_columns(dataframe, sensitive_attributes, "sensitive attribute")
    working = dataframe[quasi_identifiers + sensitive_attributes].copy()
    for column in quasi_identifiers + sensitive_attributes:
        if working[column].isna().any():
            working[column] = working[column].astype("object").where(
                working[column].notna(), "__MISSING__"
            )
    k_value = int(anonymity.k_anonymity(working, quasi_identifiers))
    l_values = {
        sensitive: int(
            anonymity.l_diversity(working, quasi_identifiers, [sensitive])
        )
        for sensitive in sensitive_attributes
    }
    return k_value, l_values


def _certificate_html(certificate: Mapping[str, Any]) -> str:
    baseline = certificate["baseline"]
    release = certificate["generalized_release_view"]
    status_class = "pass" if certificate["overall_pass"] else "fail"
    status = "PASS" if certificate["overall_pass"] else "FAIL"
    l_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{value}</td>"
        f"<td>{'PASS' if value >= certificate['thresholds']['minimum_l'] else 'FAIL'}</td></tr>"
        for name, value in release["l_by_sensitive_attribute"].items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Clinical Privacy Assessment Certificate</title>
  <style>
    body {{ font-family: system-ui, Segoe UI, sans-serif; margin: 0; background: #f4f7fb; color: #172033; }}
    main {{ max-width: 980px; margin: 36px auto; background: white; padding: 36px; border-radius: 14px; box-shadow: 0 8px 32px #193b6a20; }}
    h1, h2 {{ color: #1768ac; }}
    .badge {{ display: inline-block; padding: 8px 16px; border-radius: 999px; font-weight: 700; }}
    .pass {{ background: #d9f7e5; color: #126b39; }} .fail {{ background: #fde1e1; color: #9a2020; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d9e1ec; padding: 10px; text-align: left; }} th {{ background: #edf4fb; }}
    code {{ overflow-wrap: anywhere; }} .notice {{ border-left: 4px solid #e5a000; padding: 12px 16px; background: #fff8df; }}
  </style>
</head>
<body><main>
  <h1>Clinical Privacy Assessment Certificate</h1>
  <p class="badge {status_class}">{status}: configured k-anonymity and l-diversity thresholds</p>
  <p><strong>Generated:</strong> {html.escape(certificate['generated_at_utc'])}</p>
  <p><strong>Rows assessed:</strong> {certificate['row_count']:,}</p>
  <p><strong>Dataset SHA-256:</strong> <code>{certificate['dataset_sha256']}</code></p>

  <h2>Threat model and transformation</h2>
  <table><tr><th>Item</th><th>Value</th></tr>
    <tr><td>Direct identifiers removed</td><td>{html.escape(', '.join(certificate['direct_identifiers_removed']))}</td></tr>
    <tr><td>Baseline QI</td><td>{html.escape(', '.join(baseline['quasi_identifiers']))}</td></tr>
    <tr><td>Release-view QI</td><td>{html.escape(', '.join(release['quasi_identifiers']))}</td></tr>
    <tr><td>Age generalization</td><td>{html.escape(', '.join(certificate['age_generalization']))}</td></tr>
  </table>

  <h2>Mathematical results</h2>
  <table><tr><th>View</th><th>k</th><th>Minimum l</th><th>Decision</th></tr>
    <tr><td>Baseline exact-age view</td><td>{baseline['k']}</td><td>{baseline['minimum_l']}</td><td>{'PASS' if baseline['pass'] else 'FAIL'}</td></tr>
    <tr><td>Generalized release view</td><td>{release['k']}</td><td>{release['minimum_l']}</td><td>{'PASS' if release['pass'] else 'FAIL'}</td></tr>
  </table>
  <table><tr><th>Sensitive attribute</th><th>l</th><th>Decision</th></tr>{l_rows}</table>

  <p class="notice"><strong>Scope notice:</strong> {html.escape(certificate['hipaa_scope_notice'])}</p>
</main></body></html>"""


def run_privacy_assessment(
    dataframe: pd.DataFrame,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> PrivacyAssessment:
    """Evaluate baseline and generalized views, then write audit certificates."""

    root = Path(project_root).resolve()
    privacy = config["privacy"]
    baseline_qi = list(privacy["baseline_quasi_identifiers"])
    release_qi = list(privacy["release_quasi_identifiers"])
    sensitive = list(privacy["sensitive_attributes"])
    minimum_k = int(privacy["minimum_k"])
    minimum_l = int(privacy["minimum_l"])

    direct_ids_present = [
        column for column in privacy["direct_identifiers"] if column in dataframe.columns
    ]
    baseline_k, baseline_l_values = _measure(dataframe, baseline_qi, sensitive)
    release_frame = _privacy_frame(dataframe, privacy)
    release_k, release_l_values = _measure(release_frame, release_qi, sensitive)

    baseline_minimum_l = min(baseline_l_values.values())
    release_minimum_l = min(release_l_values.values())
    baseline_pass = baseline_k >= minimum_k and baseline_minimum_l >= minimum_l
    release_pass = release_k >= minimum_k and release_minimum_l >= minimum_l

    certificate = {
        "certificate_type": "HIPAA-aligned mathematical privacy assessment",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(dataframe)),
        "dataset_sha256": dataframe_fingerprint(dataframe),
        "thresholds": {"minimum_k": minimum_k, "minimum_l": minimum_l},
        "direct_identifiers_removed": direct_ids_present,
        "age_generalization": list(privacy["age_labels"]),
        "baseline": {
            "quasi_identifiers": baseline_qi,
            "k": baseline_k,
            "l_by_sensitive_attribute": baseline_l_values,
            "minimum_l": baseline_minimum_l,
            "pass": baseline_pass,
        },
        "generalized_release_view": {
            "quasi_identifiers": release_qi,
            "sensitive_attributes": sensitive,
            "k": release_k,
            "l_by_sensitive_attribute": release_l_values,
            "minimum_l": release_minimum_l,
            "pass": release_pass,
        },
        "overall_pass": release_pass,
        "hipaa_scope_notice": (
            "Passing k-anonymity and l-diversity supports a de-identification risk "
            "assessment but is not, by itself, legal certification of HIPAA compliance. "
            "The assessment is limited to the declared quasi-identifiers and sensitive attributes."
        ),
    }

    json_path = (root / config["reports"]["privacy_json"]).resolve()
    html_path = (root / config["reports"]["privacy_html"]).resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(certificate, indent=2, ensure_ascii=False), encoding="utf-8")
    html_path.write_text(_certificate_html(certificate), encoding="utf-8")

    return PrivacyAssessment(
        success=release_pass,
        k=release_k,
        minimum_l=release_minimum_l,
        baseline_k=baseline_k,
        baseline_minimum_l=baseline_minimum_l,
        json_path=json_path,
        html_path=html_path,
    )
