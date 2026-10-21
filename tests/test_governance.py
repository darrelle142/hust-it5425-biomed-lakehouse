"""Tests for the Clinical Data Governance and Privacy subsystem."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biomed_lake.governance.anon import run_privacy_assessment
from biomed_lake.governance.gx import (
    BP_STAGES,
    ClinicalDataset,
    StarSchemaTables,
    run_quality_assessment,
)


def _sample_dataframe(rows: int = 12) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "person_id": index + 1,
                "is_male": index % 2,
                "age": 45 + (index % 2),
                "education": 2,
                "current_smoker": index % 2,
                "cigs_per_day": 10 if index % 2 else 0,
                "bp_meds": 0,
                "prevalent_stroke": 0,
                "prevalent_hyp": 0,
                "diabetes": 0,
                "tot_chol": 210.0,
                "sys_bp": 125.0,
                "dia_bp": 80.0,
                "BMI": 24.0,
                "heart_rate": 72.0,
                "glucose": 90.0,
                "ten_year_chd": index % 2,
                "center_source": "FRAMINGHAM_US",
                "bp_stage": "STAGE_1_HYPERTENSION",
                "icd10_code": "I10",
            }
        )
    return pd.DataFrame(records)


def _config(tmp_path: Path, rows: int) -> dict:
    return {
        "quality": {
            "expected_rows": rows,
            "minimum_completeness": 0.95,
            "required_columns": list(_sample_dataframe(1).columns),
            "physiological_ranges": {
                "age": {"min": 18, "max": 100},
                "sys_bp": {"min": 0, "max": 300, "strict_min": True},
                "dia_bp": {"min": 0, "max": 200, "strict_min": True},
                "BMI": {"min": 10, "max": 80},
                "heart_rate": {"min": 20, "max": 250},
                "glucose": {"min": 20, "max": 600},
                "tot_chol": {"min": 50, "max": 700},
                "cigs_per_day": {"min": 0, "max": 100},
            },
        },
        "star_schema": {"lancedb_uri": "unused", "phenotype_vector_dim": 8},
        "privacy": {
            "direct_identifiers": ["person_id"],
            "baseline_quasi_identifiers": ["age", "is_male", "center_source"],
            "release_quasi_identifiers": ["age_band", "is_male", "center_source"],
            "sensitive_attributes": ["ten_year_chd"],
            "age_bins": [0, 39, 49, 59, 200],
            "age_labels": ["<40", "40-49", "50-59", "60+"],
            "minimum_k": 5,
            "minimum_l": 2,
        },
        "reports": {
            "data_docs_dir": str(tmp_path / "data_docs"),
            "quality_summary": str(tmp_path / "quality.json"),
            "privacy_json": str(tmp_path / "privacy.json"),
            "privacy_html": str(tmp_path / "privacy.html"),
        },
    }


def test_quality_contract_builds_data_docs(tmp_path: Path) -> None:
    dataframe = _sample_dataframe()
    config = _config(tmp_path, len(dataframe))
    assessment = run_quality_assessment(
        ClinicalDataset(dataframe, "test", tmp_path / "sample.csv"), config, PROJECT_ROOT
    )
    assert assessment.success
    assert assessment.unsuccessful_expectations == 0
    assert assessment.data_docs_index.exists()


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [("sys_bp", 301.0), ("sys_bp", 0.0), ("BMI", 81.0), ("BMI", 9.0)],
)
def test_quality_contract_rejects_physiological_outlier(
    tmp_path: Path, column: str, invalid_value: float
) -> None:
    dataframe = _sample_dataframe()
    dataframe.loc[0, column] = invalid_value
    config = _config(tmp_path, len(dataframe))
    assessment = run_quality_assessment(
        ClinicalDataset(dataframe, "test", tmp_path / "sample.csv"), config, PROJECT_ROOT
    )
    assert not assessment.success
    assert assessment.unsuccessful_expectations >= 1


def _sample_star_schema(dataframe: pd.DataFrame, tmp_path: Path) -> StarSchemaTables:
    fact = pd.DataFrame(
        {
            "event_id": range(1, len(dataframe) + 1),
            "person_id": dataframe["person_id"],
            "metric_id": dataframe["person_id"],
            "guideline_id": 3,
            "ten_year_chd": dataframe["ten_year_chd"],
            "phenotype_vector": [[0.5] * 8 for _ in range(len(dataframe))],
        }
    )
    metrics = dataframe[["person_id", "tot_chol", "sys_bp", "dia_bp"]].copy()
    metrics.insert(0, "metric_id", dataframe["person_id"])
    guidelines = pd.DataFrame(
        {
            "guideline_id": range(1, len(BP_STAGES) + 1),
            "bp_stage": BP_STAGES,
            "min_sys_bp": [0.0, 120.0, 130.0, 140.0, 180.0],
            "max_sys_bp": [119.9, 129.9, 139.9, 179.9, 300.0],
            "min_dia_bp": [0.0, 0.0, 80.0, 90.0, 120.0],
            "max_dia_bp": [79.9, 79.9, 89.9, 119.9, 200.0],
            "icd10_code": ["R03.0", "R03.0", "I10", "I10.9", "I16.9"],
        }
    )
    return StarSchemaTables(
        fact_clinical_cohort=fact,
        dim_patient=dataframe[["person_id", "age", "is_male", "center_source"]].copy(),
        dim_clinical_metrics=metrics,
        dim_aha_guidelines=guidelines,
        source_path=tmp_path / "clin.lance",
    )


def test_star_schema_contract_passes(tmp_path: Path) -> None:
    dataframe = _sample_dataframe()
    config = _config(tmp_path, len(dataframe))
    assessment = run_quality_assessment(
        ClinicalDataset(dataframe, "test", tmp_path / "sample.csv"),
        config,
        PROJECT_ROOT,
        _sample_star_schema(dataframe, tmp_path),
    )
    assert assessment.success
    assert len(assessment.suite_statistics) == 5
    assert all(stats["success"] for stats in assessment.suite_statistics.values())


@pytest.mark.parametrize(
    ("table", "column", "invalid_value", "failed_suite"),
    [
        ("fact_clinical_cohort", "person_id", 999, "star_schema_fact_clinical_cohort"),
        ("fact_clinical_cohort", "metric_id", 999, "star_schema_fact_clinical_cohort"),
        ("fact_clinical_cohort", "guideline_id", 6, "star_schema_fact_clinical_cohort"),
        ("fact_clinical_cohort", "event_id", 2, "star_schema_fact_clinical_cohort"),
        ("dim_patient", "person_id", 2, "star_schema_dim_patient"),
        ("dim_clinical_metrics", "person_id", 999, "star_schema_dim_clinical_metrics"),
    ],
)
def test_star_schema_contract_rejects_broken_keys(
    tmp_path: Path, table: str, column: str, invalid_value: int, failed_suite: str
) -> None:
    dataframe = _sample_dataframe()
    config = _config(tmp_path, len(dataframe))
    star_schema = _sample_star_schema(dataframe, tmp_path)
    getattr(star_schema, table).loc[0, column] = invalid_value
    assessment = run_quality_assessment(
        ClinicalDataset(dataframe, "test", tmp_path / "sample.csv"),
        config,
        PROJECT_ROOT,
        star_schema,
    )
    assert not assessment.success
    assert not assessment.suite_statistics[failed_suite]["success"]
    assert any(finding["suite"] == failed_suite for finding in assessment.failed_expectations)


def test_star_schema_contract_rejects_wrong_vector_dimension(tmp_path: Path) -> None:
    dataframe = _sample_dataframe()
    config = _config(tmp_path, len(dataframe))
    star_schema = _sample_star_schema(dataframe, tmp_path)
    star_schema.fact_clinical_cohort.at[0, "phenotype_vector"] = [0.5] * 7
    assessment = run_quality_assessment(
        ClinicalDataset(dataframe, "test", tmp_path / "sample.csv"),
        config,
        PROJECT_ROOT,
        star_schema,
    )
    assert not assessment.suite_statistics["star_schema_fact_clinical_cohort"]["success"]


def test_privacy_generalization_meets_k_and_l(tmp_path: Path) -> None:
    dataframe = _sample_dataframe(12)
    # Both sensitive values occur in each (age band, sex, center) class.
    dataframe["ten_year_chd"] = [0, 1, 1, 0] * 3
    config = _config(tmp_path, len(dataframe))
    assessment = run_privacy_assessment(dataframe, config, PROJECT_ROOT)
    assert assessment.k == 6
    assert assessment.minimum_l == 2
    assert assessment.success
    assert assessment.html_path.exists()
    assert assessment.json_path.exists()


def test_privacy_fails_when_sensitive_value_is_homogeneous(tmp_path: Path) -> None:
    dataframe = _sample_dataframe(12)
    dataframe["ten_year_chd"] = 0
    config = _config(tmp_path, len(dataframe))
    assessment = run_privacy_assessment(dataframe, config, PROJECT_ROOT)
    assert assessment.minimum_l == 1
    assert not assessment.success
