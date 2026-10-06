"""Great Expectations data contract for the clinical cohort.

The module deliberately keeps data loading and contract construction in one
place so that Delta Lake and CSV inputs are validated by the same rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import great_expectations as ge
import pandas as pd
import yaml
from deltalake import DeltaTable
from great_expectations import expectations as gxe
from great_expectations.checkpoint.actions import UpdateDataDocsAction


@dataclass(frozen=True)
class ClinicalDataset:
    """A loaded cohort and the source used to obtain it."""

    dataframe: pd.DataFrame
    source_kind: str
    source_path: Path


@dataclass(frozen=True)
class QualityAssessment:
    """Compact result returned to the governance pipeline."""

    success: bool
    evaluated_expectations: int
    successful_expectations: int
    unsuccessful_expectations: int
    success_percent: float
    data_docs_index: Path
    source_kind: str
    source_path: Path
    failed_expectations: tuple[dict[str, Any], ...]


def load_governance_config(config_path: str | Path) -> dict[str, Any]:
    """Load and minimally validate the governance YAML configuration."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Governance config does not exist: {path}")
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for section in ("data", "quality", "privacy", "reports"):
        if section not in config:
            raise ValueError(f"Missing '{section}' section in {path}")
    return config


def load_clinical_data(
    project_root: str | Path,
    config: Mapping[str, Any],
    preferred_source: str = "auto",
) -> ClinicalDataset:
    """Load Delta Lake first and fall back to the exported mart CSV.

    ``preferred_source`` can be ``auto``, ``delta`` or ``csv``. An explicit
    source never silently falls back, which makes command-line failures easy
    to diagnose.
    """

    if preferred_source not in {"auto", "delta", "csv"}:
        raise ValueError("preferred_source must be one of: auto, delta, csv")

    root = Path(project_root).resolve()
    delta_path = root / config["data"]["delta_path"]
    csv_path = root / config["data"]["csv_path"]

    candidates = (
        [("delta", delta_path), ("csv", csv_path)]
        if preferred_source == "auto"
        else [(preferred_source, delta_path if preferred_source == "delta" else csv_path)]
    )
    attempted: list[str] = []
    for source_kind, source_path in candidates:
        attempted.append(str(source_path))
        if not source_path.exists():
            continue
        if source_kind == "delta":
            dataframe = DeltaTable(str(source_path)).to_pyarrow_table().to_pandas()
        else:
            dataframe = pd.read_csv(source_path)
        if dataframe.empty:
            raise ValueError(f"Clinical dataset is empty: {source_path}")
        return ClinicalDataset(dataframe, source_kind, source_path.resolve())

    joined = "\n  - ".join(attempted)
    raise FileNotFoundError(f"No clinical input was found. Attempted:\n  - {joined}")


def _add_expectations(suite: ge.ExpectationSuite, quality: Mapping[str, Any]) -> None:
    """Populate the biomedical data contract."""

    expected_rows = int(quality["expected_rows"])
    completeness = float(quality["minimum_completeness"])
    required_columns = list(quality["required_columns"])

    suite.add_expectation(
        gxe.ExpectTableRowCountToEqual(
            value=expected_rows,
            description=f"Clinical cohort must contain exactly {expected_rows:,} patients.",
        )
    )

    for column in required_columns:
        suite.add_expectation(
            gxe.ExpectColumnToExist(
                column=column,
                description=f"Required contract column '{column}' must exist.",
            )
        )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeUnique(
            column="person_id",
            description="Each patient surrogate identifier must be unique.",
        )
    )

    always_present = ["person_id", "age", "is_male", "sys_bp", "dia_bp", "center_source"]
    for column in always_present:
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                description=f"Critical column '{column}' must be 100% complete.",
            )
        )

    for column in required_columns:
        if column in always_present:
            continue
        suite.add_expectation(
            gxe.ExpectColumnValuesToNotBeNull(
                column=column,
                mostly=completeness,
                description=(
                    f"Column '{column}' must be at least {completeness:.0%} complete "
                    "(missingness <= 5%)."
                ),
            )
        )

    for column, limits in quality["physiological_ranges"].items():
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeBetween(
                column=column,
                min_value=limits["min"],
                max_value=limits["max"],
                strict_min=bool(limits.get("strict_min", False)),
                strict_max=bool(limits.get("strict_max", False)),
                description=(
                    f"Physiological range contract for '{column}': "
                    f"{limits['min']} to {limits['max']}."
                ),
            )
        )

    suite.add_expectation(
        gxe.ExpectColumnPairValuesAToBeGreaterThanB(
            column_A="sys_bp",
            column_B="dia_bp",
            or_equal=True,
            description="Systolic blood pressure must be greater than or equal to diastolic pressure.",
        )
    )

    binary_columns = [
        "is_male",
        "current_smoker",
        "bp_meds",
        "prevalent_stroke",
        "prevalent_hyp",
        "diabetes",
        "ten_year_chd",
    ]
    for column in binary_columns:
        suite.add_expectation(
            gxe.ExpectColumnValuesToBeInSet(
                column=column,
                value_set=[0, 1],
                description=f"Binary clinical field '{column}' accepts only 0 or 1.",
            )
        )

    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="center_source",
            value_set=["FRAMINGHAM_US", "CARDIO_CLINICAL_INTL"],
            description="Every row must retain an approved source-cohort label.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="bp_stage",
            value_set=[
                "NORMAL",
                "ELEVATED",
                "STAGE_1_HYPERTENSION",
                "STAGE_2_HYPERTENSION",
                "HYPERTENSIVE_CRISIS",
            ],
            description="Blood-pressure stage must use the governed terminology.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="icd10_code",
            value_set=["R03.0", "I10", "I10.9", "I16.9"],
            description="ICD-10 field must use the codes emitted by Pipeline 01.",
        )
    )


def _extract_statistics(checkpoint_result: Any) -> dict[str, float | int]:
    """Extract GX validation statistics without depending on a JSON rendering."""

    totals = {
        "evaluated_expectations": 0,
        "successful_expectations": 0,
        "unsuccessful_expectations": 0,
        "success_percent": 0.0,
    }
    for run_result in checkpoint_result.run_results.values():
        # GX 1.23 returns ExpectationSuiteValidationResult directly, while
        # some older releases wrap it in a ``validation_result`` mapping.
        validation_result = (
            run_result["validation_result"]
            if isinstance(run_result, Mapping) and "validation_result" in run_result
            else run_result
        )
        statistics = validation_result.statistics
        for key in (
            "evaluated_expectations",
            "successful_expectations",
            "unsuccessful_expectations",
        ):
            totals[key] += int(statistics.get(key, 0))
    if totals["evaluated_expectations"]:
        totals["success_percent"] = round(
            100.0 * totals["successful_expectations"] / totals["evaluated_expectations"], 2
        )
    return totals


def _extract_failures(checkpoint_result: Any) -> list[dict[str, Any]]:
    """Create a compact, machine-readable list of failed expectations."""

    failures: list[dict[str, Any]] = []
    for run_result in checkpoint_result.run_results.values():
        validation_result = (
            run_result["validation_result"]
            if isinstance(run_result, Mapping) and "validation_result" in run_result
            else run_result
        )
        for expectation_result in validation_result.results:
            if expectation_result.success:
                continue
            configuration = expectation_result.expectation_config
            result = expectation_result.result
            failures.append(
                {
                    "expectation_type": configuration.type,
                    "column": configuration.kwargs.get("column"),
                    "column_A": configuration.kwargs.get("column_A"),
                    "column_B": configuration.kwargs.get("column_B"),
                    "unexpected_count": int(result.get("unexpected_count", 0)),
                    "unexpected_percent": round(
                        float(result.get("unexpected_percent", 0.0)), 4
                    ),
                    "element_count": int(result.get("element_count", 0)),
                }
            )
    return failures


def run_quality_assessment(
    dataset: ClinicalDataset,
    config: Mapping[str, Any],
    project_root: str | Path,
) -> QualityAssessment:
    """Run the GX contract and build a static local Data Docs site."""

    root = Path(project_root).resolve()
    data_docs_dir = (root / config["reports"]["data_docs_dir"]).resolve()
    data_docs_dir.mkdir(parents=True, exist_ok=True)

    context = ge.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="clinical_governance_source")
    asset = datasource.add_dataframe_asset(name="clinical_cohort")
    batch_definition = asset.add_batch_definition_whole_dataframe(name="full_cohort")

    suite = ge.ExpectationSuite(
        name="biomedical_data_contract",
        notes=(
            "Automated contract for physiological validity, completeness, identifiers, "
            "and controlled clinical vocabularies."
        ),
    )
    _add_expectations(suite, config["quality"])
    suite = context.suites.add(suite)

    validation_definition = ge.ValidationDefinition(
        name="clinical_cohort_validation",
        data=batch_definition,
        suite=suite,
    )
    validation_definition = context.validation_definitions.add(validation_definition)

    site_name = "clinical_data_docs"
    context.add_data_docs_site(
        site_name=site_name,
        site_config={
            "class_name": "SiteBuilder",
            "site_index_builder": {"class_name": "DefaultSiteIndexBuilder"},
            "store_backend": {
                "class_name": "TupleFilesystemStoreBackend",
                "base_directory": str(data_docs_dir),
            },
        },
    )
    checkpoint = ge.Checkpoint(
        name="clinical_governance_checkpoint",
        validation_definitions=[validation_definition],
        actions=[
            UpdateDataDocsAction(name="update_clinical_data_docs", site_names=[site_name])
        ],
        result_format={
            "result_format": "COMPLETE",
            "unexpected_index_column_names": ["person_id"],
            "include_unexpected_rows": True,
        },
    )
    checkpoint = context.checkpoints.add(checkpoint)
    checkpoint_result = checkpoint.run(batch_parameters={"dataframe": dataset.dataframe})
    context.build_data_docs(site_names=[site_name])

    index_path = data_docs_dir / "index.html"
    if not index_path.exists():
        raise RuntimeError(f"GX did not generate the expected Data Docs index: {index_path}")

    stats = _extract_statistics(checkpoint_result)
    failures = _extract_failures(checkpoint_result)
    summary_path = root / config["reports"]["quality_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "success": bool(checkpoint_result.success),
        **stats,
        "source_kind": dataset.source_kind,
        "source_path": str(dataset.source_path),
        "row_count": int(len(dataset.dataframe)),
        "failed_expectations": failures,
        "data_docs_index": str(index_path),
        "note": "A failed contract is a valid audit finding; inspect Data Docs for details.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return QualityAssessment(
        success=bool(checkpoint_result.success),
        evaluated_expectations=int(stats["evaluated_expectations"]),
        successful_expectations=int(stats["successful_expectations"]),
        unsuccessful_expectations=int(stats["unsuccessful_expectations"]),
        success_percent=float(stats["success_percent"]),
        data_docs_index=index_path,
        source_kind=dataset.source_kind,
        source_path=dataset.source_path,
        failed_expectations=tuple(failures),
    )
