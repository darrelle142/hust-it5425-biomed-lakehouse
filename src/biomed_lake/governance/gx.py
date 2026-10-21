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
class StarSchemaTables:
    """The four LanceDB mart tables produced by Pipeline 01."""

    fact_clinical_cohort: pd.DataFrame
    dim_patient: pd.DataFrame
    dim_clinical_metrics: pd.DataFrame
    dim_aha_guidelines: pd.DataFrame
    source_path: Path


STAR_SCHEMA_TABLES = (
    "fact_clinical_cohort",
    "dim_patient",
    "dim_clinical_metrics",
    "dim_aha_guidelines",
)

BP_STAGES = [
    "NORMAL",
    "ELEVATED",
    "STAGE_1_HYPERTENSION",
    "STAGE_2_HYPERTENSION",
    "HYPERTENSIVE_CRISIS",
]
ICD10_CODES = ["R03.0", "I10", "I10.9", "I16.9"]


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
    suite_statistics: dict[str, dict[str, float | int]]
    star_schema_path: Path | None = None


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


def load_star_schema(
    project_root: str | Path,
    config: Mapping[str, Any],
) -> StarSchemaTables | None:
    """Load the LanceDB Star Schema mart, or ``None`` when it has not been built."""

    star_config = config.get("star_schema")
    if not star_config:
        return None
    uri = (Path(project_root) / star_config["lancedb_uri"]).resolve()
    if not uri.exists():
        return None

    import lancedb

    db = lancedb.connect(str(uri))
    available = set(db.table_names())
    missing = [name for name in STAR_SCHEMA_TABLES if name not in available]
    if missing:
        raise ValueError(f"LanceDB mart {uri} is missing Star Schema tables: {missing}")
    tables = {name: db.open_table(name).to_pandas() for name in STAR_SCHEMA_TABLES}
    return StarSchemaTables(**tables, source_path=uri)


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
            value_set=BP_STAGES,
            description="Blood-pressure stage must use the governed terminology.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="icd10_code",
            value_set=ICD10_CODES,
            description="ICD-10 field must use the codes emitted by Pipeline 01.",
        )
    )


def _add_primary_key(suite: ge.ExpectationSuite, table: str, column: str) -> None:
    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeNull(
            column=column,
            description=f"Primary key '{table}.{column}' must be 100% complete.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeUnique(
            column=column,
            description=f"Primary key '{table}.{column}' must be unique.",
        )
    )


def _add_foreign_key(
    suite: ge.ExpectationSuite,
    table: str,
    column: str,
    parent: pd.DataFrame,
    parent_table: str,
    parent_column: str,
) -> None:
    """Referential integrity: every key must exist in the parent dimension."""

    suite.add_expectation(
        gxe.ExpectColumnValuesToNotBeNull(
            column=column,
            description=f"Foreign key '{table}.{column}' must be 100% complete.",
        )
    )
    suite.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column=column,
            value_set=sorted(int(value) for value in parent[parent_column].dropna().unique()),
            description=(
                f"Referential integrity: '{table}.{column}' must reference an existing "
                f"'{parent_table}.{parent_column}'."
            ),
        )
    )


def _star_schema_suites(
    tables: StarSchemaTables,
    quality: Mapping[str, Any],
    star_config: Mapping[str, Any],
) -> list[tuple[str, ge.ExpectationSuite, pd.DataFrame, list[str]]]:
    """Build one contract per Star Schema table as (table, suite, data, index columns)."""

    expected_rows = int(quality["expected_rows"])
    vector_dim = int(star_config["phenotype_vector_dim"])

    fact = ge.ExpectationSuite(name="star_schema_fact_clinical_cohort")
    for column in ["event_id", "person_id", "metric_id", "guideline_id", "ten_year_chd", "phenotype_vector"]:
        fact.add_expectation(
            gxe.ExpectColumnToExist(
                column=column,
                description=f"Fact column '{column}' must exist.",
            )
        )
    fact.add_expectation(
        gxe.ExpectTableRowCountToEqual(
            value=expected_rows,
            description=f"Fact table must contain exactly {expected_rows:,} clinical events.",
        )
    )
    _add_primary_key(fact, "fact_clinical_cohort", "event_id")
    _add_foreign_key(fact, "fact_clinical_cohort", "person_id", tables.dim_patient, "dim_patient", "person_id")
    _add_foreign_key(
        fact, "fact_clinical_cohort", "metric_id",
        tables.dim_clinical_metrics, "dim_clinical_metrics", "metric_id",
    )
    _add_foreign_key(
        fact, "fact_clinical_cohort", "guideline_id",
        tables.dim_aha_guidelines, "dim_aha_guidelines", "guideline_id",
    )
    fact.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="ten_year_chd",
            value_set=[0, 1],
            description="Outcome measure 'ten_year_chd' accepts only 0 or 1.",
        )
    )
    # GX value-length metrics measure str(value), so validate a derived length column.
    fact_frame = tables.fact_clinical_cohort.assign(
        phenotype_vector_dim=tables.fact_clinical_cohort["phenotype_vector"].map(
            lambda vector: len(vector) if vector is not None else 0
        )
    )
    fact.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="phenotype_vector_dim",
            value_set=[vector_dim],
            description=f"Every phenotype vector must have exactly {vector_dim} dimensions for k-NN search.",
        )
    )

    patient = ge.ExpectationSuite(name="star_schema_dim_patient")
    patient.add_expectation(
        gxe.ExpectTableRowCountToEqual(
            value=expected_rows,
            description=f"Patient dimension must contain exactly {expected_rows:,} patients.",
        )
    )
    _add_primary_key(patient, "dim_patient", "person_id")

    metrics = ge.ExpectationSuite(name="star_schema_dim_clinical_metrics")
    metrics.add_expectation(
        gxe.ExpectTableRowCountToEqual(
            value=expected_rows,
            description=f"Clinical metrics dimension must contain exactly {expected_rows:,} records.",
        )
    )
    _add_primary_key(metrics, "dim_clinical_metrics", "metric_id")
    _add_foreign_key(
        metrics, "dim_clinical_metrics", "person_id",
        tables.dim_patient, "dim_patient", "person_id",
    )

    guidelines = ge.ExpectationSuite(name="star_schema_dim_aha_guidelines")
    guidelines.add_expectation(
        gxe.ExpectTableRowCountToEqual(
            value=len(BP_STAGES),
            description=f"AHA/JNC-7 lookup must contain exactly {len(BP_STAGES)} guideline levels.",
        )
    )
    _add_primary_key(guidelines, "dim_aha_guidelines", "guideline_id")
    guidelines.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="guideline_id",
            value_set=list(range(1, len(BP_STAGES) + 1)),
            description=f"Guideline levels must be numbered 1 to {len(BP_STAGES)}.",
        )
    )
    guidelines.add_expectation(
        gxe.ExpectColumnValuesToBeUnique(
            column="bp_stage",
            description="Each blood-pressure stage must map to exactly one guideline.",
        )
    )
    guidelines.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="bp_stage",
            value_set=BP_STAGES,
            description="Blood-pressure stage must use the governed terminology.",
        )
    )
    guidelines.add_expectation(
        gxe.ExpectColumnValuesToBeInSet(
            column="icd10_code",
            value_set=ICD10_CODES,
            description="ICD-10 code must belong to the governed hypertension code list.",
        )
    )
    for low, high in (("min_sys_bp", "max_sys_bp"), ("min_dia_bp", "max_dia_bp")):
        guidelines.add_expectation(
            gxe.ExpectColumnPairValuesAToBeGreaterThanB(
                column_A=high,
                column_B=low,
                or_equal=True,
                description=f"Guideline threshold '{high}' must be >= '{low}'.",
            )
        )

    return [
        ("fact_clinical_cohort", fact, fact_frame, ["event_id"]),
        ("dim_patient", patient, tables.dim_patient, ["person_id"]),
        ("dim_clinical_metrics", metrics, tables.dim_clinical_metrics, ["metric_id"]),
        ("dim_aha_guidelines", guidelines, tables.dim_aha_guidelines, ["guideline_id"]),
    ]


def _validation_results(checkpoint_results: list[Any]) -> list[Any]:
    """Flatten checkpoint results into ExpectationSuiteValidationResult objects."""

    results = []
    for checkpoint_result in checkpoint_results:
        for run_result in checkpoint_result.run_results.values():
            # GX 1.23 returns ExpectationSuiteValidationResult directly, while
            # some older releases wrap it in a ``validation_result`` mapping.
            results.append(
                run_result["validation_result"]
                if isinstance(run_result, Mapping) and "validation_result" in run_result
                else run_result
            )
    return results


def _extract_statistics(validation_results: list[Any]) -> dict[str, float | int]:
    """Extract GX validation statistics without depending on a JSON rendering."""

    totals = {
        "evaluated_expectations": 0,
        "successful_expectations": 0,
        "unsuccessful_expectations": 0,
        "success_percent": 0.0,
    }
    for validation_result in validation_results:
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


def _extract_failures(validation_results: list[Any]) -> list[dict[str, Any]]:
    """Create a compact, machine-readable list of failed expectations."""

    failures: list[dict[str, Any]] = []
    for validation_result in validation_results:
        for expectation_result in validation_result.results:
            if expectation_result.success:
                continue
            configuration = expectation_result.expectation_config
            result = expectation_result.result
            failures.append(
                {
                    "suite": validation_result.suite_name,
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
    star_schema: StarSchemaTables | None = None,
) -> QualityAssessment:
    """Run the GX contracts and build a static local Data Docs site.

    The flat cohort contract always runs. When ``star_schema`` is given, one
    extra contract per mart table checks keys and referential integrity. All
    suites share one context so they are published to the same Data Docs site.
    """

    root = Path(project_root).resolve()
    data_docs_dir = (root / config["reports"]["data_docs_dir"]).resolve()
    data_docs_dir.mkdir(parents=True, exist_ok=True)

    context = ge.get_context(mode="ephemeral")
    datasource = context.data_sources.add_pandas(name="clinical_governance_source")

    suite = ge.ExpectationSuite(
        name="biomedical_data_contract",
        notes=(
            "Automated contract for physiological validity, completeness, identifiers, "
            "and controlled clinical vocabularies."
        ),
    )
    _add_expectations(suite, config["quality"])
    runs = [("clinical_cohort", suite, dataset.dataframe, ["person_id"])]
    if star_schema is not None:
        runs.extend(_star_schema_suites(star_schema, config["quality"], config["star_schema"]))

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

    # Each table needs its own dataframe, so every suite gets its own checkpoint.
    checkpoint_results = []
    for asset_name, run_suite, dataframe, index_columns in runs:
        asset = datasource.add_dataframe_asset(name=asset_name)
        batch_definition = asset.add_batch_definition_whole_dataframe(name=f"full_{asset_name}")
        validation_definition = context.validation_definitions.add(
            ge.ValidationDefinition(
                name=f"{asset_name}_validation",
                data=batch_definition,
                suite=context.suites.add(run_suite),
            )
        )
        checkpoint = context.checkpoints.add(
            ge.Checkpoint(
                name=f"{asset_name}_checkpoint",
                validation_definitions=[validation_definition],
                actions=[
                    UpdateDataDocsAction(name="update_clinical_data_docs", site_names=[site_name])
                ],
                result_format={
                    "result_format": "COMPLETE",
                    "unexpected_index_column_names": index_columns,
                    "include_unexpected_rows": True,
                },
            )
        )
        checkpoint_results.append(checkpoint.run(batch_parameters={"dataframe": dataframe}))
    context.build_data_docs(site_names=[site_name])

    index_path = data_docs_dir / "index.html"
    if not index_path.exists():
        raise RuntimeError(f"GX did not generate the expected Data Docs index: {index_path}")

    validation_results = _validation_results(checkpoint_results)
    success = all(result.success for result in checkpoint_results)
    stats = _extract_statistics(validation_results)
    suite_statistics = {
        result.suite_name: {"success": bool(result.success), **_extract_statistics([result])}
        for result in validation_results
    }
    failures = _extract_failures(validation_results)
    star_schema_path = star_schema.source_path if star_schema is not None else None
    summary_path = root / config["reports"]["quality_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "success": success,
        **stats,
        "source_kind": dataset.source_kind,
        "source_path": str(dataset.source_path),
        "row_count": int(len(dataset.dataframe)),
        "star_schema_path": str(star_schema_path) if star_schema_path else None,
        "suites": suite_statistics,
        "failed_expectations": failures,
        "data_docs_index": str(index_path),
        "note": "A failed contract is a valid audit finding; inspect Data Docs for details.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return QualityAssessment(
        success=success,
        evaluated_expectations=int(stats["evaluated_expectations"]),
        successful_expectations=int(stats["successful_expectations"]),
        unsuccessful_expectations=int(stats["unsuccessful_expectations"]),
        success_percent=float(stats["success_percent"]),
        data_docs_index=index_path,
        source_kind=dataset.source_kind,
        source_path=dataset.source_path,
        failed_expectations=tuple(failures),
        suite_statistics=suite_statistics,
        star_schema_path=star_schema_path,
    )
