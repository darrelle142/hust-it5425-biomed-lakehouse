"""
Unit Tests cho Phân hệ 1: Clinical Data Lakehouse & LanceDB
Phụ trách: TV 1 (Data Lead)
"""

import os
import sys
import pytest
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from biomed_lake.storage.d_mgr import DeltaLakeManager
from biomed_lake.storage.lancedb_mgr import LanceDBClinicalManager


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "sample.csv"
    fn.write_text(
        "male,age,education,currentSmoker,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose,TenYearCHD\n"
        "1,39,4,0,0,0,0,0,0,195,106,70,26.97,80,77,0\n"
        "0,46,2,0,0,0,0,0,0,250,121,81,28.73,95,76,0\n"
        "1,48,1,1,20,0,0,0,0,245,127.5,80,25.34,75,70,0\n"
    )
    return str(fn)


def test_delta_lake_ingest_and_audit(tmp_path, sample_csv):
    delta_dir = str(tmp_path / "delta_test")
    mgr = DeltaLakeManager(sample_csv, delta_dir)
    df = mgr.ingest_and_optimize()

    assert df.height == 3
    assert "person_id" in df.columns
    assert df["is_male"].dtype == pl.Int8

    # Test save
    dt = mgr.save_to_delta(df, mode="overwrite")
    assert dt.version() == 0

    # Test FDA audit trail
    history = mgr.get_audit_trail()
    assert len(history) >= 1

    # Test Time-Travel
    df_travel = mgr.read_time_travel(version=0)
    assert df_travel.height == 3


def test_lancedb_vector_knn(tmp_path, sample_csv):
    lance_dir = str(tmp_path / "lance_test")
    delta_dir = str(tmp_path / "delta_test_2")

    delta_mgr = DeltaLakeManager(sample_csv, delta_dir)
    df = delta_mgr.ingest_and_optimize()

    lance_mgr = LanceDBClinicalManager(lance_dir)
    table = lance_mgr.create_cohort_table(df, table_name="test_cohort", mode="overwrite")
    assert table.count_rows() == 3

    # Test vector search
    query_vec = [0.5, 0.4, 0.5, 0.3, 0.2, 0.4, 0.0, 0.3]
    res = lance_mgr.find_similar_patients(query_vec, k=2, table_name="test_cohort")
    assert res.height == 2
    assert "_distance" in res.columns
