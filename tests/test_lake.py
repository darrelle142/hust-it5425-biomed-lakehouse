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
from biomed_lake.storage.integration import ClinicalDataIntegrationEngine


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


def test_data_integration_and_star_schema(tmp_path, sample_csv):
    """
    Kiểm thử Tích hợp dữ liệu đa nguồn (Slide 4) và Mô hình Star Schema (Slide 3.2)
    """
    feeds_dir = str(tmp_path / "feeds")
    demo_file, labs_file = ClinicalDataIntegrationEngine.partition_raw_to_feeds(sample_csv, feeds_dir)

    engine = ClinicalDataIntegrationEngine()
    df_integrated, audit = engine.mediate_and_link(demo_file, labs_file, linkage_key="person_id")

    assert df_integrated.height == 3
    assert audit["matched_patients"] == 3
    assert "person_id" in df_integrated.columns
    assert "is_male" in df_integrated.columns
    assert "tot_chol" in df_integrated.columns

    # Kiểm thử tạo Star Schema trong LanceDB
    lance_dir = str(tmp_path / "lance_star_test")
    lance_mgr = LanceDBClinicalManager(lance_dir)
    star_tables = lance_mgr.create_star_schema_tables(df_integrated, mode="overwrite")

    assert "dim_patient" in star_tables
    assert "dim_clinical_metrics" in star_tables
    assert "fact_clinical_cohort" in star_tables
    assert star_tables["dim_patient"].count_rows() == 3
    assert star_tables["dim_clinical_metrics"].count_rows() == 3
    assert star_tables["fact_clinical_cohort"].count_rows() == 3


def test_multi_center_and_aha_enrichment(tmp_path):
    """
    Kiểm thử Tích hợp Đa Trung Tâm (Framingham + Cardio Study) và Làm giàu AHA/JNC-7
    """
    # 1. Tạo dữ liệu giả lập Framingham
    fhs_file = str(tmp_path / "fhs.csv")
    pl.DataFrame({
        "male": [1, 0],
        "age": [39, 46],
        "education": [4, 2],
        "currentSmoker": [0, 0],
        "cigsPerDay": [0, 0],
        "BPMeds": [0, 0],
        "prevalentStroke": [0, 0],
        "prevalentHyp": [0, 0],
        "diabetes": [0, 0],
        "totChol": [195.0, 250.0],
        "sysBP": [106.0, 145.0],
        "diaBP": [70.0, 92.0],
        "BMI": [26.97, 28.73],
        "heartRate": [80.0, 95.0],
        "glucose": [77.0, 76.0],
        "TenYearCHD": [0, 1]
    }).write_csv(fhs_file)

    # 2. Tạo dữ liệu giả lập Cardio Study
    cardio_file = str(tmp_path / "cardio.csv")
    pl.DataFrame({
        "id": [1, 2],
        "age": [18393, 20228],
        "gender": [2, 1],
        "height": [168, 156],
        "weight": [62.0, 85.0],
        "ap_hi": [110.0, 182.0],
        "ap_lo": [80.0, 122.0],
        "cholesterol": [1, 3],
        "gluc": [1, 2],
        "smoke": [0, 1],
        "alco": [0, 0],
        "active": [1, 1],
        "cardio": [0, 1]
    }).write_csv(cardio_file, separator=";")

    # 3. Tạo dữ liệu AHA Guidelines
    guidelines_file = str(tmp_path / "guidelines.csv")
    df_guide = pl.DataFrame({
        "guideline_id": [1, 2, 3, 4, 5],
        "bp_stage": ["NORMAL", "ELEVATED", "STAGE_1_HYPERTENSION", "STAGE_2_HYPERTENSION", "HYPERTENSIVE_CRISIS"],
        "min_sys_bp": [0.0, 120.0, 130.0, 140.0, 180.0],
        "max_sys_bp": [119.9, 129.9, 139.9, 179.9, 999.0],
        "min_dia_bp": [0.0, 0.0, 80.0, 90.0, 120.0],
        "max_dia_bp": [79.9, 79.9, 89.9, 119.9, 999.0],
        "clinical_action": ["Action1", "Action2", "Action3", "Action4", "Action5"],
        "icd10_code": ["R03.0", "R03.0", "I10", "I10.9", "I16.9"]
    })
    df_guide.write_csv(guidelines_file)

    # 4. Kiểm tra Tích hợp Đa Trung Tâm
    engine = ClinicalDataIntegrationEngine()
    df_merged = engine.integrate_multi_center_cohorts(fhs_file, cardio_file)
    assert df_merged.height == 4
    assert "person_id" in df_merged.columns
    assert "center_source" in df_merged.columns
    assert set(df_merged["center_source"].to_list()) == {"FRAMINGHAM_US", "CARDIO_CLINICAL_INTL"}

    # 5. Kiểm tra làm giàu AHA Guidelines
    df_enriched = engine.enrich_with_clinical_guidelines(df_merged, guidelines_file)
    assert "bp_stage" in df_enriched.columns
    assert "icd10_code" in df_enriched.columns

    # 6. Kiểm tra tạo Star Schema với Bảng Hướng dẫn
    lance_dir = str(tmp_path / "lance_multi_star")
    lance_mgr = LanceDBClinicalManager(lance_dir)
    star_tables = lance_mgr.create_star_schema_tables(df_enriched, guidelines_df=df_guide, mode="overwrite")
    assert "dim_aha_guidelines" in star_tables
    assert star_tables["dim_patient"].count_rows() == 4
    assert star_tables["dim_aha_guidelines"].count_rows() == 5


def test_clinical_cleansing_and_governance_rules(tmp_path):
    """
    Kiểm thử cơ chế Chuẩn hóa & Làm sạch dữ liệu lâm sàng (Clinical Data Cleansing):
    Đảm bảo 100% các lỗi dị thường phát hiện bởi TV 2 (Governance) được giải quyết:
    1. Lỗi thang đo cmHg -> mmHg (ap_hi = 14 -> sys_bp = 140)
    2. Lỗi thừa số 0 ở tâm trương (ap_lo = 1000 -> dia_bp = 100)
    3. Lỗi hoán đổi cột (ap_hi = 80, ap_lo = 120 -> sys_bp = 120, dia_bp = 80)
    4. Lỗi chiều cao dị thường (height = 76cm -> BMI trong ngưỡng [10, 80])
    5. Đảm bảo ràng buộc sys_bp >= dia_bp và miền giá trị sinh lý học
    """
    engine = ClinicalDataIntegrationEngine()
    test_cardio_file = str(tmp_path / "anomalous_cardio.csv")

    # Giả lập các ca dị thường điển hình từ báo cáo TV 2
    pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [18000, 19000, 20000, 21000, 22000],
        "gender": [2, 1, 2, 1, 1],
        "height": [168, 156, 175, 76, 165],     # Ca 4 có chiều cao dị thường 76cm
        "weight": [65.0, 70.0, 80.0, 55.0, 75.0],
        "ap_hi": [14.0, 160.0, 80.0, 120.0, -100.0],  # Ca 1: cmHg (14), Ca 3: đảo ngược (80), Ca 5: âm (-100)
        "ap_lo": [80.0, 1000.0, 120.0, 80.0, 80.0],  # Ca 2: thừa số 0 (1000)
        "cholesterol": [1, 2, 3, 1, 2],
        "gluc": [1, 1, 2, 1, 1],
        "smoke": [0, 0, 1, 0, 0],
        "alco": [0, 0, 0, 0, 0],
        "active": [1, 1, 1, 1, 1],
        "cardio": [0, 1, 0, 0, 1]
    }).write_csv(test_cardio_file, separator=";")

    df_cleaned = engine.wrap_cardio_study(test_cardio_file)

    # Nghiệm chứng 1: Ca 1 sửa cmHg -> mmHg
    row1 = df_cleaned.row(0, named=True)
    assert row1["sys_bp"] == 140.0
    assert row1["dia_bp"] == 80.0

    # Nghiệm chứng 2: Ca 2 sửa thừa số 0 (1000 -> 100)
    row2 = df_cleaned.row(1, named=True)
    assert row2["sys_bp"] == 160.0
    assert row2["dia_bp"] == 100.0

    # Nghiệm chứng 3: Ca 3 đảo ngược cột hợp lý (80/120 -> 120/80)
    row3 = df_cleaned.row(2, named=True)
    assert row3["sys_bp"] == 120.0
    assert row3["dia_bp"] == 80.0

    # Nghiệm chứng 4: Ca 4 chuẩn hóa chiều cao (76 -> 176), BMI hợp lệ
    row4 = df_cleaned.row(3, named=True)
    assert 10.0 <= row4["BMI"] <= 80.0

    # Nghiệm chứng 5: Ca 5 ngoại lệ cực đoan được impute an toàn
    row5 = df_cleaned.row(4, named=True)
    assert 0.0 < row5["sys_bp"] <= 300.0
    assert 0.0 < row5["dia_bp"] <= 200.0

    # Nghiệm chứng tổng thể: Không có bất kỳ vi phạm Great Expectations nào
    assert df_cleaned.filter(pl.col("sys_bp") < pl.col("dia_bp")).height == 0
    assert df_cleaned.filter((pl.col("sys_bp") <= 0) | (pl.col("sys_bp") > 300)).height == 0
    assert df_cleaned.filter((pl.col("dia_bp") <= 0) | (pl.col("dia_bp") > 200)).height == 0
    assert df_cleaned.filter((pl.col("BMI") < 10) | (pl.col("BMI") > 80)).height == 0

