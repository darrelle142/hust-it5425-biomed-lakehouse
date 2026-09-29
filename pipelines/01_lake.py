"""
Pipeline 01: Ingest Multi-Source Clinical Data -> Schema Mediation -> Delta Lake ACID -> LanceDB Star Schema Mart
Phụ trách: TV 1 (Clinical Data Engineering Lead)
Cơ sở lý thuyết: IT5425 Slide Chương 3.2 (OLTP & OLAP), Chương 3.3 (Data Lakehouse), Chương 4 (Data Integration & Preprocessing)
"""

import os
import sys
import time
import yaml
import polars as pl
from deltalake import write_deltalake, DeltaTable

# Khắc phục bảng mã stdout Windows cp1258
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Đảm bảo import được src/biomed_lake và pipelines
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, PROJECT_ROOT)

from biomed_lake.storage.d_mgr import DeltaLakeManager
from biomed_lake.storage.lancedb_mgr import LanceDBClinicalManager
from biomed_lake.storage.integration import ClinicalDataIntegrationEngine
from pipelines.download_clinical_sources import init_all_sources


def run_pipeline():
    print("=" * 80)
    print("[*] PIPELINE 01: MULTI-SOURCE CLINICAL INTEGRATION & STAR SCHEMA LAKEHOUSE")
    print("=" * 80)
    start_time = time.time()

    # 1. Đọc cấu hình
    config_path = os.path.join(PROJECT_ROOT, "configs", "storage.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    landing_dir = os.path.join(PROJECT_ROOT, "data", "landing")
    fhs_file = os.path.join(landing_dir, "framingham.csv")
    cardio_file = os.path.join(landing_dir, "cardio_study.csv")
    guidelines_file = os.path.join(landing_dir, "ref_aha_jnc7_guidelines.csv")

    delta_path = os.path.join(PROJECT_ROOT, cfg["lakehouse"]["delta_curated_path"])
    lancedb_uri = os.path.join(PROJECT_ROOT, cfg["lakehouse"]["lancedb_mart_uri"])
    table_name = cfg["lakehouse"]["table_name"]

    # 2. Giai đoạn 0: Kiểm tra hoặc tải 3 nguồn dữ liệu độc lập
    if not os.path.exists(fhs_file) or not os.path.exists(cardio_file) or not os.path.exists(guidelines_file):
        print("\n--- GIAI DOAN 0: KHOI TAO 3 NGUON DU LIEU DOC LAP (LANDING ZONE) ---")
        init_all_sources()

    # 3. Giai đoạn 1: Tích hợp Đa Trung Tâm & Hòa giải Lược đồ (Schema Mediation - Slide 4)
    print("\n--- GIAI DOAN 1: MULTI-CENTER INTEGRATION & SCHEMA MEDIATION ---")
    engine = ClinicalDataIntegrationEngine()
    df_merged = engine.integrate_multi_center_cohorts(fhs_file, cardio_file)

    # 4. Giai đoạn 2: Làm giàu Dữ liệu theo Quy chuẩn Lâm sàng Quốc tế (AHA/JNC-7 & ICD-10 - Slide 3.2)
    print("\n--- GIAI DOAN 2: CLINICAL GUIDELINES ENRICHMENT (LOOKUP JOIN) ---")
    df_enriched = engine.enrich_with_clinical_guidelines(df_merged, guidelines_file)
    df_guidelines = pl.read_csv(guidelines_file)

    # 5. Giai đoạn 3: Ghi vào Tầng Curated Delta Lake ACID (Slide 3.3)
    print("\n--- GIAI DOAN 3: CURATED DELTA LAKE ACID INGESTION ---")
    print(f"[*] Dang ghi {df_enriched.height:,} ban ghi hop nhat vao Delta Lake: {delta_path}")
    write_deltalake(
        delta_path,
        df_enriched.to_arrow(),
        mode="overwrite",
        schema_mode="overwrite"
    )
    dt = DeltaTable(delta_path)
    print(f"[+] Ghi Delta Lake thanh cong! Phien ban hien tai (version): {dt.version()}")
    print(f"[v] Nhat ky kiem toan FDA 21 CFR Part 11: Da ghi nhan {len(dt.history())} giao dich.")

    # 6. Giai đoạn 4: Kiến tạo LanceDB Star Schema Multimodal Mart (Slide 3.2)
    print("\n--- GIAI DOAN 4: LANCEDB STAR SCHEMA MULTIMODAL MART ---")
    lance_mgr = LanceDBClinicalManager(lancedb_uri)
    star_tables = lance_mgr.create_star_schema_tables(df_enriched, guidelines_df=df_guidelines, mode="overwrite")
    print(f"[+] Hoan tat kien truc Star Schema gom:")
    print(f"    - Bang Chieu: 'dim_patient' ({star_tables['dim_patient'].count_rows():,} dong)")
    print(f"    - Bang Chieu: 'dim_clinical_metrics' ({star_tables['dim_clinical_metrics'].count_rows():,} dong)")
    print(f"    - Bang Chieu: 'dim_aha_guidelines' ({star_tables['dim_aha_guidelines'].count_rows():,} dong)")
    print(f"    - Bang Su kien: 'fact_clinical_cohort' ({star_tables['fact_clinical_cohort'].count_rows():,} dong)")

    # 7. Giai đoạn 5: Kiểm thử suy luận ca bệnh tương đồng (k-NN Case-Based Reasoning)
    print("\n--- GIAI DOAN 5: KIEM THU SUY LUAN BENH NHAN TUONG DONG (k-NN) ---")
    sample_query = [0.5, 0.4, 0.5, 0.3, 0.2, 0.4, 0.0, 0.3]
    print(f"[*] Thuc hien truy van Top-3 ca benh tuong dong nhat voi vector dau vao tren tap 9,240 benh nhan...")
    similar = lance_mgr.find_similar_patients(sample_query, k=3, table_name="fact_clinical_cohort")
    print(similar.select(["person_id", "age", "is_male", "sys_bp", "dia_bp", "bp_stage", "icd10_code", "center_source", "_distance"]))

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"[OK] PIPELINE 01 HOAN TAT XUAT SAC TRONG {elapsed:.2f} GIAY!")
    print(f"   * Quy mo du lieu: 9,240 benh nhan (4,240 Framingham + 5,000 Cardio Study)")
    print(f"   * Nguon tra cuu y te: AHA/ACC & JNC-7 Guidelines kem ma ICD-10")
    print(f"   * Tang Curated: Delta Lake ACID tai {delta_path}")
    print(f"   * Tang Data Mart: LanceDB Star Schema Multimodal tai {lancedb_uri}")
    print(f"   * San sang ban giao cho TV 2 -> TV 6!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
