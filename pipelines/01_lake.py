"""
Pipeline 01: Ingest Raw Clinical CSV -> Delta Lake ACID -> LanceDB Clinical Mart
Phụ trách: TV 1 (Clinical Data Engineering Lead)
"""

import os
import sys
import time
import yaml

# Đảm bảo import được src/biomed_lake
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from biomed_lake.storage.d_mgr import DeltaLakeManager
from biomed_lake.storage.lancedb_mgr import LanceDBClinicalManager


def run_pipeline():
    print("=" * 70)
    print("🚀 PIPELINE 01: CLINICAL LAKEHOUSE & LANCEDB MART INGESTION (TV 1)")
    print("=" * 70)
    start_time = time.time()

    # 1. Đọc cấu hình
    config_path = os.path.join(PROJECT_ROOT, "configs", "storage.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_csv = os.path.join(PROJECT_ROOT, "data", "landing", "framingham.csv")
    delta_path = os.path.join(PROJECT_ROOT, cfg["lakehouse"]["delta_curated_path"])
    lancedb_uri = os.path.join(PROJECT_ROOT, cfg["lakehouse"]["lancedb_mart_uri"])
    table_name = cfg["lakehouse"]["table_name"]

    # 2. Xử lý & Ghi vào Delta Lake ACID
    print("\n--- GIAI ĐOẠN 1: DELTA LAKE ACID INGESTION ---")
    delta_mgr = DeltaLakeManager(raw_csv, delta_path)
    df_clean = delta_mgr.ingest_and_optimize()
    dt = delta_mgr.save_to_delta(df_clean, mode="overwrite")

    # Kiểm tra nhật ký kiểm toán Time-Travel
    history = delta_mgr.get_audit_trail()
    print(f"[✓] Nhật ký kiểm toán FDA 21 CFR Part 11: Đã ghi nhận {len(history)} giao dịch.")

    # 3. Nạp vào LanceDB & Đánh chỉ mục Véc-tơ tương đồng
    print("\n--- GIAI ĐOẠN 2: LANCEDB MULTIMODAL & VECTOR MART ---")
    lance_mgr = LanceDBClinicalManager(lancedb_uri)
    lance_table = lance_mgr.create_cohort_table(df_clean, table_name=table_name, mode="overwrite")

    # 4. Kiểm thử tính năng Case-Based Reasoning (k-NN Patient Similarity)
    print("\n--- GIAI ĐOẠN 3: KIỂM THỬ SUY LUẬN BỆNH NHÂN TƯƠNG ĐỒNG (k-NN) ---")
    sample_query = [0.5, 0.4, 0.5, 0.3, 0.2, 0.4, 0.0, 0.3]  # Vector giả định của một bệnh nhân
    print(f"[*] Thực hiện truy vấn Top-3 ca bệnh tương đồng nhất với vector đầu vào...")
    similar_patients = lance_mgr.find_similar_patients(sample_query, k=3, table_name=table_name)
    print(similar_patients.select(["person_id", "age", "is_male", "sys_bp", "dia_bp", "ten_year_chd", "_distance"]))

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"✅ PIPELINE 01 HOÀN TẤT THÀNH CÔNG TRONG {elapsed:.2f} GIÂY!")
    print(f"   • Tầng Curated: {delta_path} (Delta Lake ACID)")
    print(f"   • Tầng Data Mart: {lancedb_uri} (LanceDB Vector-Columnar)")
    print(f"   • Bảng phân tích: '{table_name}' sẵn sàng bàn giao cho TV 2 -> TV 6!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
