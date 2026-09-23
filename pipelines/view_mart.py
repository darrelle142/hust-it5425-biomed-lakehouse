"""
Công cụ kiểm tra và xem trước dữ liệu trong kho LanceDB (fact_clinical_cohort)
Xuất bản tệp CSV xem trước để người dùng có thể nhấp đúp mở bằng Excel / VS Code.
"""

import os
import lancedb
import polars as pl

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_URI = os.path.join(PROJECT_ROOT, "data", "mart", "clin.lance")
PREVIEW_CSV = os.path.join(PROJECT_ROOT, "data", "mart", "preview_cohort.csv")


def inspect_lancedb():
    print("=" * 70)
    print("🔍 CÔNG CỤ XEM DỮ LIỆU LANCEDB (CLINICAL DATA MART)")
    print("=" * 70)

    if not os.path.exists(DB_URI):
        print(f"[-] Chưa tìm thấy CSDL tại {DB_URI}. Hãy chạy 'python pipelines/01_lake.py' trước.")
        return

    db = lancedb.connect(DB_URI)
    table_names = db.table_names()
    print(f"[+] Các bảng có trong LanceDB ({DB_URI}): {table_names}")

    if "fact_clinical_cohort" not in table_names:
        print("[-] Chưa có bảng 'fact_clinical_cohort'.")
        return

    table = db.open_table("fact_clinical_cohort")
    row_count = table.count_rows()
    print(f"[+] Bảng 'fact_clinical_cohort' có tổng cộng: {row_count:,} bản ghi bệnh nhân.")

    # Đọc 10 dòng đầu bằng Polars
    arrow_data = table.head(10)
    df_sample = pl.from_arrow(arrow_data)

    print("\n[+] 10 DÒNG ĐẦU TIÊN CỦA BẢNG (DỮ LIỆU ĐÃ LÀM SẠCH & CÓ VÉC-TƠ):")
    # Ẩn cột véc-tơ khi in terminal để không bị tràn màn hình
    display_cols = [c for c in df_sample.columns if c != "phenotype_vector"]
    print(df_sample.select(display_cols))

    print("\n[+] VÉC-TƠ THỂ TRẠNG BỆNH NHÂN (PHENOTYPE VECTOR 8 CHIỀU) CỦA BỆNH NHÂN ĐẦU TIÊN:")
    first_vec = df_sample["phenotype_vector"][0]
    print("   ", [round(x, 4) for x in first_vec])

    # Xuất ra file CSV (chuyển cột vector thành text chuỗi để xem được trong Excel/VS Code)
    print(f"\n[*] Đang xuất bản CSV xem trước (bao gồm cả cột Vector) tại: {PREVIEW_CSV}")
    full_arrow = table.to_arrow()
    df_full = pl.from_arrow(full_arrow)
    # Chuyển list float sang string để Excel đọc mượt mà
    df_csv = df_full.with_columns(
        pl.col("phenotype_vector").map_elements(
            lambda vec: "[" + ", ".join(f"{x:.4f}" for x in (vec.to_list() if hasattr(vec, "to_list") else vec)) + "]",
            return_dtype=pl.Utf8
        )
    )
    df_csv.write_csv(PREVIEW_CSV)
    print(f"[✓] ĐÃ XUẤT THÀNH CÔNG: {PREVIEW_CSV}")
    print("    -> Bạn có thể nhấp đúp vào file 'preview_cohort.csv' để xem tận mắt cột phenotype_vector trên Excel!")


if __name__ == "__main__":
    inspect_lancedb()
