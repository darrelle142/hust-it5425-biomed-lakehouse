"""
Công cụ kiểm tra và xem trước dữ liệu trong kho LanceDB (Star Schema Mart)
Xuất bản tệp CSV xem trước để người dùng có thể nhấp đúp mở bằng Excel / VS Code.
"""

import os
import sys
import lancedb
import polars as pl

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(os.path.join(PROJECT_ROOT, "src")))
from biomed_lake.storage.lancedb_mgr import LanceDBClinicalManager

DB_URI = os.path.join(PROJECT_ROOT, "data", "mart", "clin.lance")
MART_DIR = os.path.join(PROJECT_ROOT, "data", "mart")


def inspect_lancedb():
    print("=" * 70)
    print("[*] CONG CU XEM DU LIEU LANCEDB (STAR SCHEMA MART)")
    print("=" * 70)

    if not os.path.exists(DB_URI):
        print(f"[-] Chưa tìm thấy CSDL tại {DB_URI}. Hãy chạy 'python pipelines/01_lake.py' trước.")
        return

    lance_mgr = LanceDBClinicalManager(DB_URI)
    db = lance_mgr.db
    table_names = db.table_names() if hasattr(db, "table_names") else db.list_tables()
    print(f"[+] Các bảng có trong LanceDB ({DB_URI}): {table_names}\n")

    for tbl_name in table_names:
        tbl = db.open_table(tbl_name)
        row_count = tbl.count_rows()
        cols = tbl.schema.names
        print(f"  • Bảng '{tbl_name}': {row_count:,} dòng | Cột ({len(cols)}): {cols}")

        # Xuất preview CSV cho từng bảng
        out_csv = os.path.join(MART_DIR, f"preview_{tbl_name}.csv")
        df_arrow = tbl.to_arrow()
        df_pl = pl.from_arrow(df_arrow)
        if "phenotype_vector" in df_pl.columns:
            df_pl = df_pl.with_columns(
                pl.col("phenotype_vector").map_elements(
                    lambda vec: "[" + ", ".join(f"{x:.4f}" for x in (vec.to_list() if hasattr(vec, "to_list") else vec)) + "]",
                    return_dtype=pl.Utf8
                )
            )
        df_pl.write_csv(out_csv)

    # Xuất bản ma trận phân tích Star JOIN
    print("\n[*] Đang thực hiện Star JOIN và xuất bản ma trận phân tích tổng hợp...")
    analytical_lf = lance_mgr.get_analytical_cohort()
    analytical_df = analytical_lf.collect()
    preview_analytical_csv = os.path.join(MART_DIR, "preview_cohort.csv")
    if "phenotype_vector" in analytical_df.columns:
        analytical_df = analytical_df.with_columns(
            pl.col("phenotype_vector").map_elements(
                lambda vec: "[" + ", ".join(f"{x:.4f}" for x in (vec.to_list() if hasattr(vec, "to_list") else vec)) + "]",
                return_dtype=pl.Utf8
            )
        )
    analytical_df.write_csv(preview_analytical_csv)
    print(f"[✓] ĐÃ XUẤT THÀNH CÔNG: {preview_analytical_csv} ({analytical_df.height:,} dòng, {len(analytical_df.columns)} cột)")
    print(f"    -> Các bảng Star Schema đã sẵn sàng trong thư mục: {MART_DIR}")


if __name__ == "__main__":
    inspect_lancedb()
