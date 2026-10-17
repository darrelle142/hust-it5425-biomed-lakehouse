import sys
import lancedb
import polars as pl
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from biomed_lake.biostats.test import generate_table_one, run_hypothesis_test, plot_biochemical_distributions
from biomed_lake.biostats.odds import calculate_or_ci, plot_forest


def run_pipeline():
    print("=" * 80)
    print("[*] PIPELINE 03: BIOMEDICAL EDA & HYPOTHESIS TESTING")
    print("=" * 80)

    table_dir = PROJECT_ROOT / 'reports' / 'tables'
    table_dir.mkdir(exist_ok=True, parents=True)
    figure_dir = PROJECT_ROOT / 'reports' / 'figures'
    figure_dir.mkdir(exist_ok=True, parents=True)

    print("[*] Đang lấy dữ liệu từ LanceDB")
    db = lancedb.connect(PROJECT_ROOT / 'data' / 'mart' / 'clin.lance')
    fact_tbl = db.open_table('fact_clinical_cohort').to_polars()

    continuous_cols = ['age', 'sys_bp', 'dia_bp', 'tot_chol', 'BMI', 'glucose']
    categorical_cols = ['is_male', 'current_smoker', 'bp_stage', 'diabetes', 'bp_meds']
    all_cols = continuous_cols + categorical_cols
    target_col = 'ten_year_chd'

    df_clean = fact_tbl.drop_nulls(subset=all_cols + [target_col]).collect()
    print(f"[*] Kích thước mẫu đủ điều kiện phân tích: {df_clean.height} bệnh nhân")

    df_clean = df_clean.to_pandas()

    print("\n--- Giai đoạn 1: Xây dựng Table 1 ---")
    generate_table_one(df=df_clean, cols=all_cols, categorical=categorical_cols, nonnormal=continuous_cols, target=target_col, output_dir=table_dir)

    print("\n--- Giai đoạn 2: Kiểm định giả thuyết và vẽ phân phối ---")
    hypothesis_results = run_hypothesis_test(df=df_clean, continuous=continuous_cols, categorical=categorical_cols, target=target_col)
    print("[Kết quả kiểm định giả thuyết]")
    print(hypothesis_results.to_string(index=False))

    plot_biochemical_distributions(df=df_clean, continuous=continuous_cols, target=target_col, output_dir=figure_dir)

    print("\n--- Giai đoạn 3: Tính toán tỉ số chênh lệch OR và vẽ biểu đồ ---")
    or_predictors = [col for col in all_cols if col not in ['sys_bp', 'dia_bp']]
    or_results = calculate_or_ci(df=df_clean, predictors=or_predictors, continuous=continuous_cols, target=target_col)
    print("[Bảng OR]")
    print(or_results)

    plot_forest(or_df=or_results, output_dir=figure_dir)

    print("[OK] PIPELINE 3 HOÀN TẤT")


if __name__ == "__main__":
    run_pipeline()
