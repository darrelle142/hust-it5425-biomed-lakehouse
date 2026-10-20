import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from biomed_lake.biostats.test import generate_table_one, run_hypothesis_test, plot_biochemical_distributions
from biomed_lake.biostats.odds import calculate_or_ci, plot_forest

@pytest.fixture(scope="module")
def sample_clinical_df():
    """
    Tạo DataFrame giả lập với các đặc trưng thống kê để kiểm thử
    - Nhóm 0 (Không mắc bệnh): Giá trị sinh hóa bình thường
    - Nhóm 1 (Có mắc bệnh): Giá trị sinh hóa cao hơn, rủi ro cao
    """
    np.random.seed(42)
    n_samples = 200
    
    # Tạo dữ liệu giả lập có phân phối lệch cho nhóm 1
    data = {
        'age': np.concatenate([np.random.normal(45, 8, 100), np.random.normal(55, 8, 100)]),
        'sys_bp': np.concatenate([np.random.normal(120, 15, 100), np.random.normal(145, 20, 100)]),
        'dia_bp': np.concatenate([np.random.normal(80, 10, 100), np.random.normal(90, 12, 100)]),
        'BMI': np.concatenate([np.random.normal(24, 3, 100), np.random.normal(28, 4, 100)]),
        'glucose': np.concatenate([np.random.normal(85, 10, 100), np.random.normal(110, 25, 100)]),
        'tot_chol': np.concatenate([np.random.normal(190, 30, 100), np.random.normal(230, 40, 100)]),
        'is_male': np.random.randint(0, 2, n_samples),
        'current_smoker': np.concatenate([np.random.choice([0, 1], 100, p=[0.7, 0.3]), 
                                          np.random.choice([0, 1], 100, p=[0.4, 0.6])]),
        'diabetes': np.concatenate([np.zeros(100), np.random.choice([0, 1], 100, p=[0.8, 0.2])]),
        'bp_stage': np.random.choice(['NORMAL', 'ELEVATED', 'STAGE_1_HYPERTENSION'], n_samples),
        'ten_year_chd': np.concatenate([np.zeros(100), np.ones(100)]) # Target
    }
    
    return pd.DataFrame(data)

@pytest.fixture(scope="module")
def continuous_cols():
    return ['age', 'sys_bp', 'dia_bp', 'tot_chol', 'BMI', 'glucose']

@pytest.fixture(scope="module")
def categorical_cols():
    return ['is_male', 'current_smoker', 'diabetes', 'bp_stage']

def test_generate_table_one(tmp_path, sample_clinical_df, continuous_cols, categorical_cols):
    """
    Kiểm thử chức năng sinh Table 1:
    - Đảm bảo file CSV và HTML được tạo ra
    - Đảm bảo có cột p-value
    """
    all_cols = continuous_cols + categorical_cols
    table = generate_table_one(
        df=sample_clinical_df,
        cols=all_cols,
        categorical=categorical_cols,
        target='ten_year_chd',
        nonnormal=continuous_cols,
        output_dir=tmp_path
    )
    
    assert (tmp_path / 'table1.csv').exists()
    assert (tmp_path / 'table1.html').exists()
    
    csv_path = tmp_path / 'table1.csv'
    
    # Kiểm tra object tableone có chứa thông tin chia nhóm
    table_df = table.tableone
    assert 'Grouped by ten_year_chd' in table_df.columns.levels[0]
    
    # Đọc file CSV vừa tạo ra để kiểm tra chắc chắn có cột P-Value
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_content = f.read().lower()
        assert 'pval' in csv_content or 'p-value' in csv_content

def test_hypothesis_testing(sample_clinical_df, continuous_cols, categorical_cols):
    """
    Kiểm thử thuật toán rẽ nhánh kiểm định:
    - Trả về đúng DataFrame gồm 3 cột: Biến, Kiểm định, P-value
    - Phát hiện đúng các biến có ý nghĩa thống kê (do data giả lập nhóm 1 cố tình set cao hơn)
    """
    results = run_hypothesis_test(
        df=sample_clinical_df,
        continuous=continuous_cols,
        categorical=categorical_cols,
        target='ten_year_chd'
    )
    
    assert isinstance(results, pd.DataFrame)
    assert list(results.columns) == ['Biến', 'Kiểm định', 'P-value']
    assert len(results) == len(continuous_cols) + len(categorical_cols)
    
    # Kiểm tra xem huyết áp và tuổi (được set khác biệt rõ rệt) có p-value < 0.05 không
    sig_vars = results[results['P-value'] < 0.05]['Biến'].tolist()
    assert 'age' in sig_vars
    assert 'sys_bp' in sig_vars

def test_plot_biochemical_distributions(tmp_path, sample_clinical_df, continuous_cols):
    """
    Kiểm thử hàm vẽ biểu đồ: Đảm bảo hình ảnh được lưu thành công không văng lỗi
    """
    plot_biochemical_distributions(
        df=sample_clinical_df,
        continuous=continuous_cols,
        target='ten_year_chd',
        output_dir=tmp_path
    )
    assert (tmp_path / 'split_violin_plots.png').exists()

def test_calculate_odds_ratio(sample_clinical_df, continuous_cols):
    """
    Kiểm thử mô hình Logistic Regression tính OR:
    - Chạy thành công, tính toán chuẩn hóa Z-score
    - Trả về DataFrame chứa CI_lower, CI_upper, OR, p_value
    - Đảm bảo không bị văng lỗi định danh Dummy variables
    """
    predictors = ['age', 'tot_chol', 'BMI', 'glucose', 'is_male', 'current_smoker', 'diabetes', 'bp_stage']
    
    or_df = calculate_or_ci(
        df=sample_clinical_df,
        predictors=predictors,
        continuous=continuous_cols,
        target='ten_year_chd'
    )
    
    # Cột mặc định
    assert set(['CI_lower', 'CI_upper', 'OR', 'p_value']).issubset(or_df.columns)
    
    # Đảm bảo bp_stage đã được One-Hot Encoding (trừ drop_first)
    expected_dummy_cols = [c for c in or_df.index if 'bp_stage_' in c]
    assert len(expected_dummy_cols) > 0
    
    # Đảm bảo giá trị OR hợp lệ (lớn hơn 0 do là hàm mũ)
    assert (or_df['OR'] > 0).all()

def test_plot_forest(tmp_path, sample_clinical_df, continuous_cols):
    """
    Kiểm thử chức năng vẽ Forest Plot:
    Truyền kết quả OR giả lập vào để vẽ, kiểm tra không bị lỗi KeyError: 0
    """
    predictors = ['age', 'BMI', 'current_smoker']
    or_df = calculate_or_ci(
        df=sample_clinical_df,
        predictors=predictors,
        continuous=continuous_cols,
        target='ten_year_chd'
    )
    
    plot_forest(or_df=or_df, output_dir=tmp_path)
    assert (tmp_path / 'forest_plot.png').exists()
