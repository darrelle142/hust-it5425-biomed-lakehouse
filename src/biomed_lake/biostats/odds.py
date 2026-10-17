import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


def calculate_or_ci(df, predictors, continuous, target):
    print("[+] Đang tính toán Odds Ratio")

    df_clean = df[predictors + [target]].dropna()

    scaler = StandardScaler()
    cols_to_scale = [c for c in continuous if c in predictors]
    if cols_to_scale:
        df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])

    X = pd.get_dummies(df_clean[predictors], drop_first=True).astype(float)
    X = sm.add_constant(X)
    y = df_clean[target].astype(float)

    model = sm.Logit(y, X).fit(disp=0)
    params = model.params
    conf = model.conf_int()
    conf['OR'] = params
    conf.columns = ['CI_lower', 'CI_upper', 'OR']

    or_df = np.exp(conf).drop('const', errors='ignore')
    or_df['p_value'] = model.pvalues.drop('const', errors='ignore')

    return or_df

def plot_forest(or_df, output_dir):
    print("[+] Đang vẽ biểu đồ Forest")

    or_df = or_df.sort_values(by='OR')
    variables = or_df.index
    or_vals = or_df['OR']
    err_lower = or_vals - or_df['CI_lower']
    err_upper = or_df['CI_upper'] - or_vals

    plt.figure(figsize=(12, 5))
    plt.errorbar(or_vals, variables, xerr=[err_lower, err_upper], fmt='o', color='darkred', capsize=5, elinewidth=2)
    plt.axvline(x=1, color='black', linestyle='--')
    plt.xscale('log')
    plt.xlabel("OR (Log scale)")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.title("Forest plot of risk factors (OR and 95% CI)")
    plt.tight_layout()
    plt.savefig(output_dir / 'forest_plot.png', dpi=300)
    plt.close()
