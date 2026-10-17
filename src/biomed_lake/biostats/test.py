import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from tableone import TableOne


def generate_table_one(df, cols, categorical, nonnormal, target, output_dir):
    print("[+] Đang xây dựng Table 1")

    table = TableOne(df, columns=cols, categorical=categorical, nonnormal=nonnormal, groupby=target, pval=True)
    table.to_csv(output_dir / 'table1.csv')
    table.to_html(output_dir / 'table1.html')

    return table

def run_hypothesis_test(df, continuous, categorical, target):
    print("[+] Đang kiểm định giả thuyết")

    results = []
    g0 = df[df[target] == 0]
    g1 = df[df[target] == 1]

    for col in continuous:
        _, p_norm0 = stats.normaltest(g0[col].dropna())
        _, p_norm1 = stats.normaltest(g1[col].dropna())

        if p_norm0 > 0.05 and p_norm1 > 0.05:
            _, p_val = stats.ttest_ind(g0[col].dropna(), g1[col].dropna(), equal_var=False)
            test_name = "Welch's t-test"

        else:
            _, p_val = stats.mannwhitneyu(g0[col].dropna(), g1[col].dropna(), alternative='two-sided')
            test_name = "Mann-Whitney U test"

        results.append({
            'Biến': col,
            'Kiểm định': test_name,
            'P-value': p_val
        })

    for col in categorical:
        cont_table = pd.crosstab(df[col], df[target])
        _, p_val, _, _ = stats.chi2_contingency(cont_table)

        results.append({
            'Biến': col,
            'Kiểm định': "Chi-square",
            'P-value': p_val
        })

    return pd.DataFrame(results)

def plot_biochemical_distributions(df, continuous, target, output_dir):
    print("[+] Đang vẽ biểu đồ Split-Violin và Box cho các chỉ số sinh hóa")

    plt.figure(figsize=(15, 5))

    for i, col in enumerate(continuous, 1):
        plt.subplot(1, len(continuous), i)
        sns.violinplot(x=target, y=col, data=df, inner='quartile', split=True, hue=target, palette='Set2')
        sns.boxplot(x=target, y=col, data=df, width=0.1, color='white', boxprops={'zorder': 2})
        plt.title(f"{col} distribution")

    plt.tight_layout()
    plt.savefig(output_dir / 'split_violin_plots.png', dpi=300)
    plt.close()
