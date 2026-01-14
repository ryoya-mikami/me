import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 日本語フォント設定
rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Hiragino Sans']
rcParams['axes.unicode_minus'] = False

# CSVファイルを読み込み
csv_path = r"c:\Users\ryoya\OneDrive\Documents\課題\プログラミング\me\week8\bike.csv"
df = pd.read_csv(csv_path, encoding='utf-8')

print("データの形状:", df.shape)
print("\nカラム名:")
print(df.columns.tolist())
print("\n最初の5行:")
print(df.head())

# 数値データの統計を確認
print("\n数値カラムの統計:")
print(df.describe())

# ジニ係数を計算する関数
def gini_coefficient(data):
    """
    ジニ係数を計算する
    
    Parameters:
    data: numpy array or list - 所得/資産データ
    
    Returns:
    float - ジニ係数（0-1）
    """
    # データをソート
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # ジニ係数の計算
    # G = (2 * Σ(i * y_i)) / (n * Σy_i) - (n + 1) / n
    cumsum = np.cumsum(sorted_data)
    return (2 * np.sum(np.arange(1, n + 1) * sorted_data)) / (n * np.sum(sorted_data)) - (n + 1) / n


# ローレンツ曲線用のデータを計算する関数
def lorenz_curve(data):
    """
    ローレンツ曲線のデータを計算する
    
    Parameters:
    data: numpy array or list - 所得/資産データ
    
    Returns:
    tuple - (累積人口比率, 累積資産比率)
    """
    sorted_data = np.sort(data)
    cumsum = np.cumsum(sorted_data)
    n = len(sorted_data)
    
    # 累積比率を計算
    cumulative_population = np.arange(1, n + 1) / n
    cumulative_wealth = cumsum / cumsum[-1]
    
    # 最初に(0, 0)を追加
    cumulative_population = np.insert(cumulative_population, 0, 0)
    cumulative_wealth = np.insert(cumulative_wealth, 0, 0)
    
    return cumulative_population, cumulative_wealth


# データ分析：被害者の年齢別の盗難件数
print("\n=== 被害者年齢別の盗難件数 ===")
age_counts = df['被害者の年齢'].value_counts().sort_index()
print(age_counts)

# 各年齢グループの件数でジニ係数を計算
age_distribution = age_counts.values
gini = gini_coefficient(age_distribution)
print(f"\n年齢別分布のジニ係数: {gini:.4f}")

# ローレンツ曲線を計算
pop_ratio, wealth_ratio = lorenz_curve(age_distribution)

# グラフの作成
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ローレンツ曲線
ax1.plot(pop_ratio, wealth_ratio, 'b-', linewidth=2, label='ローレンツ曲線')
ax1.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='完全平等線')
ax1.fill_between(pop_ratio, wealth_ratio, pop_ratio, alpha=0.3)
ax1.set_xlabel('累積人口比率', fontsize=12)
ax1.set_ylabel('累積盗難件数比率', fontsize=12)
ax1.set_title(f'ローレンツ曲線 (ジニ係数: {gini:.4f})', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# 年齢別の盗難件数の分布
ax2.bar(age_counts.index.astype(str), age_counts.values, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('被害者年齢', fontsize=12)
ax2.set_ylabel('盗難件数', fontsize=12)
ax2.set_title('被害者年齢別の自転車盗難件数', fontsize=14, fontweight='bold')
ax2.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(r"c:\Users\ryoya\OneDrive\Documents\課題\プログラミング\me\week8\gini_lorenz_analysis.png", dpi=300, bbox_inches='tight')
print(f"\nグラフを保存しました: gini_lorenz_analysis.png")

# 管轄警察署別の分析も実施
print("\n=== 管轄警察署別の盗難件数 ===")
police_counts = df['管轄警察署（発生地）'].value_counts()
print(police_counts)

gini_police = gini_coefficient(police_counts.values)
print(f"\n警察署別分布のジニ係数: {gini_police:.4f}")

pop_ratio_p, wealth_ratio_p = lorenz_curve(police_counts.values)

# 警察署別のローレンツ曲線
fig2, ax3 = plt.subplots(figsize=(10, 8))
ax3.plot(pop_ratio_p, wealth_ratio_p, 'g-', linewidth=2.5, label='ローレンツ曲線')
ax3.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='完全平等線')
ax3.fill_between(pop_ratio_p, wealth_ratio_p, pop_ratio_p, alpha=0.3, color='green')
ax3.set_xlabel('累積警察署数比率', fontsize=12)
ax3.set_ylabel('累積盗難件数比率', fontsize=12)
ax3.set_title(f'警察署別分布のローレンツ曲線 (ジニ係数: {gini_police:.4f})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)

plt.tight_layout()
plt.savefig(r"c:\Users\ryoya\OneDrive\Documents\課題\プログラミング\me\week8\gini_police_analysis.png", dpi=300, bbox_inches='tight')
print(f"警察署別グラフを保存しました: gini_police_analysis.png")

# 結果のサマリー
print("\n" + "="*50)
print("分析結果サマリー")
print("="*50)
print(f"総盗難件数: {len(df)}")
print(f"\n【年齢別分布】")
print(f"  ジニ係数: {gini:.4f}")
print(f"  解釈: {'平等' if gini < 0.3 else '中程度' if gini < 0.6 else '不平等'}")
print(f"\n【警察署別分布】")
print(f"  ジニ係数: {gini_police:.4f}")
print(f"  解釈: {'平等' if gini_police < 0.3 else '中程度' if gini_police < 0.6 else '不平等'}")
print("="*50)

plt.show()
