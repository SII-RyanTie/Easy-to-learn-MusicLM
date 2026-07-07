import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 路径配置
# ==========================================
# 这里填入你上一步生成的 CSV 路径
csv_path = "/inspire/hdd/project/powersystem/tieruian-253108120115/chenxie/music_project/eval/mulan_sim/MTG_ground_truth_MSMlabel.csv"

# 直方图保存路径 (保存在同一个目录下)
output_img = os.path.join(os.path.dirname(csv_path), "similarity_distribution_MSMlabel.png")

# ==========================================
# 2. 读取数据与计算平均值
# ==========================================
print(f"📖 正在读取数据: {csv_path}")
df = pd.read_csv(csv_path)

# 提取 similarity 这一列
similarities = df['similarity']

# 计算平均值、最大值、最小值
mean_sim = similarities.mean()
max_sim = similarities.max()
min_sim = similarities.min()

print("\n" + "="*30)
print(f"📊 统计结果 (共 {len(similarities)} 条数据):")
print(f"🔹 平均相似度 (Mean): {mean_sim:.4f}")
print(f"🔹 最高相似度 (Max):  {max_sim:.4f}")
print(f"🔹 最低相似度 (Min):  {min_sim:.4f}")
print("="*30 + "\n")

# ==========================================
# 3. 绘制直方图并保存
# ==========================================
print("🎨 正在绘制分布直方图...")

# 设置图片大小和清晰度
plt.figure(figsize=(10, 6), dpi=300)

# 使用 seaborn 画直方图，加上 KDE (核密度估计曲线) 让分布更平滑好看
sns.histplot(similarities, bins=50, kde=True, color='#4CB391', edgecolor='black', alpha=0.7)

# 画一条红色的虚线代表平均值，方便肉眼定位
plt.axvline(mean_sim, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_sim:.4f}')

# 设置标题和坐标轴标签
plt.title('Distribution of MuLan Similarity Scores', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Similarity Score', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)

# 添加图例和网格线
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 紧凑布局并保存图片
plt.tight_layout()
plt.savefig(output_img)
plt.close()

print(f"🎉 直方图已成功保存至: {output_img}")
