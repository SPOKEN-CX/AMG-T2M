import numpy as np
import matplotlib.pyplot as plt

# 数据
x_labels = ['30', '40', '50']
methods = ['MotionDiffuse', 'Momask', 'MotionGPT', 'T2M-GPT', 'Ours']
data = [
    [0.38, 0.35, 0.95],
    [0.06, 0.17, 0.57],
    [0.09, 0.21, 0.59],
    [0.07, 0.20, 0.60],
    [0.05, 0.13, 0.29],
]

# 转成 numpy 方便操作
data = np.array(data)  # shape: (5, 3)
x = np.arange(len(x_labels))  # [0, 1, 2]

# 颜色设置
colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

# 柱状图组间隔
bar_width = 0.15
fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    # 每个方法在 x 的基础上偏移 i * bar_width
    ax.bar(x + i * bar_width,
           data[i],
           width=bar_width,
           label=method,
           color=colors[i],
           edgecolor='black' if method == 'Ours' else 'none',  # Ours加黑色边框突出
           linewidth=1.5 if method == 'Ours' else 0.5)

    # 在柱子顶端加上数值
    for xi, yi in zip(x + i * bar_width, data[i]):
        ax.text(xi, yi + 0.02, f'{yi:.2f}',
                ha='center', va='bottom',
                fontsize=10)

ax.set_xticks(x + 2 * bar_width)  # 让 x 轴刻度对齐到中间
ax.set_xticklabels(x_labels, fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Text Length', fontsize=13)
ax.set_ylabel('FID ', fontsize=13)
ax.set_title('FID Performance ', fontsize=15, weight='bold')

ax.grid(True, linestyle='--', axis='y', alpha=0.5)
ax.legend(fontsize=11)

plt.tight_layout()
plt.show()
