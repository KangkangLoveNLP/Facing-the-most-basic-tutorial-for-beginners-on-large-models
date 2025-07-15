import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体为微软雅黑
# 定义禁带宽度
Eg = 1.17  # eV


# 定义高对称点的位置
high_symmetry_points = {
    'Γ': 0,
    'X': 2,
    'L': 4,
    'U': 6,
    'K': 8
}

# 定义能带数据（示例数据）
k_points = np.linspace(0, 8, 100)
valence_band = -2 - np.sin(k_points)
conduction_band = 2 + np.sin(k_points)

# 绘制能带图
plt.figure(figsize=(10, 6))
plt.plot(k_points, valence_band, label='价带', color='blue')
plt.plot(k_points, conduction_band, label='导带', color='red')

# 标注高对称点
for point, position in high_symmetry_points.items():
    plt.axvline(x=position, color='black', linestyle='--', alpha=0.5)
    plt.text(position, max(valence_band) + 1, point, ha='center')

# 绘制费米能级线
plt.axhline(y=0, color='black', linestyle='--', label='费米能级')

# 填充价带区域（T=0K时）
plt.fill_between(k_points, valence_band, min(valence_band) - 1, color='blue', alpha=0.2)

# 设置图例和标签
plt.title('硅能带结构在T=0K', fontsize=14)
plt.xlabel('波长（k）', fontsize=12)
plt.ylabel('Energy (eV)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()