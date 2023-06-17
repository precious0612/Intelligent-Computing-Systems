import matplotlib.pyplot as plt
import seaborn as sns

# 数据
methods = ['numpy.dot', 'numpy.matmul', 'scipy.linalg.blas', 'PyTorch CPU', 'PyTorch GPU', 'TensorFlow GPU', 'C++23', 'C11']
times = [295.2202699184418, 304.445191860199, 292.08862113952637, 0.08031201362609863, 0.48150014877319336, 0.27233099937438965, 104.3, 1.732]

# 创建颜色梯度
cmap = plt.get_cmap("coolwarm")
norm = plt.Normalize(min(times), max(times))
colors = cmap(norm(times))

# 创建图表
plt.figure(figsize=(10,6))
barplot = sns.barplot(x=times, y=methods, palette=colors)

# 设置标题和坐标轴标签
plt.title('Time taken by different methods to run VGG19')
plt.xlabel('Time (s)')
plt.ylabel('Methods')

# 显示图表
plt.show()
