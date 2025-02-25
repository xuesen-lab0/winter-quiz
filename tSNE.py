import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 确保文件路径正确
file_path = r"C:\Users\xuwt\Desktop\winter-quiz\embeddings.txt"

# 加载数据
try:
    embeddings = np.loadtxt(file_path)
    print(f"Number of samples: {embeddings.shape[0]}")  # 打印样本数量
except FileNotFoundError:
    print(f"文件未找到，请检查路径：{file_path}")
    exit()

# 数据标准化
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

# 初始化 t-SNE 模型
tsne = TSNE(
    n_components=2, 
    random_state=42, 
    perplexity=50,  
    learning_rate=500, 
    n_iter=3000,  # 迭代次数
    early_exaggeration=12
)

# 进行降维
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    alpha=0.5,  # 调整透明度
    c='blue', 
    cmap='viridis', 
    s=20)  # 调整点大小
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()