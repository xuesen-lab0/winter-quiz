'''
Author: xuesen-lab0 kfsaghak19@gmail.com
Date: 2025-02-25 10:46:16
LastEditors: xuesen-lab0 kfsaghak19@gmail.com
LastEditTime: 2025-02-25 14:23:04
FilePath: \winter-quiz\tSNE.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
    perplexity=70,  
    learning_rate=200, 
    n_iter=1000,  # 迭代次数
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