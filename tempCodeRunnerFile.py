: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载数据
file_path = r"C:\\Users\\xuwt\\Desktop\\winter-quiz\\embeddings.txt"
embeddings = np.loadtxt(file_path)

# 初始化 t-SNE 模型
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)

# 进行降维
embeddings_2d = tsne.fit_transform(embeddings)

# 可视化结果
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, c='blue')
plt.title('t-SNE Visualization of Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True)
plt.show()