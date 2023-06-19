from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# 加载iris数据集
iris = load_iris()

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(iris.data)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, iris.target, test_size=0.2, random_state=42)

# 定义自动编码器模型
autoencoder = MLPRegressor(hidden_layer_sizes=(2,), activation='relu', solver='adam')

# 训练自动编码器
autoencoder.fit(X_train, X_train)

# 提取特征
encoder = autoencoder.hidden_layer_sizes(X_train)

# 输出特征
# print(encoder)

import matplotlib.pyplot as plt

# 绘制特征图
plt.scatter(encoder[:, 0], encoder[:, 1], c=y_train)
plt.title('Autoencoder Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()