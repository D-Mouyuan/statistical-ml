import numpy as np


class OneLogisticRegression:
    def __init__(self, lr=0.01, num_iter=1000):
        self.lr = lr
        self.num_iter = num_iter

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, data, label):
        # init args
        self.omega = 0
        self.b = 0

        for i in range(self.num_iter):
            # 计算线性预测值
            z = self.omega * data + self.b
            h = self.sigmoid(z)
            # 计算梯度
            omega_gradient = np.dot(data.T, (h - label)) / label.size
            b_gradient = np.sum(h - label) / label.size
            # 更新参数
            self.omega -= self.lr * omega_gradient
            self.b -= self.lr * b_gradient

            # 可选：打印损失
            if i % 100 == 0:
                loss = -label * np.log(h) - (1 - label) * np.log(1 - h)
                print(f'Loss at iteration {i}: {np.mean(loss)}')

    def predict_prob(self, X):
        # 预测实例的概率
        return self.sigmoid(self.omega * X + self.b)

    def predict(self, X, threshold=0.5):
        # 预测标签
        return self.predict_prob(X) >= threshold


if __name__ == "__main__":
    # 生成模拟数据
    data = np.random.randn(100)
    label = np.random.randint(0, 2, 100)

    # 创建并训练模型
    model = OneLogisticRegression()
    model.fit(data, label)

    # 预测新数据
    predict = model.predict(data)

    # 计算准确率
    accuracy = (predict == label).mean()
    print(f'Accuracy: {accuracy}')
