import numpy as np


class MultivariateLinearRegression:
    def __init__(self):
        self.beta = None

    def fit(self, data, label):
        """
        训练多元线性回归模型。

        参数:
        X -- 输入特征数据，尺寸为 (n_samples, n_features)
        y -- 目标值，尺寸为 (n_samples,)
        """
        # 在X前添加一列1，用于计算截距项
        X_b = np.hstack([np.ones((data.shape[0], 1)), data])
        # 使用最小二乘法计算模型参数
        self.beta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ label

    def predict(self, input_data):
        """
        使用训练好的模型进行预测。

        参数:
        X -- 输入特征数据，尺寸为 (n_samples, n_features)

        返回:
        预测值，尺寸为 (n_samples,)
        """
        X_b = np.hstack([np.ones((input_data.shape[0], 1)), input_data])
        return X_b @ self.beta


if __name__ == "__main__":
    datas = np.array([
        [2104, 5],
        [1416, 3],
        [1534, 3],
        [852, 2]
    ])
    labels = np.array([400, 232, 315, 178])

    # 创建模型实例
    model = MultivariateLinearRegression()

    # 训练模型
    model.fit(datas, labels)

    # 使用模型进行预测
    predictions = model.predict(datas)
    print("预测值:", predictions)
