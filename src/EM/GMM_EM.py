import numpy as np


def gaussian_pdf(x, mean, var):
    """ 高斯概率密度函数 """
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))


class EM:
    def __init__(
        self,
        K: int = 2,
        Mu: np.ndarray = None,
        Sigma: np.ndarray = None,
        Alpha: np.ndarray = None,
        N: int = 0,
        sample_data: np.ndarray = None,
        max_iter_step: int = 1000,
        epsilon: float = 0.001,
        debug: bool = False
    ):
        # GMM
        self.K = K                              # 多少个高斯分布
        self.Mu = Mu                            # 第 k 个高斯分布的初始均值
        self.Sigma = Sigma                      # 第 k 个高斯分布的初始标准差
        self.Alpha = Alpha                      # 第 k 个高斯分布的占比

        # Sample data
        self.N = N                              # 样本量
        self.data = sample_data                 # 样本数据
        self.gamma = np.zeros((self.N, self.K)) # 隐变量 γ

        # train setting
        self._max_iter_step = max_iter_step     # 最高迭代次数
        self._threshold = epsilon               # 误差阈值，低于则退出
        self._debug = debug                     # debug

    def e_step(self):
        """ E步：补全隐变量信息 """
        for i in range(self.K):
            self.gamma[:, i] = self.Alpha[i] * gaussian_pdf(self.data, self.Mu[i], self.Sigma[i])
        self.gamma /= np.sum(self.gamma, axis=1, keepdims=True)

    def m_step(self):
        """ M步：更新高斯分布的参数和混合系数 """
        for i in range(self.K):
            weight = np.sum(self.gamma[:, i])
            self.Mu[i] = np.dot(self.gamma[:, i], self.data) / weight
            self.Sigma[i] = np.dot(self.gamma[:, i], (self.data - self.Mu[i]) ** 2) / weight
            self.Alpha[i] = weight / self.N

    def run(self):
        """ run EM algorithm """
        step = 0
        for step in range(self._max_iter_step):
            old_Mu = self.Mu.copy()
            self.e_step()
            self.m_step()

            if np.linalg.norm(self.Mu - old_Mu) < self._threshold:
                if self._debug: print(f"Converged at iteration {step + 1}")
                break

        # 限定步数内未收敛
        if self._debug and step + 1 == self._max_iter_step:
            print("Reached maximum iterations without convergence.")


if __name__ == '__main__':
    cur_K = 2
    cur_Mu = np.array([0.25, 2.5])
    cur_Sigma = np.array([0.125, 1.67])
    cur_Alpha = np.array([0.5, 0.5])
    cur_sample_data = np.array([0, 0.5, 1, 2, 3, 4])
    cur_N = len(cur_sample_data)

    em = EM(K=cur_K, Mu=cur_Mu, Sigma=cur_Sigma, Alpha=cur_Alpha, N=cur_N, sample_data=cur_sample_data, debug=True)
    em.run()
