import numpy as np
import random
from kmeans.plot import draw_clusters


class KMeans_dataset:
    def __init__(self, args):
        self.args = args
        self.random_dataset = args['random']
        self.data = None
        self.label = None # no use (kmeans is unsupervised learning)

    def get_dataset(self) -> np.ndarray:
        if self.random_dataset: self.generate_random_data()
        return self.data

    def read_data(self):
        pass

    def generate_random_data(self, show: bool = True):
        """ generate _dataset by config file
        """
        kernel = self.args['kernel']
        size = self.args['size']
        data_value_min = self.args['range']['min']
        data_value_max = self.args['range']['max']
        radius = self.args['radius']

        data = [[] for _ in range(kernel)]
        for i in range(kernel):
            # random kernel x,y
            cur_kernel_pos_x = random.uniform(data_value_min, data_value_max)
            cur_kernel_pos_y = random.uniform(data_value_min, data_value_max)
            # generate random data with random kernel
            for _ in range(size):
                x = random.uniform(-radius, radius)
                y = random.uniform(-radius, radius)
                data[i].append([x+cur_kernel_pos_x, y+cur_kernel_pos_y])
        data = np.array(data, dtype=np.float32)
        if show: draw_clusters(data, kernel)
        shape = data.shape
        self.data = data.reshape(shape[0] * shape[1], shape[2])
