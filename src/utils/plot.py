import matplotlib.pyplot as plt


def draw_clusters(data, kernels, save: bool = False, save_prefix: str = '_'):
    """ draw clusters with kernels in different colors by matplotlib
    """
    for i in range(kernels):
        plt.scatter(data[i][:, 0], data[i][:, 1], color='m')
    plt.show(block=False)
    if save: plt.savefig(save_prefix + 'clusters.png')


def draw_clusters_map(data: dict):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for k in data:
        for point in data[k]:
            plt.scatter(point[0], point[1], color=colors[k])
    plt.show(block=False)
