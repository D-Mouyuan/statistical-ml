import matplotlib.pyplot as plt
import numpy as np


def draw_clusters(data, kernels, save: bool = False, save_prefix: str = '_'):
    """ [kmeans] draw clusters with kernels in different colors by matplotlib
    """
    for i in range(kernels):
        plt.scatter(data[i][:, 0], data[i][:, 1], color='m')
    plt.show(block=False)
    if save: plt.savefig(save_prefix + 'clusters.png')


def draw_clusters_map(data: dict):
    """ [kmeans] draw results
    """
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for k in data:
        for point in data[k]:
            plt.scatter(point[0], point[1], color=colors[k])
    plt.show(block=False)


def plot_decision_boundary(model, X, y):
    """ [decision tree classify] plot results
    """
    # set min,max value and margin line
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # predict full grid type
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot result
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()
