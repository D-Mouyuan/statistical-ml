import numpy as np
from dataset import KMeans_dataset
from kmeans.plot import draw_clusters_map


class KMeans:
    def __init__(self, args):
        self.dataset = KMeans_dataset(args)
        self.data = self.dataset.get_dataset()

        # args
        self.kernel = args['kernel']
        self.max_iterations = args.get('max_iterations', 100)
        self.tol = args.get('tolerance', 0.0001)

        # run record
        self.centroids = []

    def run(self, draw_result: bool = True):
        # random init k kernels
        self.centroids = [self.data[i] for i in np.random.choice(range(len(self.data)), self.kernel, replace=False)]

        for i in range(self.max_iterations):
            classes = {c: [] for c in range(self.kernel)}
            # collect point
            for point in self.data:
                distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
                classification = distances.index(min(distances))    # find nearest kernel
                classes[classification].append(point)

            previous = np.array(self.centroids)

            # update kernels
            for classification in classes:
                self.centroids[classification] = np.average(classes[classification], axis=0)

            optimized = True
            for centroid in range(len(self.centroids)):
                original_centroid = previous[centroid]
                current = self.centroids[centroid]
                if np.sum((current - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                if draw_result: draw_clusters_map(classes)
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - centroid) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
