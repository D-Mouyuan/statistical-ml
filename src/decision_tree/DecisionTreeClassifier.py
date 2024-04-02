import numpy as np


class DecisionTreeClassifier:
    def __init__(self, max_depth=None, criterion="gini"):
        self.max_depth = max_depth
        self.criterion = criterion
        self.tree = None

    class Node:
        """ inner class """
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if len(np.unique(y)) == 1:
            return self.Node(value=np.unique(y)[0])
        if depth >= self.max_depth:
            return self.Node(value=self._most_common_label(y))

        best_feature, best_threshold = self._best_split(X, y, num_samples, num_features)
        if best_feature is None:
            return self.Node(value=self._most_common_label(y))

        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return self.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, num_samples, num_features):
        best_metric = float("inf") if self.criterion == "gini" else -float("inf")
        split_idx, split_threshold = None, None
        for feature_index in range(num_features):
            X_column = X[:, feature_index]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                metric = self._calculate_metric(y, X_column, threshold)
                is_better_split = metric < best_metric if self.criterion == "gini" else metric > best_metric
                if is_better_split:
                    best_metric = metric
                    split_idx = feature_index
                    split_threshold = threshold
        return split_idx, split_threshold

    def _calculate_metric(self, y, X_column, threshold):
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return float("inf") if self.criterion == "gini" else -float("inf")

        if self.criterion == "gini":
            metric = self._gini(y[left_idxs]) + self._gini(y[right_idxs])
        else:  # entropy
            metric = self._information_gain(y, X_column, threshold)
        return metric

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - np.sum(probabilities ** 2)

    def _information_gain(self, y, X_column, split_threshold):
        parent_metric = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_threshold)
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_metric = (n_l / n) * e_l + (n_r / n) * e_r
        ig = parent_metric - child_metric
        return ig

    @staticmethod
    def _entropy(y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities))

    def _split(self, X_column, split_threshold):
        left_idxs = np.where(X_column <= split_threshold)[0]
        right_idxs = np.where(X_column > split_threshold)[0]
        return left_idxs, right_idxs

    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
