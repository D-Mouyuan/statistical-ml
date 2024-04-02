import os
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from DecisionTreeClassifier import DecisionTreeClassifier
from utils.plot import plot_decision_boundary

os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"))


def read_config(file_name: str = 'decision_tree.yaml') -> dict:
    """ read configuration """
    file = open(file_name, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    cfg = yaml.load(file_data, Loader=yaml.FullLoader)
    return cfg


def get_dataset():
    """ use sklearn to generate dataset """
    train_dataset, test_dataset = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    return train_dataset, test_dataset


def split_dataset(train_dataset, test_dataset):
    """ split data to data and label """
    train_data, test_data, train_label, test_label = \
        train_test_split(train_dataset, test_dataset, test_size=0.25, random_state=42)
    # show raw data
    plt.scatter(train_dataset[:, 0], train_dataset[:, 1], c=test_dataset, cmap='rainbow', edgecolor='k', s=20)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Generated Classification Data')
    plt.show()
    return train_data, test_data, train_label, test_label


def train(model, train_datas, train_labels):
    model.fit(train_datas, train_labels)


def evaluate(model, test_datas, test_labels):
    y_pred = model.predict(test_datas)
    accuracy = np.sum(y_pred == test_labels) / len(test_labels)
    print(f"model accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    # args
    args = read_config()

    # dataset
    train_dataset, test_dataset = get_dataset()
    train_datas, test_datas, train_labels, test_labels = split_dataset(train_dataset, test_dataset)

    # run
    learner = DecisionTreeClassifier(max_depth=3, criterion="gini")
    train(learner, train_datas, train_labels)
    evaluate(learner, test_datas, test_labels)

    # plot decision boundary
    plot_decision_boundary(learner, train_dataset, test_dataset)
