# -*- coding: utf-8 -*-
# @Author: Crookery

import numpy as np
import logging


def mse_score(y_predict, y_test):
    """ MSE Loss Function """
    mse = np.mean((y_predict - y_test) ** 2)
    return mse


class LinearRegression:
    def __init__(self):
        """ Multi Linear Regression Model, theta = (w, b) """
        self.theta = None

    def fit_normal(self, train_data: np.ndarray, train_label: np.ndarray):
        x = np.hstack([np.ones((len(train_data), 1)), train_data])
        # theta = (X X^T)^{-1} X y
        self.theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(train_label)
        return self.theta

    def predict(self, test_data: np.ndarray):
        x = np.hstack([np.ones((len(test_data), 1)), test_data])
        return x.dot(self.theta)


def main():
    # logging config
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

    # train data
    train_data = np.array([[1], [2], [3], [4]])
    train_label = np.array([2, 4, 5, 7])
    # test data
    test_data = np.array([[5], [6]])
    test_label = np.array([8, 10])

    # model
    model = LinearRegression()
    model.fit_normal(train_data, train_label)
    predictions = model.predict(test_data)

    mse = mse_score(predictions, test_label)
    logging.info(f"Model parameters: {model.theta}")
    logging.info(f"Predictions: {predictions}")
    logging.info(f"Mean Squared Error (MSE): {mse}")


if __name__ == '__main__':
    main()
