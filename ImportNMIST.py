import numpy as np
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt

def importdata(train_size):
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]
    X = X / 255.0
    y.astype(int)

    digits = 10
    examples = y.shape[0]

    y = y.reshape(1, examples)

    y_new = np.eye(digits)[y.astype('int32')]
    y_new = y_new.T.reshape(digits, examples)
    y = y_new

    m_test = X.shape[0] - train_size

    X_train, X_test = X[:train_size].T, X[train_size:].T
    y_train, y_test = y_new[:, :train_size], y_new[:, train_size:]

    shuffle_index = np.random.permutation(train_size)
    X_train, y_train = X_train[:, shuffle_index], y_train[:, shuffle_index]

    return X_train, X_test, y_train, y_test, digits

def addNoiseAdversarials(X, y):
    new = (1/255) * np.random.randint(low=-35, high=35, size=X.shape)
    X = np.hstack((X, X+new))
    y = np.hstack((y, y))
    return X, y