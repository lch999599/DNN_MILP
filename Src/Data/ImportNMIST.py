import numpy as np
from sklearn.datasets import fetch_openml

# Module used for importing MNIST data.

# Imports data from server. Run this function once to import images and labels once
# Saves data then to numpy arrays and saves them locally.
def importdata():
    mnist = fetch_openml('mnist_784')
    X, y = mnist["data"], mnist["target"]
    # normalize pixel values
    X = X / 255.0
    # correction for string labels
    y.astype(int)

    digits = 10

    # reshape arrays
    examples = y.shape[0]

    y = y.reshape(1, examples)

    y_new = np.eye(digits)[y.astype('int32')]
    y = y_new.T.reshape(digits, examples)

    # save arrays
    np.save("X", X)
    np.save("y", y)
    np.save("digits", digits)

# function for loading in data from saved arrays.
def loadData(train_size):
    X, y, digits= np.load("X.npy").T, np.load("y.npy"), np.load("digits.npy")

    # shuffle data
    shuffle_index = np.random.permutation(X.shape[1])
    X, y = X[:, shuffle_index], y[:, shuffle_index]

    # uncomment this line below for truly random sample
    # comment lines below
    # X_train, y_train = X[:,:train_size], y[:, :train_size]

    # take sample with all digits taken in an equal amount
    X_test, y_test = X[:,train_size:], y[:, train_size:]
    X_train, y_train = np.zeros((784, train_size)), np.zeros((10, train_size))
    averages = train_size*0.1*np.ones((10,1))
    index, count = 0, 0
    while(count<train_size):
        if averages[np.argmax(y[:,index])] > 0:
            averages[np.argmax(y[:,index])] -= 1
            X_train[:,count], y_train[:,count] = X[:,index], y[:,index]
            count += 1
        index += 1

    return X_train, X_test, y_train, y_test, digits

importdata()