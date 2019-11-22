import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def activation(z):
    s = np.exp(z) / np.sum(np.exp(z), axis=0)
    return s


def ReLU(z):
    for i in range(z.shape[0]) :
        for j in range(z.shape[1]):
            if(z[i,j] <= 0):
                z[i,j] = 0
    return z


def step(z):
    return np.heaviside(z, 1)


def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1. / m) * L_sum
    return L


def compute_loss_ReLU(Y, Y_hat):
    L_sum = np.sum(np.power(Y_hat - Y, 2))
    m = Y.shape[1]
    return (1./m) * L_sum


def feed_forward(X, params):
    cache = {}

    cache["Z1"] = np.matmul(params["W0"], X) + params["b0"]
    cache["X1"] = sigmoid(cache["Z1"])
    cache["Z2"] = np.matmul(params["W1"], cache["X1"]) + params["b1"]
    cache["X2"] = sigmoid(cache["Z2"])
    cache["Z3"] = np.matmul(params["W2"], cache["X2"]) + params["b2"]
    cache["X3"] = sigmoid(cache["Z3"])# np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)
    return cache


def feed_forward_ReLU(X, params):
    cache = {}

    cache["Z1"] = (np.matmul(params["W0"], X) + params["b0"])
    cache["X1"] = ReLU(cache["Z1"])
    cache["Z2"] = (np.matmul(params["W1"], cache["X1"]) + params["b1"])
    cache["X2"] = ReLU(cache["Z2"])
    cache["Z3"] = (np.matmul(params["W2"], cache["X2"]) + params["b2"])
    cache["X3"] = ReLU(cache["Z3"])  # np.exp(cache["Z3"]) / np.sum(np.exp(cache["Z3"]), axis=0)
    return cache


def back_propagate(X0, Y, params, cache, m):
    dZ3 = cache["X3"] - Y
    dW2 = (1./m) * np.matmul(dZ3, cache["X2"].T)
    db2 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)

    dX2 = np.matmul(params["W2"].T, dZ3)
    dZ2 = dX2 * sigmoid(cache["Z2"]) * (1 - sigmoid(cache["Z2"]))
    dW1 = (1./m) * np.matmul(dZ2, cache["X1"].T)
    db1 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dX1 = np.matmul(params["W1"].T, dZ2)
    dZ1 = dX1 * sigmoid(cache["Z1"]) * (1 - sigmoid(cache["Z1"]))
    dW0 = (1./m) * np.matmul(dZ1, X0.T)
    db0 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW0": dW0, "db0": db0,
             "dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2}

    return grads


def back_propagate_ReLU(X0, Y, params, cache, m):
    dZ3 = np.multiply(cache["X3"] - Y, step(cache["Z3"]))
    dW2 = (1./m) * np.matmul(dZ3, cache["X2"].T)
    db2 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)

    dX2 = np.matmul(params["W2"].T, dZ3)
    dZ2 = np.multiply(dX2, step(cache["Z2"]))
    dW1 = (1./m) * np.matmul(dZ2, cache["X1"].T)
    db1 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

    dX1 = np.matmul(params["W1"].T, dZ2)
    dZ1 = np.multiply(dX1, step(cache["Z1"]))
    dW0 = (1./m) * np.matmul(dZ1, X0.T)
    db0 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW0": dW0, "db0": db0,
             "dW1": dW1, "db1": db1,
             "dW2": dW2, "db2": db2,}

    return grads

