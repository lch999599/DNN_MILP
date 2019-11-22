from ImportNMIST import *
from supportingFunctions import *
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


def main():
    X_train, X_test, y_train, y_test, digits = importdata(55000)

    learning_rate = 0.05

    n_x = X_train.shape[0]
    n_h = 64
    m = X_train.shape[1]
    beta = 0.25
    batch_size = 128
    batches = -(-m//batch_size)

    # initialization
    params = {"W0": np.random.randn(n_h, n_x) * np.sqrt(1. / n_x) ,
              "b0": np.zeros((n_h, 1)),
              "W1": np.random.randn(n_h, n_h)* np.sqrt(1. / n_h),
              "b1": np.zeros((n_h, 1)),
              "W2": np.random.randn(digits, n_h) * np.sqrt(1. / n_h),
              "b2": np.zeros((digits, 1))}

    V_dW0 = np.zeros(params["W0"].shape)
    V_db0 = np.zeros(params["b0"].shape)
    V_dW1 = np.zeros(params["W1"].shape)
    V_db1 = np.zeros(params["b1"].shape)
    V_dW2 = np.zeros(params["W2"].shape)
    V_db2 = np.zeros(params["b2"].shape)


    for i in range(3):
        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]
        Y_train_shuffled = y_train[:, permutation]

        for j in range(batches):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)
            X = X_train_shuffled[:, begin:end]
            Y = Y_train_shuffled[:, begin:end]
            m_batch = end - begin

            cache = feed_forward_ReLU(X, params)
            grads = back_propagate_ReLU(X, Y, params, cache, m_batch)

            V_dW0 = (beta * V_dW0 + (1. - beta) * grads["dW0"])
            V_db0 = (beta * V_db0 + (1. - beta) * grads["db0"])
            V_dW1 = (beta * V_dW1 + (1. - beta) * grads["dW1"])
            V_db1 = (beta * V_db1 + (1. - beta) * grads["db1"])
            V_dW2 = (beta * V_dW2 + (1. - beta) * grads["dW2"])
            V_db2 = (beta * V_db2 + (1. - beta) * grads["db2"])

            params["W0"] = params["W0"] - learning_rate * V_dW0
            params["b0"] = params["b0"] - learning_rate * V_db0
            params["W1"] = params["W1"] - learning_rate * V_dW1
            params["b1"] = params["b1"] - learning_rate * V_db1
            params["W2"] = params["W2"] - learning_rate * V_dW2
            params["b2"] = params["b2"] - learning_rate * V_db2

        cache = feed_forward_ReLU(X_train, params)
        train_cost = compute_loss_ReLU(y_train, cache["X3"])
        cache = feed_forward_ReLU(X_test, params)
        test_cost = compute_loss_ReLU(y_test, cache["X3"])
        predictions = np.argmax(cache["X3"], axis=0)
        labels = np.argmax(y_test, axis=0)
        print("accuracy is: ", accuracy_score(predictions, labels))
        print("Epoch {}: training cost = {}, test cost = {}".format(i + 1, train_cost, test_cost))

    print("Done.")

    cache = feed_forward_ReLU(X_test, params)
    predictions = np.argmax(cache["X3"], axis=0)
    labels = np.argmax(y_test, axis=0)

    print(classification_report(predictions, labels))

    return params, cache
