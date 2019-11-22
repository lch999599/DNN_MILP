from Src.Data.ImportNMIST import *
from Src.supportingFunctions import *
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

X_train, X_test, y_train, y_test, digits = importdata(55000)

learning_rate = 0.05
K = 4

n_x = X_train.shape[0]
n_h = 64
m = X_train.shape[1]
beta = 0.25
batch_size = 128
batches = -(-m//batch_size)

# initialization
W = [np.random.randn(n_h, n_x) * np.sqrt(1. / n_x)]
b = [np.zeros((n_h, 1))]
for i in range(1, K-1):
    W.append(np.random.randn(n_h, n_h) * np.sqrt(1. / n_h))
    b.append(np.zeros((n_h, 1)))
W.append(np.random.randn(digits, n_h) * np.sqrt(1. / n_h))
b.append(np.zeros((digits, 1)))

V_dW, V_db = [], []
for i in range(0, K):
    V_dW.append(np.zeros(W[i].shape))
    V_db.append(np.zeros(b[i].shape))


for i in range(3):
    permutation = np.random.permutation(X_train.shape[1])
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = y_train[:, permutation]

    for j in range(batches):
        begin = j * batch_size
        end = min(begin + batch_size, X_train.shape[1] - 1)
        X0 = X_train_shuffled[:, begin:end]
        Y = Y_train_shuffled[:, begin:end]
        m_batch = end - begin

        Z, X = feed_forward_ReLU(X0, W, b, K)
        dW, db = back_propagate_ReLU(X, Y, W, b, Z, K, m_batch)

        for k in range(0, K):
            V_dW[k] = (beta * V_dW[k] + (1. - beta) * dW[k])
            V_db[k] = (beta * V_db[k] + (1. - beta) * db[k])
            W[k] = W[k] - learning_rate * V_dW[k]
            b[k] = b[k] - learning_rate * V_db[k]

    Z, X = feed_forward_ReLU(X_train, W, b, K)
    train_cost = compute_loss_ReLU(y_train, X[K])
    Z_tested, X_tested = feed_forward_ReLU(X_test, W, b, K)
    test_cost = compute_loss_ReLU(y_test, X_tested[K])
    predictions = np.argmax(X_tested[K], axis=0)
    labels = np.argmax(y_test, axis=0)
    print("accuracy is: ", accuracy_score(predictions, labels))
    print("Epoch {}: training cost = {}, test cost = {}".format(i + 1, train_cost, test_cost))

print("Done.")

Z_tested, X_tested = feed_forward_ReLU(X_test, W, b, K)
predictions = np.argmax(X_tested[K], axis=0)
labels = np.argmax(y_test, axis=0)

print(classification_report(predictions, labels))

for i in range(0, K):
    np.savetxt("ReLUTrain/W{}".format(i), W[i], delimiter=',')
    np.savetxt("ReLUTrain/b{}".format(i), b[i], delimiter=',')