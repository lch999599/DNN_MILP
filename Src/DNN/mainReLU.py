from Src.Data.ImportNMIST import *
from Src.supportingFunctions import *
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os, shutil
from math import sqrt
from Src.MILP.Adversarials import *
import time
import copy

print("loading data...")
Xprimary, X_test, yprimary, y_test, digits = loadData(60000)
print(Xprimary.shape, X_test.shape)
print("data loaded.")
print("Initializing...")
learning_rate = 0.75
K = 2

n_x = Xprimary.shape[0]
n_h = 64
m = Xprimary.shape[1]
beta = 0.1
batch_size = 128
batches = -(-m//batch_size)
fakeList = [6,7,8,5,9,3,5,9,3,4]

for p in range(0,1):
    print('training ', p)
    message = ''
    X_train = copy.copy(Xprimary)
    y_train = copy.copy(yprimary)
    # initialization
    W = [np.random.randn(n_h, n_x)*sqrt(1./n_x)]
    b = [np.zeros((n_h, 1))]
    for i in range(1, K-1):
        W.append(np.random.randn(n_h, n_h)*sqrt(1./n_h))
        b.append(np.zeros((n_h, 1)))
    W.append(np.random.randn(digits, n_h)*sqrt(1./n_h))
    b.append(np.zeros((digits, 1)))

    V_dW, V_db = [], []
    for i in range(0, K):
        V_dW.append(np.zeros(W[i].shape))
        V_db.append(np.zeros(b[i].shape))

    print("Starting training...")
    adversarials = False
    for i in range(9):
        for j in range(batches):
            begin = j * batch_size
            end = min(begin + batch_size, X_train.shape[1] - 1)
            X0 = X_train[:, begin:end]
            Y = y_train[:, begin:end]
            m_batch = end - begin

            Z, X = feed_forward_ReLU(X0, W, b, K)
            dW, db = back_propagate_ReLU(X, Y, W, b, Z, K, m_batch)

            for k in range(0, K):
                V_dW[k] = beta * V_dW[k] + (1. - beta) * dW[k]
                V_db[k] = beta * V_db[k] + (1. - beta) * db[k]
                W[k] = W[k] - learning_rate * V_dW[k]
                b[k] = b[k] - learning_rate * V_db[k]

        X_train = X_train[:, 0:100]
        y_train = y_train[:, 0:100]
        permutation = np.random.permutation(X_train.shape[1])
        X_train = X_train[:, permutation]
        y_train = y_train[:, permutation]

        Z, X = feed_forward_ReLU(X_train, W, b, K)
        predictions = np.argmax(X[K], axis=0)
        labels = np.argmax(y_train, axis=0)
        message += "training accuracy is: " + str(accuracy_score(predictions, labels)) + '\n'
        print("training accuracy is: ", accuracy_score(predictions, labels))
        if adversarials and i==4:
            for k in range(0, m):
                if predictions[k] == labels[k]:
                    X_train, y_train = GenerateAdversarial(K, W, b, X_train, y_train, k, predictions[k], fakeList[labels[k]])
        Z_tested, X_tested = feed_forward_ReLU(X_test, W, b, K)
        test_cost = compute_loss_ReLU(y_test, X_tested[K])
        predictions = np.argmax(X_tested[K], axis=0)
        labels = np.argmax(y_test, axis=0)
        print("Epoch {}: test cost = {}".format(i + 1, test_cost))
        message += "testing accuracy is: " + str(accuracy_score(predictions, labels)) + '\n'
        print("testing accuracy is: ", accuracy_score(predictions, labels))


    print("Training done. Printing results:")

    Z_tested, X_tested = feed_forward_ReLU(X_test, W, b, K)
    predictions = np.argmax(X_tested[K], axis=0)
    labels = np.argmax(y_test, axis=0)

    print(classification_report(predictions, labels))

    print('deleting old matrices...')
    folder = r'C:\Users\marie\Documents\Software\BEP\Src\DNN\ReLUTrain\{}_layers'.format(K)
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    print("Saving weights and biases...")
    for i in range(0, K):
        np.save("ReLUTrain/{}_layers/W{}".format(K, i), W[i])
        np.save("ReLUTrain/{}_layers/b{}".format(K, i), b[i])

    timelab = time.time()

    message += str(classification_report(predictions, labels))
    message += "\n" + "accuracy of network is " + str(accuracy_score(predictions, labels))
    message += "\n trained with {} hidden layers of size {} with learning rate of {}".format(K, n_h, learning_rate)
    text_file = open("ReLuTrain/{}_layers_outputs/output {} {}.txt".format(K, timelab, str(adversarials)), "w")
    text_file.write(message)
    text_file.close()

    print(confusion_matrix(predictions, labels))

    print("done.")


