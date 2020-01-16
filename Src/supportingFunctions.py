import numpy as np

# implements sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# implements ReLU function
def ReLU(z):
    for i in range(z.shape[0]) :
        for j in range(z.shape[1]):
            if(z[i,j] <= 0):
                z[i,j] = 0
    return z

# Implements heaviside function
def step(z):
    return np.heaviside(z, 1)

# Compute the loss for sigmoid activation with logistic regression
def compute_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1. / m) * L_sum
    return L

# Compute the loss for the ReLU activation with difference squared
def compute_loss_ReLU(Y, Y_hat):
    L_sum = np.sum(np.power(Y_hat - Y, 2))
    m = Y.shape[1]
    return (1./m) * L_sum

# Feeding the input forward the DNN with K layers using sigmoid acitvation
def feed_forward(X0, W, b, K):
    Z, X = [0], [X0]
    for i in range(0, K):
        Z.append(np.dot(W[i], X[i]) + b[i])
        X.append(sigmoid(Z[i+1]))
    return Z, X

# Feeding the input forward the DNN with K layers using ReLU acitvation
def feed_forward_ReLU(X0, W, b, K):
    Z, X = [0], [X0]
    for i in range(0, K):
        Z.append((W[i] @ X[i]) + b[i])
        X.append(ReLU(Z[i+1]))
    return Z, X

# back propagation algorithm for sigmoid activation
def back_propagate(X, Y, W, b, Z, K, m):
    dZ, dW, db, dX = [],[],[],[]
    for i in range(0, K+1):
        dZ.append(0), dW.append(0), db.append(0), dX.append(0)
    dZ[K] = X[K] - Y
    dW[K-1] = (1./m) * np.matmul(dZ[K], X[K-1].T)
    db[K-1] = (1./m) * np.sum(dZ[K], axis=1, keepdims=True)

    for i in reversed(range(1, K)):
        dX[i] = np.matmul(W[i].T, dZ[i+1])
        dZ[i] = dX[i] * sigmoid(Z[i]) * (1 - sigmoid(Z[i]))
        dW[i-1] = (1. / m) * np.matmul(dZ[i], X[i-1].T)
        db[i-1] = (1. / m) * np.sum(dZ[i], axis=1, keepdims=True)

    return dW, db

# back propagation algorithm for ReLU activation
def back_propagate_ReLU(X, Y, W, b, Z, K, m):
    dZ, dW, db, dX = [], [], [], []
    for i in range(0, K + 1):
        dZ.append(0), dW.append(0), db.append(0), dX.append(0)
    dZ[K] = (1./m) * np.multiply(X[K] - Y, step(Z[K]))
    dW[K-1] = np.matmul(dZ[K], X[K-1].T)
    db[K-1] = np.sum(dZ[K], axis=1, keepdims=True)
    for i in reversed(range(1, K)):
        dX[i] = (1./m) * np.matmul(W[i].T, dZ[i+1])
        dZ[i] = np.multiply(dX[i], step(Z[i]))
        dW[i-1] = np.matmul(dZ[i], X[i-1].T)
        db[i-1] = np.sum(dZ[i], axis=1, keepdims=True)

    return dW, db

