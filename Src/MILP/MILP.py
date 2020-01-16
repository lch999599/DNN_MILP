import random
import gurobipy as grb
from Src.Data.ImportNMIST import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Src.supportingFunctions import *

import time

K = 2
W, b = [], []
print("loading weights and biases...")
for k in range(0, K):
    W.append(np.load("Src/DNN/ReLUTrain/{}_layers/W{}.npy".format(K, k)))
    b.append(np.load("Src/DNN/ReLUTrain/{}_layers/b{}.npy".format(K, k)))

print("loading test data...")
X_train, X_test, y_train, y_test, digits = loadData(60000)

begin = time.time()
print("loaded.")
print("setting up variables...")
found = False
# while(not found):
#     p = random.randint(0, 10000)
#     x = X_train[:, p].reshape(X_train.shape[0], 1)
#     y = y_train[:, p].reshape(y_train.shape[0], 1)
#     e, f = feed_forward_ReLU(x, W, b, K)
#     if np.argmax(f[K]) != np.argmax(y):
#         found = True

p = random.randint(0, 10000)
x = X_train[:, p].reshape(X_train.shape[0], 1)
y = y_train[:, p].reshape(y_train.shape[0], 1)

fakeList = [6,7,8,5,9,3,5,9,3,4]
real = list(y).index(1.)
fake = fakeList[real]

cost = []

n = {0: X_train.shape[0]}
for i in range(0, K):
    n[i+1] = W[i].shape[0]

set_K = range(0, K+1)

M1 = 100
M2 = 100

opt_model = grb.Model(name="MIP Model {}")
opt_model.setParam(grb.GRB.Param.OutputFlag, 1)

for i in range(0, 10):
    if i != 3:
        cost.append(opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                     lb=0,
                                     ub=0,
                                     name="cost_{}.format(i)"))

    else:
        cost.append(opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                     lb=1,
                                     ub=1,
                                     name="cost_{}.format(i)"))

x_vars, s_vars, z_vars, d_vars = {}, {}, {}, {}

for j in range(0, n[0]):
    x_vars[(j, 0)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                      lb=0,
                                      ub=1,
                                      name="x_{0}_{1}".format(j, 0))
    # d_vars[j] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
    #                              lb=0,
    #                              ub=1,
    #                              name="d_{0}".format(j))

# for j in range(0, n[0]):
#     x_vars[(j, 0)] = x[j]

for k in set_K[1:]:
    for j in range(0, n[k]):
        x_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=0,
                                          ub=100,
                                          name="x_{0}_{1}".format(j, k))
        s_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=0,
                                          ub=100,
                                          name="s_{0}_{1}".format(j, k))
        z_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.BINARY,
                                          name="z_{0}_{1}".format(j, k))

init_constraints = {}

# for j in range(0, n[0]):
#     init_constraints[(j, 1)] = opt_model.addConstr(
#         lhs=-d_vars[j] - x_vars[j, 0],
#         sense=grb.GRB.LESS_EQUAL,
#         rhs=-x[j],
#         name="init_constraint_{}-1".format(j)
#     )
#     init_constraints[(j, 1)] = opt_model.addConstr(
#         lhs=x_vars[j, 0] - d_vars[j],
#         sense=grb.GRB.LESS_EQUAL,
#         rhs=x[j],
#         name="init_constraint_{}-2".format(j)
#     )

print("setting up constraints...")
constraints, bin_constraints, lin_constraints = {}, {}, {}
for k in set_K[1:]:
    for j in range(0, n[k]):
        constraints[(j, k)] = opt_model.addConstr(
            lhs=grb.quicksum(W[k - 1][j, i] * x_vars[i, k - 1]
                             for i in range(0, n[k - 1])) - x_vars[j, k] + s_vars[j, k],
            sense=grb.GRB.EQUAL,
            rhs=-b[k - 1][j, 0],
            name="constraint_{0}_{1}".format(j, k)
        )
        bin_constraints[(j, k, 0)] = opt_model.addConstr(
            lhs=x_vars[j, k] + M1 * z_vars[j, k],
            sense=grb.GRB.LESS_EQUAL,
            rhs=M1,
            name="binary_constraint1_{0}_{1}".format(j, k)
        )
        bin_constraints[(j, k, 1)] = opt_model.addConstr(
            lhs=s_vars[j, k] - M2 * z_vars[j, k],
            sense=grb.GRB.LESS_EQUAL,
            rhs=0,
            name="binary_constraint2_{0}_{1}".format(j, k)
        )
        lin_constraints[(j, k, 0)] = opt_model.addConstr(
            lhs=grb.quicksum(W[k - 1][j, i] * x_vars[i, k - 1]
                             for i in range(0, n[k - 1])),
            sense=grb.GRB.GREATER_EQUAL,
            rhs=-M2 - b[k - 1][j, 0],
            name="linear_constraint1_{0}_{1}".format(j, k)
        )
        lin_constraints[(j, k, 1)] = opt_model.addConstr(
            lhs=grb.quicksum(W[k - 1][j, i] * x_vars[i, k - 1]
                             for i in range(0, n[k])),
            sense=grb.GRB.LESS_EQUAL,
            rhs=M1 - b[k - 1][j, 0],
            name="linear_constraint2_{0}_{1}".format(j, k)
        )

# exit_constraints = {}
# for j in range(0,n[K]):
#     if (j != fake):
#         exit_constraints[j] = opt_model.addConstr(
#             lhs=x_vars[fake,K]-1.2*x_vars[j,K],
#             sense=grb.GRB.GREATER_EQUAL,
#             rhs=0,
#             name="exit_constraint{0}".format(j)
#         )

print("setting objective...")

obj = []
# for k in set_K:
#     for j in range(0, n[k]):
#         obj.append(0*x_vars[j, k])
#         if k >= 1:
#             obj.append(0*z_vars[j, k])

for j in range(0, n[K]):
    obj.append((cost[j]-x_vars[j,K])*(cost[j]-x_vars[j,K]))

objective = grb.quicksum(obj)

opt_model.setObjective(objective, grb.GRB.MINIMIZE)

print("Setup done. optimizing model...")
opt_model.optimize()

print("optimizing done. Printing solution:")

end = time.time()

x_end = np.zeros((n[0],1))
for i in range(0, n[0]):
    x_end[i] = x_vars[i,0].x

for i in range(0, n[K]):
    print(x_vars[i,K].getAttr(grb.GRB.Attr.X))

#print('program done. actual is {} Took {} seconds to complete'.format(real, end-begin))

plt.show()
plt.imshow(x.reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")

plt.show()
plt.imshow(x_end.reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()



e,f = feed_forward_ReLU(x,W,b,K)
print(f[K])

e,f = feed_forward_ReLU(x_end,W,b,K)
print(f[K])

# opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
# opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
# opt_df.reset_index(inplace=True)
