import gurobipy as grb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from Src.supportingFunctions import *

def GenerateAdversarial(K, W, b, X, y, i, predicted, expected):
    x = X[:,i]
    n = {0: X.shape[0]}
    set_K = range(0, K + 1)
    for i in range(0, K):
        n[i + 1] = W[i].shape[0]

    M1 = 25
    M2 = 25

    opt_model = grb.Model(name="MIP Model {}")
    opt_model.setParam(grb.GRB.Param.OutputFlag, 0)

    x_vars, s_vars, z_vars, d_vars = {}, {}, {}, {}
    for j in range(0, n[0]):
        x_vars[(j, 0)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=0,
                                          ub=1,
                                          name="x_{0}_{1}".format(j, 0))
        d_vars[j] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                     lb=0,
                                     ub=1,
                                      name="d_{0}".format(j))

    for k in set_K[1:]:
        for j in range(0, n[k]):
            x_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              lb=0,
                                              ub=5,
                                              name="x_{0}_{1}".format(j, k))
            s_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              lb=0,
                                              ub=5,
                                              name="s_{0}_{1}".format(j, k))
            z_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.BINARY,
                                              name="z_{0}_{1}".format(j, k))

    init_constraints = {}
    for j in range(0, n[0]):
        init_constraints[(j, 1)] = opt_model.addConstr(
            lhs=-d_vars[j] - x_vars[j, 0],
            sense=grb.GRB.LESS_EQUAL,
            rhs=-x[j],
            name="init_constraint_{}-1".format(j)
        )
        init_constraints[(j, 1)] = opt_model.addConstr(
            lhs=x_vars[j, 0] - d_vars[j],
            sense=grb.GRB.LESS_EQUAL,
            rhs=x[j],
            name="init_constraint_{}-2".format(j)
        )

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

    exit_constraints = {}
    for j in range(0,n[K]):
        if (j != expected):
            exit_constraints[j] = opt_model.addConstr(
                lhs=x_vars[expected,K]-1.2*x_vars[j,K],
                sense=grb.GRB.GREATER_EQUAL,
                rhs=0,
                name="exit_constraint{0}".format(j)
            )

    obj = []
    for i in range(0, n[0]):
        obj.append(d_vars[i])

    objective = grb.quicksum(obj)
    opt_model.ModelSense = grb.GRB.MINIMIZE
    opt_model.setObjective(objective)
    opt_model.optimize()
    newX = np.zeros((n[0], 1))
    newy = np.zeros((n[K], 1))
    newy[predicted] = 1
    for i in range(0, n[0]):
        newX[i] = x_vars[i, 0].x
    X = np.hstack((X, newX))
    y = np.hstack((y,newy))
    plt.imshow(newX.reshape(28, 28), cmap= matplotlib.cm.binary)
    plt.axis("off")
    plt.show()
    e, f = feed_forward_ReLU(newX, W, b, K)
    print(f[K])
    print('predicted, expected: ', predicted, expected)
    return X, y

