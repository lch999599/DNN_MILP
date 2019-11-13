import random
import gurobipy as grb
import pandas as pd
from Main import *
import numpy as np

params, cache = main()
X_train, X_test, y_train, y_test, digits = importdata(55000)
p = random.randint(0, 55000)
x = X_train[:, p].reshape(X_train.shape[0], 1)
y = y_train[:, p].reshape(y_train.shape[0], 1)

K = int(len(params)/2)
n = {0: X_train.shape[0]}
for i in range(0, K):
    n[i+1] = params["W{}".format(i)].shape[0]

set_K = range(0, K+1)
l, u = {}, {}
for k in set_K[1:]:
    for j in range(0, n[k]):
        l[(j, k)] = 0
        u[(j, k)] = 500

opt_model = grb.Model(name="MIP Model")

x_vars, s_vars, z_vars = {}, {}, {}

for j in range(0, n[0]):
    x_vars[(j, 0)] = x[j]

for k in set_K[1:]:
    for j in range(0, n[k]):
        x_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=l[j, k],
                                          ub=u[j, k],
                                          name="x_{0}_{1}".format(j, k))
        s_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=l[j, k],
                                          ub=u[j, k],
                                          name="s_{0}_{1}".format(j, k))
        z_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.BINARY,
                                          name="z_{0}_{1}".format(j, k))

M1 = opt_model.addVar(vtype=grb.GRB.CONTINUOUS, name="M1")
M2 = opt_model.addVar(vtype=grb.GRB.CONTINUOUS, name="M2")

constraints = {}
bin_constraints = {}
lin_constraints = {}
# for k in set_K[1:]:
#     for j in range(0, n[k]):
#             constraints[(j,k)] = opt_model.addConstr(
#                 lhs=grb.quicksum(params["W{}".format(k-1)][j, i]*x_vars[i, k-1] + params["b{}".format(k-1)][j,0]
#                                  for i in range(0, n[k-1])),
#                 sense=grb.GRB.EQUAL,
#                 rhs=x_vars[j, k] - s_vars[j, k],
#                 name="constraint_{0}_{1}".format(j,k)
#             )

for k in set_K[1:]:
    for j in range(0, n[k]):
        bin_constraints[(j, k)] = opt_model.addConstr(
            lhs=x_vars[j, k],
            sense=grb.GRB.LESS_EQUAL,
            rhs=M1 * (1-z_vars[j, k]),
            name="binary_constraint_{0}_{1}".format(j, k)
        )
        bin_constraints[(j, k)] = opt_model.addConstr(
            lhs=s_vars[j, k],
            sense=grb.GRB.LESS_EQUAL,
            rhs=M2 * z_vars[j, k],
            name="binary_constraint_{0}_{1}".format(j, k)
        )

for k in set_K[1:]:
    for j in range(0, n[k]):
        lin_constraints[(j, k)] = opt_model.addConstr(
            lhs=-M2,
            sense=grb.GRB.LESS_EQUAL,
            rhs=grb.quicksum(params["W{}".format(k-1)][j, i]*x_vars[i, k-1] + params["b{}".format(k-1)][j,0]
                             for i in range(0, n[k-1])),
            name="linear_constraint1_{0}_{1}".format(j,k)
        )
        lin_constraints[(j, k)] = opt_model.addConstr(
            lhs=M1,
            sense=grb.GRB.GREATER_EQUAL,
            rhs=grb.quicksum(params["W{}".format(k - 1)][j, i] * x_vars[i, k - 1] + params["b{}".format(k - 1)][j, 0]
                             for i in range(0, n[k - 1])),
            name="linear_constraint2_{0}_{1}".format(j, k)
        )

obj = []
for k in set_K:
    for j in range(0, n[k]):
        obj.append(x_vars[j, k])
        if k >= 1:
            obj.append(z_vars[j, k])

objective = grb.quicksum(obj)

opt_model.ModelSense = grb.GRB.MINIMIZE

opt_model.setObjective(objective)

opt_model.optimize()

opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
opt_df.reset_index(inplace=True)
