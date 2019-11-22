import random
import gurobipy as grb
from Src.DNN.mainReLU import *

params, cache = main()
X_train, X_test, y_train, y_test, digits = importdata(55000)
p = random.randint(0, 55000)
x = X_train[:, p].reshape(X_train.shape[0], 1)
y = y_train[:, p].reshape(y_train.shape[0], 1)
real = list(y).index(1.)

K = int(len(params)/2)
n = {0: X_train.shape[0]}
for i in range(0, K):
    n[i+1] = params["W{}".format(i)].shape[0]

set_K = range(0, K+1)
l, u = {}, {}
for k in set_K[1:]:
    for j in range(0, n[k]):
        l[(j, k)] = 0
        u[(j, k)] = 1500

opt_model = grb.Model(name="MIP Model")

x_vars, s_vars, z_vars = {}, {}, {}

for j in range(0, n[0]):
    x_vars[(j, 0)] = x[j]

for k in set_K[1:]:
    for j in range(0, n[k]):
        x_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=0,
                                          ub=u[j, k],
                                          name="x_{0}_{1}".format(j, k))
        s_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          lb=0,#lb=l[j, k],
                                          #ub=u[j, k],
                                          name="s_{0}_{1}".format(j, k))
        z_vars[(j, k)] = opt_model.addVar(vtype=grb.GRB.BINARY,
                                          name="z_{0}_{1}".format(j, k))

M1 = 20000
M2 = 20000

constraints = {}
bin_constraints = {}
lin_constraints = {}
for k in set_K[1:]:
    for j in range(0, n[k]):
            constraints[(j,k)] = opt_model.addConstr(
                lhs=grb.quicksum(params["W{}".format(k-1)][j, i]*x_vars[i, k-1]
                                 for i in range(0, n[k-1])) - x_vars[j, k] + s_vars[j, k],
                sense=grb.GRB.EQUAL,
                rhs=-params["b{}".format(k-1)][j,0],
                name="constraint_{0}_{1}".format(j,k)
            )

for k in set_K[1:]:
    for j in range(0, n[k]):
        bin_constraints[(j, k)] = opt_model.addConstr(
            lhs=x_vars[j, k] + M1 * z_vars[j, k],
            sense=grb.GRB.LESS_EQUAL,
            rhs=M1,
            name="binary_constraint1_{0}_{1}".format(j, k)
        )
        bin_constraints[(j, k)] = opt_model.addConstr(
            lhs=s_vars[j, k] - M2 * z_vars[j, k],
            sense=grb.GRB.LESS_EQUAL,
            rhs=0,
            name="binary_constraint2_{0}_{1}".format(j, k)
        )

for k in set_K[1:]:
    for j in range(0, n[k]):
        lin_constraints[(j, k)] = opt_model.addConstr(
            lhs=grb.quicksum(params["W{}".format(k-1)][j, i] * x_vars[i, k-1]
                             for i in range(0, n[k-1])),
            sense=grb.GRB.GREATER_EQUAL,
            rhs= -M2 - params["b{}".format(k-1)][j,0],
            name="linear_constraint1_{0}_{1}".format(j,k)
        )
        lin_constraints[(j, k)] = opt_model.addConstr(
            lhs=grb.quicksum(params["W{}".format(k-1)][j, i] * x_vars[i, k-1]
                             for i in range(0, n[k])),
            sense=grb.GRB.LESS_EQUAL,
            rhs= M1 - params["b{}".format(k-1)][j, 0],
            name="linear_constraint2_{0}_{1}".format(j, k)
        )

obj = []
for k in set_K:
    for j in range(0, n[k]):
        obj.append(0*x_vars[j, k])
        if k >= 1:
            obj.append(0*z_vars[j, k])

objective = grb.quicksum(obj)

opt_model.ModelSense = grb.GRB.MINIMIZE

opt_model.setObjective(objective)

opt_model.optimize()

maximum = 0
for i in range(0,10):
    print("value for {}: ".format(i), x_vars[i,3].x)
    if x_vars[i,3].x > x_vars[maximum,3].x:
        maximum = i


plt.imshow(x.reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print("real: ", real, ", predicted: ", maximum)

# opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
# opt_df.index = pd.MultiIndex.from_tuples(opt_df.index, names=["column_i", "column_j"])
# opt_df.reset_index(inplace=True)
