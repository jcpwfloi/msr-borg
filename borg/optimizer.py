# u_ij integers, i-th candidate -- how many type j jobs in schedule
# l_1 + ... + l_K = 1
# l_1, ..., l_K >= 0 nonnegative
# u=l_1u_1 + ... + l_Ku_K, uD <= C stability
# u_1, u_2, ..., u_K suffice resource constraint
# max ||u - lambda||_2

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp
import sys

C=(197, 373, 103)
D=(
    (7, 5, 17),
    (3, 13, 5),
    (7, 17, 3)
)

C, D = np.array(C), np.array(D)
K, R = D.shape[0], D.shape[1]
lmb = np.array([1.0, 2.0, 3.0])

m = gp.Model("mip1")
u = m.addMVar(shape=(K, K), vtype=GRB.INTEGER, name="candidate set")
m.addConstr(u >= np.zeros((K, K)), name="nonnegative candidates")
l = m.addMVar(shape=(1, K), vtype=GRB.CONTINUOUS, name="alphas")
m.addConstr(l.sum() <= 1.0, name="alpha average")
m.addConstr(l >= np.zeros((1, K)), name="nonnegative alphas")
m.addConstr(l @ u >= lmb, name="stability")
m.addConstr(u @ D <= C, name="capacity")

m.setObjective((l @ u - lmb).sum(), GRB.MAXIMIZE)

m.optimize()

alpha = np.array(l.X)
candidates = np.array(u.X)

print("completion rates:", alpha @ candidates)
print(C)
print("capacity limit", candidates @ D)

print(alpha @ candidates - lmb)

# from docplex.mp.model import Model

# C=(197, 373, 103)
# D=(
#     (7, 5, 17),
#     (3, 13, 5),
#     (7, 17, 3)
# )

# K = len(D)
# R = len(C)
# lmb = [1.0, 2.0, 3.0]

# milp_model = Model(name="MIQP")
# u = [[] for x in range(K)]
# for i in range(K):
#     for j in range(K):
#         u[i].append(milp_model.integer_var(name=f"u_{i}{j}", lb=0))
# l = [milp_model.continuous_var(name=f"l{x}", lb=0) for x in range(K)]

# completion = [0 for _ in range(K)]
# for j in range(K):
#     for i in range(K):
#         completion[j] += l[i] * u[i][j]

# for n in range(K):
#     # check validity of u[n]
#     M = [0 for _ in range(R)]
#     for i in range(K):
#         for j in range(R):
#             M[j] += u[n][i] * D[i][j]
#     for j in range(R):
#         milp_model.add_constraint(M[j] <= C[j], ctname="capacity")

# milp_model.add_constraint(sum(l) <= 1.0, ctname="alphas")
# for i in range(K):
#     milp_model.add_constraint(lmb[i] <= completion[i], ctname="stability")
#     milp_model.add_constraint(0.0 <= l[i], ctname="nonnegative")

# obj_fn = sum([completion[x] - lmb[x] for x in range(K)])
# milp_model.set_objective("max", obj_fn)

# milp_model.print_information()
# milp_model.solve()
# print(milp_model.solve_details)
# # milp_model.print_solution()
