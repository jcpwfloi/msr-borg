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

def get_candidates(C, D, lmb, verbose=False):
    with gp.Env(empty=True) as env:
        if not verbose:
            env.setParam('OutputFlag', 0)
        env.start()
        with gp.Model(env=env) as m:
            # Build model m here
            C, D = np.array(C), np.array(D)
            K, R = D.shape[0], D.shape[1]

            # m = gp.Model("mip1")
            u = m.addMVar(shape=(K, K), vtype=GRB.INTEGER, name="candidate set")
            m.addConstr(u >= np.zeros((K, K)), name="nonnegative candidates")
            l = m.addMVar(shape=(1, K), vtype=GRB.CONTINUOUS, name="alphas")
            m.addConstr(l.sum() <= 1.0, name="alpha average")
            m.addConstr(l >= np.zeros((1, K)), name="nonnegative alphas")
            m.addConstr(l @ u >= lmb, name="stability")
            m.addConstr(u @ D <= C, name="capacity")

            gap_vector = l @ u - lmb

            min_gap = m.addVar(vtype=GRB.CONTINUOUS, name="min gap")
            for i in range(K):
                m.addConstr(min_gap * lmb[i] <= gap_vector[0][i], name="gap %d" % i)

            m.setObjective(min_gap, GRB.MAXIMIZE)
            # m.setObjective((l @ u - lmb).sum(), GRB.MAXIMIZE)

            m.optimize()

            if m.SolCount > 0:
                alpha = np.array(l.X)
                candidates = np.array(u.X)
                candidates[candidates == 0.] = 0.
                return alpha, candidates
            
            return None

# C=(197, 373, 103)
# D=(
#     (7, 5, 17),
#     (3, 13, 5),
#     (7, 17, 3)
# )
# lmb = np.array([0.4, 0.6, 2])

# alpha, candidates = get_candidates(C, D, lmb)
# print(alpha @ candidates)
# print(C, candidates @ D)
