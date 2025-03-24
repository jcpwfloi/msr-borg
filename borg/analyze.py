import pandas as pd
import numpy as np
from maxmin import get_candidates
import time

scale = 0.018

df = pd.read_csv("popular.csv")
stats = df.groupby(["ceiledcpu", "ceiledmemory"]).size()
stats = pd.concat([stats, df.groupby(["ceiledcpu", "ceiledmemory"]).mean()["job_size"]], axis=1)
stats = stats.reset_index().rename(columns={0: "count"})
stats["lambda"] = stats["count"] / df["arrival_time"].max()
stats["load"] = stats["lambda"] * stats["job_size"] * scale
D = np.concatenate((np.array([stats["ceiledcpu"]]).T, np.array([stats["ceiledmemory"]]).T), axis=1)
R = D.shape[1]
C = np.ones(R)
lmb = np.array(stats["load"])
print("lambda: ", lmb)
stats.to_csv("popularstats.csv")

start = time.time()
alpha, candidates = get_candidates(C, D, lmb)
print("time elapsed:", time.time() - start)
print(alpha)
print(candidates)
print(lmb)
print(alpha @ candidates)
print(lmb / (alpha @ candidates)[0])
print(C, candidates @ D)
with open('popular.npy', 'wb') as f:
    np.save(f, alpha)
    np.save(f, candidates)
