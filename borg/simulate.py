import numpy as np
import pandas as pd
from maxmin import get_candidates
import plotly.express as px
import subprocess
import sys
import os
import time
from multiprocessing import Pool
from math import inf
from scipy.special import factorial

# binary_dir = "/nas/longleaf/home/jcpwfloi/mama/build/src"
command = ""

def generate_stats(df):
    stats = df.groupby(["ceiledcpu", "ceiledmemory"]).size()
    stats = pd.concat([stats, df.groupby(["ceiledcpu", "ceiledmemory"]).mean()["job_size"]], axis=1)
    stats = stats.reset_index().rename(columns={0: "count"})
    stats["lambda"] = stats["count"] / df["arrival_time"].max()
    stats["load"] = stats["lambda"] * stats["job_size"]
    return stats

def simulate_ff(lmb):
    print(lmb, flush=True)
    command = "../build/src/borg"
    filename = f"scaled/{'%.1f' % lmb}.csv"
    df = pd.read_csv(filename)
    stats = generate_stats(df)
    type_info = (stats["ceiledcpu"].astype("str") + " " + stats["ceiledmemory"].astype("str") + " " + stats["lambda"].astype("str")).str.cat(sep='\n')
    inp = [filename, len(stats), type_info]
    input = "\n".join([str(ss) for ss in inp])
    print(input)
    out = subprocess.run(command, input=bytes(input, "utf-8"), shell=True, capture_output=True)
    out = out.stdout.decode("utf-8").split("\n")[:-1]
    out = np.array(out).astype("float")[None]
    print(out)
    return out

def simulate_rt(lmb, clock_speed=0.01):
    print(lmb, flush=True)
    command = "../build/src/borg-rt"
    filename = f"scaled/{'%.1f' % lmb}.csv"
    df = pd.read_csv(filename)
    stats = generate_stats(df)
    type_info = (stats["ceiledcpu"].astype("str") + " " + stats["ceiledmemory"].astype("str") + " " + stats["lambda"].astype("str")).str.cat(sep='\n')
    clock = [str(clock_speed) for i in range(10)]
    clock = " ".join(clock)
    inp = [filename, len(stats), type_info, clock]
    input = "\n".join([str(ss) for ss in inp])
    print(input, flush=True)
    out = subprocess.run(command, input=bytes(input, "utf-8"), shell=True, capture_output=True)
    out = out.stdout.decode("utf-8").split("\n")[:-1]
    out = np.array(out).astype("float")[None]
    print("+++", lmb, out, flush=True)
    return out

def simulate_maxweight(lmb):
    print(lmb, flush=True)
    command = "../build/src/borg-maxweight"
    filename = f"scaled/{'%.1f' % lmb}.csv"
    df = pd.read_csv(filename)
    stats = generate_stats(df)
    type_info = (stats["ceiledcpu"].astype("str") + " " + stats["ceiledmemory"].astype("str") + " " + stats["lambda"].astype("str")).str.cat(sep='\n')
    inp = [filename, len(stats), type_info]
    input = "\n".join([str(ss) for ss in inp])
    print(input, flush=True)
    out = subprocess.run(command, input=bytes(input, "utf-8"), shell=True, capture_output=True)
    try:
        out = float(out.stdout.decode("utf-8")[:-1])
    except:
        print(lmb, out.stdout, flush=True)
        print(lmb, out.stderr, flush=True)
        out = -1.0
    print("++++++", lmb, out, flush=True)
    return out

def run_with_input(zipped):
    x, input = zipped
    global command
    out = subprocess.run(command, input=bytes(input, "utf-8"), shell=True, capture_output=True)
    out = out.stdout.decode("utf-8").split("\n")[:-1]
    out = np.array(out).astype("float")[None]
    print(command, x, out, flush=True)
    return out

def simulate_msr(lmb, alpha_scale=1.0, dryrun=False):
    filename = f"scaled/{'%.1f' % lmb}.csv"
    df = pd.read_csv(filename)
    stats = generate_stats(df)
    type_info = (stats["ceiledcpu"].astype("str") + " " + stats["ceiledmemory"].astype("str") + " " + stats["lambda"].astype("str")).str.cat(sep='\n')
    D = np.concatenate((np.array([stats["ceiledcpu"]]).T, np.array([stats["ceiledmemory"]]).T), axis=1)
    R = D.shape[1]
    C = np.ones(R)
    lmb = np.array(stats["load"])
    before = time.time()
    alpha, candidates = get_candidates(C, D, lmb)
    print("-=-=-=-=-=-=-=", time.time() - before)
    alpha = alpha[0]
    idx = np.arange(0, alpha.shape[0], 1)
    idx = sorted(idx, key = lambda x : -alpha[x])
    alpha = alpha[idx]
    candidates = candidates[idx]
    cutoff = 0.001
    candidates = candidates[alpha > cutoff]
    alpha = alpha[alpha > cutoff]
    alpha /= alpha.sum()
    print(alpha)
    print(candidates)
    print(lmb)
    print(alpha @ candidates)
    print(lmb / (alpha @ candidates))
    print(C, candidates @ D)
    alpha_str = " ".join("%.8f" % x for x in alpha)
    rates = alpha[0] / alpha #* alpha_scale
    print(rates, np.dot(alpha, 1.0 / rates))
    rates *= (np.dot(alpha, 1.0 / rates)) * alpha_scale
    print(rates, np.dot(alpha, 1.0 / rates))
    rates = " ".join(["%.8f" % x for x in rates])
    candidates = "\n".join([" ".join(["%d" % a for a in x]) for x in candidates])
    # inp = [filename, len(stats), type_info, alpha.shape[0], alpha_str, rates, candidates]
    inp = [alpha.shape[0], alpha_str, rates, candidates]
    input = "\n".join([str(ss) for ss in inp])
    print(input)
    return input

simulate_msr(0.8)

def simulate_msr_from_candidates_string(lmb, base):
    filename = f"scaled/{'%.1f' % lmb}.csv"
    df = pd.read_csv(filename)
    stats = generate_stats(df)
    type_info = (stats["ceiledcpu"].astype("str") + " " + stats["ceiledmemory"].astype("str") + " " + stats["lambda"].astype("str")).str.cat(sep='\n')
    D = np.concatenate((np.array([stats["ceiledcpu"]]).T, np.array([stats["ceiledmemory"]]).T), axis=1)
    R = D.shape[1]
    C = np.ones(R)
    lmb = np.array(stats["load"])
    inp = [filename, len(stats), type_info, base]
    input = "\n".join([str(ss) for ss in inp])
    return input

def simulate_msr_main(save_file, plot_file, ff=True):
    global command

    command = "../build/src/borg-msr" if ff else "../build/src/borg-msr-no-ff"

    x = np.arange(0.1, 1.0, 0.1)
    base_input = simulate_msr(0.9, alpha_scale=0.1)
    inputs = [simulate_msr_from_candidates_string(i, base_input) for i in x]
    with Pool() as p:
        y = p.map(run_with_input, zip(x, inputs))
    y = np.concatenate(y)
    fig = px.line(x=x, y=y.T[-1])
    fig.write_image(plot_file)

    with open(save_file, "wb") as f:
        np.save(f, inputs)
        np.save(f, y)

def simulate_nmsr_main(save_file, plot_file, ff=True, alpha_scale=1.0):
    global command

    command = "../build/src/borg-nmsr" if ff else "../build/src/borg-nmsr-no-ff"

    x = np.arange(0.1, 1.0, 0.1)
    base_input = simulate_msr(0.9, alpha_scale=alpha_scale)
    inputs = [simulate_msr_from_candidates_string(i, base_input) for i in x]
    with Pool() as p:
        y = p.map(run_with_input, zip(x, inputs))
    y = np.concatenate(y)
    fig = px.line(x=x, y=y.T[-1])
    fig.write_image(plot_file)

    with open(save_file, "wb") as f:
        np.save(f, inputs)
        np.save(f, y)

def simulate_nmsr_alpha(save_file, plot_file, ff=True, alpha_scale=1.0, lmb=0.6):
    global command

    command = "../build/src/borg-nmsr" if ff else "../build/src/borg-nmsr-no-ff"

    x = np.logspace(-7, -5, num=10)
    inputs = [simulate_msr_from_candidates_string(0.9, simulate_msr(0.9, alpha_scale=i)) for i in x]
    with Pool() as p:
        y = p.map(run_with_input, zip(x, inputs))
    y = np.concatenate(y)
    fig = px.line(x=x, y=y.T[-1])
    fig.write_image(plot_file)

    y = np.array(y)
    with open(save_file, "wb") as f:
        np.save(f, inputs)
        np.save(f, y)

def PQ(k, rho):
    i = np.arange(0, k, 1)
    pi0 = (np.power(k*rho, i) / factorial(i)).sum() + np.power(k*rho, k) / factorial(k) / (1-rho)
    pi0 = 1.0 / pi0
    return np.power(k*rho, k) * pi0 / factorial(k) / (1-rho)

def compute_estimate(orig_ctmc, mus, lmb):
    ctmc = np.array(orig_ctmc)
    mus = np.array(mus)
    for i in range(len(ctmc)):
        rowsum = sum([ctmc[i][x] for x in range(len(ctmc)) if i != x])
        ctmc[i][i] = -rowsum
    ctmc = np.array(ctmc).T
    b = np.zeros(ctmc.shape[0])
    ctmc[-1] = np.ones(ctmc.shape[0])
    b[-1] = 1
    Y = np.linalg.solve(ctmc, b)
    mu_star = np.dot(Y, mus.T)
    if mu_star < lmb:
        return inf, inf, inf, inf

    ctmc = np.array(orig_ctmc)
    for i in range(len(ctmc)):
        rowsum = sum([ctmc[i][x] for x in range(len(ctmc)) if i != x])
        ctmc[i][i] = -rowsum
    b = mu_star - mus
    b[-1] = 0
    ctmc[-1] = Y
    D = np.linalg.solve(ctmc, b)

    Yd = Y * mus / mu_star
    EYd = np.dot(D, Yd)
 
    rho = lmb / mu_star

    ub = (rho + EYd) / (1 - rho) + (-D).max() + mus.max()
    lb = (rho + EYd) / (1 - rho) + (-D).min()
    # (rho + EYd) / (1 - rho) - EYd
    # = (rho + EYd) / (1 - rho) + (EYd (rho - 1)) / (1-rho)
    # = (rho + EYd + EYd(rho - 1)) / (1-rho)
    # = (rho + EYd * rho) / (1 - rho)
    prediction = (1 + EYd) / (1 - rho) * rho #+ mus.max() * (1-rho)
    prediction = PQ(mu_star, rho) * prediction + rho * mu_star #* (1-PQ(mu_star, rho))
    prediction = max(lb, prediction)

    return lb, prediction, ub, mu_star

def compute_estimate_from_candidates(ctmc, candidates, lmb, stats):
    lmb = np.array(stats["lambda"])
    ans = []
    for i in range(candidates.shape[1]):
        ans.append(np.array(compute_estimate(ctmc, candidates.T[i], lmb[i]))[None] / lmb[i])
    ans = np.concatenate(ans)
    # print(ans)
    # lmb = np.array(stats["lambda"])
    return ((lmb / lmb.sum())[None] @ ans)[0]

# def get_nmsr(lmb, ctmc, candidate_set, alpha=10.0, mu=None):
#     if mu is None:
#         mu = np.ones(lmb.shape)
#     edge_list = []
#     N_W = ctmc.shape[0]
#     K = candidate_set.shape[1]
#     N_vertices = ctmc.shape[0]

#     def add_edge(u, v, w):
#         edge_list.append([u, v, w])
#     def insert_transitions(u, v):
#         global ctmc, candidate_set
#         current_mu = candidate_set[u]
#         t = candidate_set.shape[0]
#         candidate_set = np.append(candidate_set, [current_mu], axis=0)
#         add_edge(u, t, ctmc[u, v])
#         while np.any(current_mu > candidate_set[v]):
#             ind = np.random.randint(0, K)
#             N_pausing = current_mu[ind] - candidate_set[v][ind]
#             if N_pausing <= 0: continue
#             current_mu[ind] -= 1
#             if np.all(current_mu <= candidate_set[v]):
#                 add_edge(t, v, mu[ind])
#             else:
#                 next_v = candidate_set.shape[0]
#                 candidate_set = np.append(candidate_set, [current_mu], axis=0)
#                 add_edge(t, next_v, N_pausing * mu[ind])
#                 t = next_v
#     for i in range(N_W):
#         for j in range(N_W):
#             if i != j and ctmc[i][j] > 0:
#                 insert_transitions(i, j)
#     ans_ctmc = np.zeros((candidate_set.shape[0], candidate_set.shape[0]))
#     # ans_ctmc[:N_W, :N_W] = ctmc
#     for u, v, w in edge_list:
#         ans_ctmc[u, v] = w
#     return ans_ctmc, candidate_set

def get_ctmc_and_candidates(lmb, alpha_scale=1.0):
    filename = f"scaled/{'%.1f' % lmb}.csv"
    df = pd.read_csv(filename)
    stats = generate_stats(df)
    D = np.concatenate((np.array([stats["ceiledcpu"]]).T, np.array([stats["ceiledmemory"]]).T), axis=1)
    R = D.shape[1]
    C = np.ones(R)
    lmb = np.array(stats["load"])
    alpha, candidates = get_candidates(C, D, lmb)
    # solve for the transition rates
    alpha = alpha[0]
    idx = np.arange(0, alpha.shape[0], 1)
    idx = sorted(idx, key = lambda x : alpha[x])
    alpha = alpha[idx]
    candidates = candidates[idx]
    cutoff = 0.001
    candidates = candidates[alpha > cutoff]
    alpha = alpha[alpha > cutoff]
    alpha /= alpha.sum()
    rates = alpha[0] / alpha * alpha_scale
    N = rates.shape[0]
    ctmc = np.zeros((N, N))
    for i in range(N):
        ctmc[i][(i+1)%N] = rates[i]
    return ctmc, candidates, lmb, stats

def get_estimate(lmb, alpha_scale=1.0):
    ctmc, candidates, lmb, stats = get_ctmc_and_candidates(lmb, alpha_scale=alpha_scale)
    return compute_estimate_from_candidates(ctmc, candidates, lmb, stats)

