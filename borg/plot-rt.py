import numpy as np
import pandas as pd
import plotly.express as px
from simulate import simulate_rt
from multiprocessing import Pool

x = np.arange(0.1, 1.0, 0.1)
with Pool() as p:
    y = p.map(simulate_rt, x)
y = np.concatenate(y)
fig = px.line(x=x, y=y.T[-1])
fig.write_image("7.pdf")

with open("rtimer.npy", "wb") as f:
    np.save(f, y)
