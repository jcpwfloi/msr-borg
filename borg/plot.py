import numpy as np
import pandas as pd
import plotly.express as px
from simulate import simulate_ff
from multiprocessing import Pool

binary_dir = "/nas/longleaf/home/jcpwfloi/mama/build/src"

x = np.arange(0.1, 1.0, 0.1)
with Pool() as p:
    y = p.map(simulate_ff, x)
y = np.concatenate(y)
fig = px.line(x=x, y=y.T[-1])
fig.write_image("1.pdf")

with open("ff.npy", "wb") as f:
    np.save(f, y)
