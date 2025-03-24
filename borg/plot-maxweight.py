from simulate import simulate_maxweight
import numpy as np
import plotly.express as px
from multiprocessing import Pool

x = np.arange(0.1, 1.0, 0.1)
with Pool() as p:
    y = p.map(simulate_maxweight, x)
# y = [simulate_maxweight(i) for i in x]
fig = px.line(x=x, y=y)
fig.write_image("4.pdf")

y = np.array(y)
with open("maxweight.npy", "wb") as f:
    np.save(f, y)
