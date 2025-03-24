import numpy as np
import pandas as pd
import plotly.express as px
import time

with open("ff.npy", "rb") as f:
    ff = np.load(f)
    ff = ff.T[-1]

with open("msr.npy", "rb") as f:
    _, msr = np.load(f), np.load(f)
    msr = msr.T[-1]

with open("msr-no-ff.npy", "rb") as f:
    _, msrnoff = np.load(f), np.load(f)
    msrnoff = msrnoff.T[-1]

with open("nmsr.npy", "rb") as f:
    _, nmsr = np.load(f), np.load(f)
    nmsr = nmsr.T[-1]

with open("nmsr-no-ff.npy", "rb") as f:
    _, nmsrnoff = np.load(f), np.load(f)
    nmsrnoff = nmsrnoff.T[:-1]
    nmsrnoff = nmsrnoff.T
    nmsrnoff = nmsrnoff[:, 150:]
    nmsrnoff = nmsrnoff.mean(axis=1)

with open("maxweight.npy", "rb") as f:
    maxweight = np.load(f)

with open("rtimer.npy", "rb") as f:
    rtimer = np.load(f)
    rtimer = rtimer.T[-1]

# print(ff[-1]/msrnoff[-1], maxweight[-1]/msrnoff[-1])

df1 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), ff]).transpose()
df1["policy"] = "First-Fit"
df2 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), msr]).transpose()
df2["policy"] = "pMSR w/ BackFilling"
df3 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), msrnoff]).transpose()
df3["policy"] = "pMSR"
df4 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), nmsr]).transpose()
df4["policy"] = "nMSR w/ BackFilling"
df5 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), nmsrnoff]).transpose()
df5["policy"] = "nmsr-no-backfilling"
df6 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), maxweight]).transpose()
df6["policy"] = "MaxWeight"
df7 = pd.DataFrame([np.arange(0.1, 1.0, 0.1), rtimer]).transpose()
df7["policy"] = "Randomized Timers"
df = pd.concat([df1, df2, df3, df4, df6, df7])
df.columns = ["x", "y", "policy"]

fig = px.line(df, x="x", y="y", color="policy")
fig.update_layout(template="simple_white", 
                  yaxis_title=r"$\text{Mean Response Time}, \mathbb{E}[T]$",
                  xaxis_title=r"$\text{System Load}, \rho$",
                  legend_title_text="",
                #   legend=dict(
                #     x=.72,
                #     y=0.0,
                #   ),
                  font=dict(
                      size=18
                  ),
                  width=800,
                  height=400
                  )
fig.update_xaxes(showgrid=True, exponentformat='power')
fig.update_yaxes(showgrid=True)

fig['data'][1]['line']['dash'] = 'dash'
fig['data'][1]['line']['color'] = 'red'
# fig['data'][2]['line']['dash'] = 'dash'
fig['data'][2]['line']['color'] = 'red'
fig['data'][3]['line']['dash'] = 'dash'

order = [0, 4, 5, 2, 1, 3]

fig['data'] = (fig['data'][0], fig['data'][4], fig['data'][5], fig['data'][2], fig['data'][1], fig['data'][3])

# fig

fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

fig.write_image("../borg.png")
time.sleep(1)
fig.write_image("../borg.png")
