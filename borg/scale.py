import pandas as pd
import numpy as np

fullscale = 0.01983

df = pd.read_csv("popular.csv")

res = [pd.DataFrame(df.sample(frac=fullscale * i, random_state=1)).sort_values("arrival_time").to_csv(f"scaled/{'%.1f' % i}.csv", index=False) for i in np.arange(0.1, 1.0, 0.1)]
