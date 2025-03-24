import pandas as pd
import glob
import sys

def process(file):
    df = pd.read_csv(file)
    df = df[df["job_size"] < 2e5]
    df = df.drop("Unnamed: 0", axis=1)
    df.to_csv(file, index=False)

files = glob.glob("./*.csv")
for file in files:
    process(file)