from os import sep
import pandas as pd

df=pd.read_csv("data/MLLSratings.csv")

df=df[["userId","movieId","rating"]]
df.to_csv("Baseline/data/bprMLLS.csv",sep="\t",header=False,index=False)
print(len(df))