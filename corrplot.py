from scipy import stats
import numpy as np
import pandas as pd

df = pd.read_csv('re2.csv')
print(df)
print(stats.spearmanr(df)) #.Pred, df.Real))
