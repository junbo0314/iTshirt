import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import wilcoxon
import seaborn as sns

'''
'Log Likelihood Based on the Number of Goal Maps
'''
monkey = 'p'
window_size=3
num_traj=7
maps = [2,3,4]
df = pd.DataFrame(0,index=range(10),columns=range(7), dtype=float)
bias_lam=[25,25,25]
file_path = './data/p/LL_one_trial/one_trial_LLtable.csv'

for day in range(2,9) :
    data = pd.read_csv("./data/p/LL_one_trial/"+"LLtable_"+str(day)+"_[25, 25, 25].csv")
    for window in range (10):
        start=1+window*window_size
        end=start+num_traj
        df1=data.iloc[:,start:end]
        x = df1.mean(axis=1)
        df.iloc[window, day-2] = x
df.to_csv(file_path, index=False, encoding='utf-8')
print(df)