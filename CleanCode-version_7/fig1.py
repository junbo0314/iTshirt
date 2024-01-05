import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import glob

'''
weight 평균 보기
'''
monkey = 'p'
T_dict = {'p' : [23,37,28,20,17,13,25], 's':[9,13,13,12,14,11,16,27]}
Time = T_dict[monkey]
T = int(sum(Time)/len(Time))
n_maps = 4
time_range = range(T)
df = [pd.DataFrame() for i in range(n_maps)]

for day in range(2,9) :
    time = Time[day-2]
    weights = np.load('parameters/'+str(monkey)+'/day'+str(day)+'/3_9_map'+\
                      str(n_maps)+'_time'+str(time)+'_bias[25, 25, 25, 25]_weights_stat.npy')
    import pdb; pdb.set_trace()
    for map in range(n_maps) :
        degree = 10  # 다항식의 차수 설정
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly_features.fit_transform(np.arange(time).reshape(-1, 1))
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, weights[map])
        X_new = np.linspace(0,time-1,T,endpoint=True)
        X_new_poly = poly_features.transform(X_new.reshape(-1, 1))
        y_new = poly_reg.predict(X_new_poly)
        data = pd.DataFrame({'X': time_range, 'Y': y_new})
        df[map] = pd.concat([df[map], data], axis=0, ignore_index=True)


labels = ['Current Goal', 'Old Goal', 'Sub Goal', 'Exploration']
colors = ['red','blue','green','purple']

if n_maps ==2 :
    fig = plt.figure(figsize=(7.1, 4))
if n_maps == 4 :
    fig = plt.figure(figsize=(14, 4))
for map in range(n_maps) :
    plt.subplot(1,n_maps,map+1)
    sns.lineplot(x='X', y='Y', data=df[map], color=colors[map], errorbar='sd',
                 label=labels[map],linewidth=2.5,linestyle="--")
    plt.axhline(y=0, color='k', alpha=0.6, linestyle="--", linewidth=0.5)
    plt.xlabel('Time [step]', fontsize=15)
    plt.ylabel('')
    plt.xticks(range(0,T,5),fontsize=10)
    plt.yticks([-1, 0, 1], fontsize=10)
    plt.title('Weight for '+labels[map], fontsize=15)
    
plt.tight_layout()
plt.legend()
fig.savefig('/Users/swankim/Desktop/'+monkey+'weights_avg_map'+str(n_maps)+'_stat.png')