import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


'''
LL 
'''

monkey = 'p'
n_maps = 3

df_non = pd.read_csv('./data/p/LL_one_trial/one_trial_LLtable.csv')
#data_non = df_non.drop('Unnamed: 0', axis=1) 
data_non = df_non.rename(columns=lambda x: int(x))
df_str = pd.read_csv("./data/p/LL_one_trial/LLtable_3maps.csv")
data_str = df_str.drop('Unnamed: 0', axis=1) 
data_str = data_str.rename(columns=lambda x: int(x))

data_non = pd.melt(data_non, var_name='Day', value_name='LL')


data_str = pd.melt(data_str, var_name='Day', value_name='LL')


# x를 정수로 변환
data_non['Day'] = data_non['Day'].astype(int)
data_str['Day'] = data_str['Day'].astype(int)


# 각 x 값에 대한 통계 계산 (평균 및 표준 편차)
stats_data_non = data_non.groupby('Day')['LL'].agg(['mean', 'std']).reset_index()
m_non = stats_data_non['mean']
stats_data_str = data_str.groupby('Day')['LL'].agg(['mean', 'std']).reset_index()
m_str = stats_data_str['mean']


# 그림 초기화
plt.figure(figsize=(10, 7))
'''
for i, (mean, std) in enumerate(zip(stats_data_non['mean'], stats_data_non['std'])):
    rect = Rectangle((i - 0.25, mean - std), 0.5, 2 * std, color='green', alpha=0.3)
    plt.gca().add_patch(rect)
'''
plt.errorbar(range(2,9), stats_data_non['mean'], 
             yerr=stats_data_non['std'], fmt='none', color='green', capsize=5)
plt.plot(range(2,9), m_non.to_numpy(), color='purple',marker='o', label='one_trial')


plt.errorbar(range(2,9), stats_data_str['mean'], 
             yerr=stats_data_str['std'], fmt='none', color='orange', capsize=5)
plt.plot(range(2,9), m_str.to_numpy(), color='orange',marker='o', label='window')



plt.plot(range(2,9),-2*np.ones(7),color='red', label='Random')
plt.plot(range(2,9),-0.7*np.ones(7),color='blue', label='Mouse')

day_label = ['day2','day3','day4','day5','day6','day7','day8']
# 그래프에 레이블, 제목 등 추가
plt.xlabel('Day', fontsize=15, labelpad=15)
plt.ylabel('LL', fontsize=15, labelpad=15)
#plt.ylim([-2.2,-0.6])
plt.xticks(fontsize=12)
#plt.yticks([-2.0,-1.6,-1.2,-0.8,-0.7], fontsize=12)
plt.title('Log Likelihood (LL) of '+str(n_maps)+' Goal Maps', fontsize=17, pad=20)
plt.legend(loc='upper right', fontsize=15)
# 그래프 표시
plt.savefig("./figures/p/LL_across_day_map.png")
