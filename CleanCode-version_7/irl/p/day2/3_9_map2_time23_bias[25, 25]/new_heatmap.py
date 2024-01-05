import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def make_dirl_heatmap(i, maxv) :
    reward_file = pd.read_csv(address+"/time"+str(i)+".csv")
    reward_map = reward_file.to_numpy()
    maxv = 3.3
    minv = 0
    heatmap = sns.heatmap(reward_file, vmin = minv, vmax = maxv, cmap='YlGnBu',
                          xticklabels=False, yticklabels=False,
                          cbar_kws={'label': 'Reward'})
    plt.title("time"+str(i+1))
    plt.plot()
    plt.savefig(address+"/time"+str(i)+'.png')
    plt.close()

def make_dirl_goal_map(n_maps) :
    global_max_value = float('-inf')
    # 최대값 찾기
    for i in range(n_maps):
        file_path = address+f'/goal{i}.csv'
        df = pd.read_csv(file_path, index_col=0)
        local_max_value = np.max(df.values)
        global_max_value = max(global_max_value, local_max_value)
    minv = 0
    for i in range(n_maps):
        # CSV 파일 읽기
        file_path = address+f'/goal{i}.csv'
        df = pd.read_csv(file_path, index_col=0)
        # 데이터를 numpy 배열로 변환
        data = df.values
        data[data == 0] = np.nan
        # 2D 히트맵 그리기
        plt.imshow(data, cmap='viridis_r', interpolation='none', vmin=0, vmax=global_max_value, extent=[0, 11, 0, 11])
        plt.colorbar(label='Reward', pad=0.05)
        # x, y 축 눈금 및 라벨 제거
        plt.xticks([]), plt.yticks([])
        # 별 그리기
        plt.scatter(2.5, 3.5, color='yellow', marker='*', s=100)
        plt.scatter(4.5, 3.5, color='yellow', marker='*', s=100)
        plt.scatter(2.5, 9.5, color='yellow', marker='*', s=100)
        plt.scatter(4.5, 7.5, color='yellow', marker='*', s=100)
        plt.scatter(6.5, 5.5, color='yellow', marker='*', s=100)
        plt.scatter(6.5, 9.5, color='yellow', marker='*', s=100)
        plt.scatter(8.5, 7.5, color='yellow', marker='*', s=100)
        plt.scatter(9.5, 4.5, color='yellow', marker='*', s=100)
        plt.scatter(2.5, 5.5, color='white', marker='*', s=100)
        plt.scatter(5.5, 4.5, color='red', marker='*', s=100)
        # 제목 추가
        if i == 0:
             plt.title('Current Goal',fontweight="bold",fontsize=20)
        elif i==1:
             plt.title('Old Goal',fontweight="bold",fontsize=20)
        elif i==2:
             plt.title('Sub Goal',fontweight="bold",fontsize=20)
        elif i==3 :
             plt.title('Exploration',fontweight="bold",fontsize=20)
        # 그림 저장 (DPI 설정 추가)
        plt.savefig(address+f'/heatmap_goal_{i}.png', dpi=400)
        # 그림 표시
        plt.close()

monkey = "p"

address = 'irl/p/day2/3_9_map2_time23_bias[25, 25]'

if __name__ == "__main__" :

    #for i in range(time) :
        #make_dirl_heatmap(i, 3.4)


    make_dirl_goal_map(n_maps=2)

