import matplotlib.pyplot as plt
import numpy as np
def plot_final_weights(DIR_NAME, rec_weights, n_maps) :


    labels = ['Current Goal', 'Old Goal', 'Sub Goal', 'Free']
    colors = ['red','blue','green','purple']

    time = rec_weights.shape[1]

    if n_maps >3 :
        fig = plt.figure(figsize=(14, 4))
    else :
        fig = plt.figure(figsize=(10, 6))
    
    for map in range(n_maps) :
        plt.subplot(1,n_maps,map+1)
        plt.plot(rec_weights[map], colors[map], linewidth=1.5,
                    linestyle="--", zorder=2, label=labels[map])
        plt.axhline(y=0, color='k', alpha=0.6, linestyle="--", linewidth=0.5)
        plt.yticks([-1, 0, 1], fontsize=10)
        plt.xticks(range(0,time,5),fontsize=10)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel('Time [step]', fontsize=15)
        plt.title('Weight for '+labels[map], fontsize=15)

    plt.tight_layout() 
    #plt.legend()
    fig.savefig('figures/'+DIR_NAME+'.png')    
