import numpy as np

def save_trial_weights(DIR_NAME, rec_weights, n_maps, time) :
    weight_mean=list()
    for map in range(n_maps) :
        weight_mean.append(np.mean(rec_weights[map]))
    weight_mean_numpy=np.array(weight_mean)
    np.save('figures/'+DIR_NAME+'weight_mean_numpy.npy',weight_mean_numpy)
    
