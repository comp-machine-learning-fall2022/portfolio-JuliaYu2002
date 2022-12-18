import numpy as np

def cold_standard(data_name):
    data_np = np.genfromtxt(data_name, delimiter = ",")
    mean_vec = np.mean(data_np, axis = 0)
    sd_vec = np.std(data_np, axis = 0)

    data_std = data_np.copy()

    for i in range(data_np.shape[1]):
        data_std[:,i] = (data_np[:,i] - mean_vec[i] * np.ones(data_np.shape[0])) / sd_vec[i]
    return data_std