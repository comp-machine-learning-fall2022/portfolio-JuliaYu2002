import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans

def cold_means(np_arr, k, state, loop_num): #mykmeans, hw2
    data_pd = pd.DataFrame(np_arr)
    centers = data_pd.sample(k, random_state = state)
    centers_np = centers.to_numpy()
    for a in range (loop_num):
        dists = distance.cdist(np_arr, centers_np, 'euclidean')
        clusters = np.argmin(dists, axis=1)
        for i in range(k):
            centers_np[i, :] = np.mean(np_arr[clusters[:] == i], axis = 0)
    return (centers_np, clusters)

def dataProcess(data, columns):
    """
    takes in a selection of columns to normalize the contents of
    """
    # note: data is a numpy array
    # normalizes the data from the selected columns
    data_pd = pd.DataFrame(data)
    copy_data = data_pd[columns]
    data_np = copy_data.to_numpy()
    for i in range (len(columns)): # lab 4 code
        currCol = data_np[:,i]
        # get the min and max of this column
        mx = np.max(currCol)
        mn = np.min(currCol)
        # get the normalized numbers and round
        norm = (currCol - mn)/(mx - mn)
        norm = np.around(norm, decimals = 2)
        data_np[:,i] = norm #set the current column to be the resulting normalized numbers
    return data_np

def looping_kmeans(data, k_vals):
    arr = data
    goodness = np.zeros((len(k_vals)))
    for inds in range(len(k_vals)):
        i = k_vals[inds]
        km_alg = KMeans(n_clusters=i, init="random",random_state = 1, max_iter = 200)
        
        arr_fit = km_alg.fit(arr)
        arr_centers = arr_fit.cluster_centers_
        c_total = 0
        for cluster in range(i):
            c = distance.cdist(arr[arr_fit.labels_ == cluster], arr_centers[[cluster]], 'euclidean')
            c_total += np.sum(c)
        goodness[inds] = c_total
    return goodness.tolist()
