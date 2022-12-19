import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans

def cold_means(data_name, k, state, cols, loop_num):
    """
    cold_means: gets the labels and cluster centers of a dataframe
    input: a filename, the number of clusters k, a random state for replication, a list of 2 columns, 
            and the number of times to loop to find more accurate centers
    output: the centers of each of the clusters and the labels of each of the points in the dataset
    """
    data_pd = pd.read_csv(data_name, sep = ",")
    data_pd = data_pd[cols]
    data_np = data_pd.to_numpy()
    centers = data_pd.sample(k, random_state = state)
    centers_np = centers.to_numpy()

    for current_loop in range(loop_num):
        dists = distance.cdist(data_np, centers_np, 'euclidean')
        clusters = np.argmin(dists, axis=1)
        for i in range(k):
            centers_np[i, :] = np.mean(data_np[clusters[:] == i], axis = 0)

    return (centers_np, clusters)