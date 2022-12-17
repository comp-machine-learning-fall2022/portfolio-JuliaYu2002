import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans

# 1
"""
full_kmeans:
this function runs a kmeans instance on some data
imput: the data numpy array and the number of clusters
output: the cluster labels and the cluster centers
"""
def full_kmeans(data, k):
    kmeans_setting = KMeans(n_clusters=k, init="random",random_state = 1, max_iter = 200)
    # for i in range (data.shape[1]): # lab 4 code
    #     currCol = data[:,i]
    #     # get the min and max of this column
    #     mx = np.max(currCol)
    #     mn = np.min(currCol)
    #     # get the normalized numbers and round
    #     norm = (currCol - mn)/(mx - mn)
    #     norm = np.around(norm, decimals = 2)
    #     data[:,i] = norm
    data_fit = kmeans_setting.fit(data)
    return (k - 1, data_fit.cluster_centers_)

# 2a
"""
make_adj:
this function makes an adjacency matrix depending
on the distances of points from each other
input: a numpy array
output: a numpy array of only 0s and 1s
"""
def make_adj(np_arr):
    # adjacency matrix
    dist_arr = distance.cdist(np_arr, np_arr, 'euclidean')
    # put a placeholder for the zeros so they dont change
    get_zero = dist_arr == 0
    dist_arr[get_zero] = 2
    # get the stuff under a half (not already 0s) and set to 1
    less_half = dist_arr < .5
    dist_arr[less_half] = 1
    # get the rest of the stuff and set it to 0
    not_half = dist_arr != 1 # set this stuff to 0
    dist_arr[not_half] = 0
    return dist_arr #np arr of only 0s and 1s

# 2b
"""
my_laplacian:
this function will make a laplacian array
it does this by getting the 1s in the adjacency array
then putting them into a new array diagonalized
and subtracting the adjacency array from this result
input: an adjacency numpy array
output: the laplacian numpy array
"""
def my_laplacian(adj):
    A = adj
    D = np.zeros(A.shape[0])
    for point_row in range(A.shape[0]): #loop through rows for points
        count = np.count_nonzero(A[point_row] == 1)
        D[point_row] = count
    D = np.diag(D)
    return np.subtract(D, A) #laplacian unnormalized arr

# 3
"""
spect_clustering:
this function makes a reduced matrix with eigenvectors/values
input: the laplacian numpy array and the number of clusters you want
output: the cluster labels and their centers
"""
def spect_clustering(lap, clusters):
    eig_vals, eig_vecs = np.linalg.eig(lap)
    inds = (np.abs(eig_vals)).argsort() #no minus because we don't want it descending
    eig_vals = eig_vals[inds]
    vecs_asc = eig_vecs[:,inds]
    trim_vecs = vecs_asc[:, :clusters] # first k eigenvectors
    km_alg = KMeans(n_clusters=clusters, init="random",random_state = 1, max_iter = 200)
    vecs_fit = km_alg.fit(trim_vecs)
    return (vecs_fit.labels_, vecs_fit.cluster_centers_)
