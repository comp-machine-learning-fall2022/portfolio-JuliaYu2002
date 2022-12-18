import numpy as np
from scipy.spatial import distance
from sklearn.cluster import KMeans

def full_kmeans(data, k):
    """
    full_kmeans: runs a kmeans instance on some data utilizing the builtin sklearn function
    imput: a numpy array data, and the number of clusters k
    output: the cluster labels and the cluster centers
    """
    kmeans_setting = KMeans(n_clusters = k, init = "random", random_state = 1, max_iter = 200)
    data_fit = kmeans_setting.fit(data)
    
    return (data_fit.labels_, data_fit.cluster_centers_)

def make_adj(np_arr):
    """
    make_adj: makes an adjacency matrix depending on the distances of points from each other
    input: a numpy array
    output: a numpy array of only 0s and 1s
    """
    dist_arr = distance.cdist(np_arr, np_arr, 'euclidean')
    # put a placeholder for the zeros so they dont change
    get_zero = dist_arr == 0
    dist_arr[get_zero] = 2
    # get the stuff under a half (not already 0s) and set to 1
    less_half = dist_arr < .5
    dist_arr[less_half] = 1
    # get the rest of the stuff and set it to 0
    not_half = dist_arr != 1 # set these to 0
    dist_arr[not_half] = 0

    return dist_arr #np arr of only 0s and 1s

def my_laplacian(adj):
    """
    my_laplacian: makes a laplacian array by getting the 1s in the adjacency array, putting them into a new array diagonalized, and subtracting the adjacency array from this result
    input: an adjacency numpy array
    output: the laplacian numpy array
    """
    A = adj
    D = np.zeros(A.shape[0])

    for point_row in range(A.shape[0]): #loop through rows for points
        count = np.count_nonzero(A[point_row] == 1)
        D[point_row] = count
    D = np.diag(D)

    return np.subtract(D, A) #laplacian unnormalized arr

def spect_clustering(lap, clusters):
    """
    spect_clustering: makes a reduced matrix with eigenvectors/values
    input: the laplacian numpy array lap, and the number of clusters k
    output: the cluster labels and their centers
    """
    eig_vals, eig_vecs = np.linalg.eig(lap)
    inds = (np.abs(eig_vals)).argsort()
    eig_vals = eig_vals[inds]
    vecs_asc = eig_vecs[:,inds]
    trim_vecs = vecs_asc[:, :clusters] # first k eigenvectors

    km_alg = KMeans(n_clusters=clusters, init="random",random_state = 1, max_iter = 200)
    vecs_fit = km_alg.fit(trim_vecs)

    return (vecs_fit.labels_, vecs_fit.cluster_centers_)
