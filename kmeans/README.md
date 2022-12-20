# KMeans

## The Cold Coded KMeans

My implemenation of KMeans takes a file name, the number of clusters desired, a random state, a list of columns to get the relationship between, and the max number of loops to find the "most accurate" center points that the data lie around.

It reads in the data as a Pandas dataframe creates some Numpy arrays from it to work. Then using a nested for loop, it obtains the distances of all points to the centers and updates the centers accordingly. After it finishes looping, the centers of the clusters and the labels showing what cluster each point belongs to is returned as a tuple.

## The Sklearn [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html)

The Sklearn KMeans implementation is used by declaring an instance of itself with various parameters such as the number of clusters, a random state, the method for selecting the starting centers, and the maximum iterations before the fitting stops and returns the centers.

After creating the KMeans instance, it has to be fit to the data in order to actually use it. This stores the clusters and the point labels which are accessable through dot notation.

# The Results

Using `%%timeit`, my implementation runs faster at a rate of 9.13 ms with a standard deviation of 347 µs. The Sklearn implementation runs at 25.1 ms with a standard deviation of 1.38 ms. Both were ran 7 times and each with 15 loops to keep things constant.

Looking at the amount of memory used by both processes by using `%%memit`, my implementation uses .63 MiB while the Sklearn module uses .04 MiB.

With my implementation taking more memory but also taking less time, it's dependent on which a person would want to conserve more of in the long run.