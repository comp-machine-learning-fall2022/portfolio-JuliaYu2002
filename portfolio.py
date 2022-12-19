import numpy as np
import time
import timeit
import line_profiler
import memory_profiler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# data: https://www.kaggle.com/datasets/mariotormo/complete-pokemon-dataset-updated-090420?select=pokedex_%28Update_05.20%29.csv
# note: file name originally "pokedex_(Update_05.20)", changed to "pokedex" since its shorter, put this in the readme ig

""" possible import statements for functions, stick the links about the functions i end up comparing in the add statement istg
https://scikit-learn.org/stable/modules/classes.html

import time # stuff on time is required to do comparisons, rely on lab 19 for this stuff
import timeit
import line_profiler
import memory_profiler

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html :: mse!!

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html :: standardizing data!
"""

""" use these with the matching files only if i end up having time
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html :: loocv
this one is more split the data and then you'd have to do all the actual cross val stuff seperately

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html :: laplacian (hw3)?

from sklearn.cluster import SpectralClustering https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
"""

""" course goals to highlight (pick 4 of them plz) probs going with 1, 2, 3, 5?
1: Detail differences between supervised and unsupervised learning tasks and methods

2: Implement a variety of machine learning algorithms in python and assess their efficacy

3: Compare and assess the efficacy of machine learning algorithms and results using evaluation metrics and in terms of the context of the data's domain

4: Develop an appreciation for ethical implications of machine learning algorithms

5: Work iteratively and reflectively to apply machine learning techniques to a data set of interest with informative documentation, written for a variety of audiences
"""
