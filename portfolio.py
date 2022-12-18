import numpy as np
import time
import timeit
import line_profiler
import memory_profiler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.preprocessing import StandardScaler

# data: https://www.kaggle.com/datasets/mariotormo/complete-pokemon-dataset-updated-090420?select=pokedex_%28Update_05.20%29.csv
# note: file name originally "pokedex_(Update_05.20)", changed to "pokedex" since its shorter

""" possible import statements for functions, stick the links about the functions i end up comparing in the add statement istg
https://scikit-learn.org/stable/modules/classes.html

from sklearn.model_selection import KFold https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

from sklearn.model_selection import cross_validate https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html

import time # stuff on time is required to do comparisons, rely on lab 19 for this stuff
import timeit
import line_profiler
import memory_profiler

from sklearn.cluster import KMeans https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html or https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html

from sklearn.cluster import SpectralClustering https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html :: loocv
this one is more split the data and then you'd have to do all the actual cross val stuff seperately

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.laplacian_kernel.html :: laplacian (hw3)?

https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html :: mse!!

https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html :: standardizing data!
"""

""" course goals to highlight (pick 4 of them plz) probs going with 1, 2, 3, 5?
1: Detail differences between supervised and unsupervised learning tasks and methods

2: Implement a variety of machine learning algorithms in python and assess their efficacy

3: Compare and assess the efficacy of machine learning algorithms and results using evaluation metrics and in terms of the context of the data's domain

4: Develop an appreciation for ethical implications of machine learning algorithms
can this use decision trees and their drawbacks?

5: Work iteratively and reflectively to apply machine learning techniques to a data set of interest with informative documentation, written for a variety of audiences
stuff from my project 2
"""
