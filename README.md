# Cold Code or Library Function: Which is Better?

## An examination of speed and memory.

Sklearn supplies us with functions and features that allow us to implement many Machine Learning techniques through a less long winded sequence of lines, however I wondered if this was truly the best way to do this since time is money in industry.

*As a precursory note*, I am referencing the results of `%%timeit`, `%%time`, and `%%memit` from my laptop. These results would likely differ based on the specs of the machine that the code is run on, so the numbers would likely not be the same but possibly similar.



# References

- https://scikit-learn.org/stable/modules/classes.html
  - Documentation about all classes in Sklearn (function specific below)

- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html
  - Cross Val Score using Kfold

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
  - Accuracy score, an mean squared error variant

- https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html
  - KMeans

- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
  - MinMaxScalar, otherwise known as a data normalizer

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
  - Mean squared error, or MSE

- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
  - StandardScalar, a data standardizer

- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  - Train_test_split, separates data into a train set and test set

- https://stackoverflow.com/questions/45318536/trying-to-understand-python-memory-profiler

- https://jakevdp.github.io/PythonDataScienceHandbook/01.07-timing-and-profiling.html
  - The above 2 links elaborate more upon the timing and memory logging functions I use

- https://www.techtarget.com/searchstorage/definition/mebibyte-MiB
  - This is what makes up a MiB, referenced in `%%timeit`

- https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions
  - Why the calculations from accuracy_score make sense

- https://www.kaggle.com/datasets/mariotormo/complete-pokemon-dataset-updated-090420?select=pokedex_%28Update_05.20%29.csv
  - The data I use in the Jupyter Notebooks
    - The original name was "pokedex_(Update_05.20)", but I changed it to "pokedex" since its shorter