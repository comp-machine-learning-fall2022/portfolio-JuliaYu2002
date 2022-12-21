# Kfold Cross Validation

### What is Kfold?

Kfold is a way to split data into groups such that each group can be used as a test set once while everything else is used to train a model. This occurs when the number of groups is not equal to the total number of data points. It's used in cross validation.

###  What is Cross Validation?

Cross validation is a strategy to find the error that a specific model has when predicting values of data that it has never seen before.

## The Cold Coded Kfold Cross Validation (Using a Decision Tree)

My cold implementation of Kfoldcv takes in a Numpy data array, all the column names, the input columns and output column, and the number of folds to split the data into.

After splitting the data into groups, it then will loop through each group such that each group of data would serve as the test set once (which is the Kfold part of the function). Using the rest of the data as the training set, it fits the decision tree with this data and predicts the output of the test data.

These predictions are then compared to the actual values via a classification mean squared error, which get placed in a list. After all the folds have been used as test once, it returns the mean of all the mses.

## The Sklearn [Kfold Cross Val](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)

The Sklearn function for Kfoldcv uses an estimator (for example, an initialized decision tree, linear regression, etc.), the input data, the output data, how to analyze the results (the type of analysis to perform), the number of folds, and various other parameters.

This only requires an initialized estimator to work, and it returns a Numpy array of the cross val results.

# The Results

When using `%%timeit`, the cold coded function runs in 17.1 ms with a standard deviation of 1.31 ms. The Sklearn implementation runs in 18.6 ms with a 1.26 ms standard deviation.

The memory used by both my cold coded function and the Sklearn one come up as 0 MiB as shown by `%memit`, which really means that the both of them use under $2^{20}$ bytes.

By virtue of my cold coded function running in slightly less time, it seems to be the better option. Though it might be interesting to note that when using `%%time`, the amount of time used by the CPU is higher when using the cold coded one at 31.2 ms, while the Sklearn implementation uses 15.6 ms. So the differences are small but with some computational differences.


<!-- ## The Sklearn [Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

## The Sklearn [Classification MSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)

## The Sklearn [Mean Squared Error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) -->