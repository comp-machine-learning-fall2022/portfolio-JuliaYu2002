import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def cold_mse(truth_vec, predict_vec):
    """
    cold_mse: calculates the mse between the truth and predicted values
    input: a truth numpy array and a prediction numpy array
    output: the mean squared error
    """
    return np.mean((truth_vec - predict_vec)**2)

def cold_classification(pred_class, class_truth):
    """
    cold_classification: calculates the number of right to wrong (or yes/no) in the predictions vs the actual
    input: 2 numpy arrays that have the predictions based on some data and the actual classifications
    output: the ratio of the classifications
    """
    wrong = np.sum(class_truth[:] != pred_class[:])
    return np.mean(wrong / class_truth.shape[0])

def data_reduction(dataset_file, lst):
    """
    data_reduction: finds all unique values in a column and replaces each of them with an integer
    input: a file name and a list of columns to transform
    output: a pandas dataframe with the reduced columns
    """
    data = pd.read_csv(dataset_file, sep=",")
    replace_val = 0

    for i in lst:
        current = data[i] # get the column in the data specified by the column list
        col_stuff = pd.unique(current)
        for item in range(col_stuff.shape[0]):
            data.loc[current == col_stuff[item], i] = replace_val
            replace_val += 1

    return data
				

def cold_kfoldcv(data, col_names, inputs, output, k):
    """
    cold_kfoldcv: calculates the mse of a dataset using kfold
    input: a numpy array, a list of column names, the input columns as a list, the output column, and the number of folds
    output: the mse average
    """
    data_groups = np.array_split(data, k)
    in_vals = []
    for x in inputs:
        in_loc = col_names.index(x)
        in_vals.append(in_loc)
    
    out_location = col_names.index(output)
    
    errors = []

    for i in range(k):
        test = data_groups.pop(i)
        train = np.vstack(data_groups)
        
        tree = DecisionTreeClassifier(ccp_alpha = 0.001, max_depth=3)
        tree.fit(train[:, in_vals], train[:, out_location].astype(int))
        predictions = tree.predict(test[:, in_vals])
        errors.append(cold_classification(predictions, test[:, [out_location]]))
        
        data_groups.insert(i, test)
    return np.mean(errors)