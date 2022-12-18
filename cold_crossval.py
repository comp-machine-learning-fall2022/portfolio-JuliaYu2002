import pandas as pd
import numpy as np
from sklearn import linear_model

def cold_mse(truth_vec, predict_vec):
    """
    cold_mse: calculates the mse between the truth and predicted values
    input: a truth numpy array and a prediction numpy array
    output: the mean squared error
    """
    return np.mean((truth_vec - predict_vec)**2)

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
    data_groups = np.array_split(data, k) # list of numpy arrays
    in_vals = []
    for x in inputs:
        in_loc = col_names.index(x)
        in_vals.append(in_loc)
    
    out_location = col_names.index(output)
    
    errors = []

    for i in range(k):
        test = data_groups.pop(i)
        train = np.vstack(data_groups)
        
        lm_fit = linear_model.LinearRegression()
        mod_A = lm_fit.fit(train[:, in_vals], train[:, [out_location]])
        test_predictions = mod_A.predict(test[:, in_vals])
        errors.append(cold_mse(test_predictions, test[:, [out_location]]))
        
        data_groups.insert(i, test)
    return np.mean(errors)