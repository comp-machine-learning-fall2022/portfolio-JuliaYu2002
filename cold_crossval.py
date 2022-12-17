import pandas as pd
import numpy as np
from sklearn import linear_model

def compute_mse(truth_vec, predict_vec):
    return np.mean((truth_vec - predict_vec)**2)

def data_wrangle(dataset_file, lst):
    data = pd.read_csv(dataset_file, sep=",")
    replace_val = 0
    for i in lst:
        current = data[i] # get the column in the data specified by the column list
        col_stuff = pd.unique(current)
        for item in range(col_stuff.shape[0]):
            data.loc[current == col_stuff[item], i] = replace_val
            replace_val += 1
    return (data.columns.values, data.to_numpy().astype(float))
				

def kfold_CV(data, col_names, inputs, output, k):
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
        errors.append(compute_mse(test_predictions, test[:, [out_location]]))
        
        data_groups.insert(i, test)
    return np.mean(errors)