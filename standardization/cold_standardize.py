import numpy as np
import pandas as pd

def cold_standard(data_name, stand_cols):
    """
    cold_standard: standardizes data in given columns
    input: a filename and a selection of columns to normalize the contents of as a list
    output: a numpy array with standardized columns
    """
    data_pd = pd.read_csv(data_name, sep = ",")
    data_std = data_pd.copy()

    for col in stand_cols:
        mean_vec = np.mean(data_pd[col], axis = 0)
        sd_vec = np.std(data_pd[col], axis = 0)
        sd_vec = np.std(data_pd[col], axis = 0)
        data_std[col] = (data_pd[col] - mean_vec * np.ones(len(data_pd))) / sd_vec

    return data_std.to_numpy()