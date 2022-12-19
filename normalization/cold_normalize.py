import numpy as np
import pandas as pd

def cold_normalization(data_name, norm_cols):
    """
    cold_normalization: normalizes data in given columns
    input: a filename and a selection of columns to normalize the contents of as a list
    output: a numpy array with normalized columns
    """
    data_pd = pd.read_csv(data_name, sep = ",")

    for col in norm_cols:
        mx = np.max(data_pd[col])
        mn = np.min(data_pd[col])

        norm = (data_pd[col] - mn) / (mx - mn)
        round_norm = np.around(norm, decimals = 2)
        data_pd[col] = round_norm
        
    return data_pd.to_numpy()