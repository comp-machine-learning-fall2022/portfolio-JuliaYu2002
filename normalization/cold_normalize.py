import numpy as np
import pandas as pd

def cold_normalization(data_name, norm_cols):
    data_pd = pd.read_csv(data_name, sep = ",")

    for col in norm_cols:
        mx = np.max(data_pd[col])
        mn = np.min(data_pd[col])

        norm = (data_pd[col] - mn) / (mx - mn)
        round_norm = np.around(norm, decimals = 2)
        data_pd[col] = round_norm
    return data_pd.to_numpy()