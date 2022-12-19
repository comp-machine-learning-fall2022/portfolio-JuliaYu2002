import numpy as np
from sklearn import linear_model

def cold_mse(truth_vec, predict_vec):
    """
    cold_mse: calculates the mse between the truth and predicted values
    input: a truth numpy array and a prediction numpy array
    output: the mean squared error
    """
    return np.mean((truth_vec - predict_vec)**2)

def cold_loocv(data_name):
    """
    cold_loocv: 
    """
    data = np.genfromtxt(data_name, delimiter = ",")
    n_data = data.shape[0]

    test_errors = []

    for i in range(n_data):
        
        test_data = data[[i], :]
        train_inds = list(set(range(n_data)).difference(set([i])))
        train_data = data[train_inds, :]
        
        lm = linear_model.LinearRegression()
        mod = lm.fit(train_data[:,1:4], train_data[:,0])
        
        preds = mod.predict(test_data[:,1:4])
        t_error = cold_mse(test_data[:,0], preds)
        
        test_errors.append(t_error)

    return np.mean(test_errors)