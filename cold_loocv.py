# LOOCV implementation, check lab 13

# Find the number of data points in your data
n_data = mystery_np.shape[0]

# Initialize the list of errors
test_errors = []

# Loop over all points in the data set, letting each act as the test set
for i in range(n_data):
    
    # Split data into train and test
    test_data = mystery_np[[i], :]
    train_inds = list(set(range(n_data)).difference(set([i])))
    train_data = mystery_np[train_inds, :]
    
    # Create and train a model
    lm= linear_model.LinearRegression()
    mod = lm.fit(train_data[:,1:4], train_data[:,0])
    
    # Compute the testing error and add it to the list of testing errors
    preds = mod.predict(test_data[:,1:4])
    t_error = compute_mse(test_data[:,0],preds)
    
    test_errors.append(t_error)
    

# Compute the cross-val error
np.mean(test_errors)