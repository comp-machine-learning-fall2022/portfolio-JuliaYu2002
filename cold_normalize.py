# lab 4
# Normalizing Coffee variable

coffee = justtwo_np[:,0] # all rows, only data in col 0
# get the min and max of this column
mx = np.max(coffee)
mn = np.min(coffee)

coffee_norm = (coffee - mn)/(mx - mn)
coffee_norm = np.around(coffee_norm, decimals = 2)
coffee_norm