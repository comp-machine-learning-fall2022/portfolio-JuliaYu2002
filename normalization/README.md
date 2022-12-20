# Normalization

### What is Normalization and How is it Calculated?

The equation to calculate a normalized column is the following:

$Var_{norm} = \dfrac{Var - Var_{min}}{Var_{max} - Var_{min}}$

Normalization changes all values in the dataframe to be between 0 and 1.

## The Cold Coded Normalization

My implementation of normalization takes in a file name and the columns that the user desires to have normalized as a list. It reads in the data as a Pandas dataframe and then loops through the columns in the list and calculates the minimum and maximum of the column. Then it calculates the normalized numbers for the column. It then returns this as a Numpy array.

## The Sklearn [Normalization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)

The Sklearn function that normalizes data is called `Min_Max_Scalar`. In order to use it, first an instance needs to be initialized, then the instance needs to be fit to the data and then the data needs to be transformed for it to have any effect. Alternatively, the fit and transformation can be done in the same step through `fit_transform()`.

# The Results

The `%%timeit` time for my implementation was 9.3 ms with a standard deviation of 303 µs. The Sklearn implementation runs in 88.3 µs with 30 µs as standard deviation. Both are run 7 times with 15 loops each.

`%%memit` being run on my implementation shows that the cold coded function takes .03 MiB. For the Sklearn implementation, it is shown to be 0 MiB which likely means that its much smaller than the $2^{20}$ bytes that a MiB is made up of.

Both the amount of time and amount of memory it takes to normalize an array is smaller when using the Sklearn implementation, so it's more beneficial to use the inbuilt function than using a cold implementation as it saves time and space.