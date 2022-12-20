# Standardization

The equation to calculate a standardized column is the following:

$Var_{standard} = \dfrac{Var - Var_{mean}}{Var_{sd}}$

## The Cold Coded Standardization

My cold coded function takes in a file name and a list of columns to standardize. It makes a Pandas dataframe and loops through the column list to access each column to standardize. It does this by calculating the mean and standard deviations of the column and doing the math in the equation. After it finishes all the columns listed, it returns the data as a Numpy array.

## The Sklearn [Standardization](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

The Sklearn module has its own standardizing function which requires the initialization of an instance and then this would be fit to the data. Additional things that can be pulled from this are the means of the data and the standard deviations which are accessible with dot notation. 

# The Results

Using `%%timeit` the time recorded for my cold function was 12.1 ms with a standard deviation of 972 µs, while the time for the Sklearn function was 309 µs with a standard deviation of 48.2 µs.

The memory use recorded by `%memit` for the cold function was .38 MiB. Similar to the normalize feature of Sklearn, the memory used by the standardizing Sklearn function is shown to be 0 MiB, implying that it is actually much lower than $2^{20}$ bytes.

With this, I'm able to say that it's much better to use the standardize Sklearn function due to it's speed and low memory usage. It's also more convenient since you can access the mean and standard deviations of the columns standardized.