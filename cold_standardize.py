# lab 5

mean_vec = np.mean(justtwo, axis=0)
sd_vec = np.std(justtwo, axis=0)

justtwo_std = justtwo.copy()

for i in range(justtwo.shape[1]):
    justtwo_std[:,i] = (justtwo[:,i] - mean_vec[i]*np.ones(justtwo.shape[0]))/sd_vec[i]
#     print(justtwo_std[:,i])
justtwo