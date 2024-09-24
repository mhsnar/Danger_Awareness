import numpy as np

# Load the array
s = np.load('P_xH_all_MP.npy')

# Loop through each s[i, :, :, :] and count the elements greater than 0.1
for i in range(s.shape[0]):
    count = np.sum(s[i, :, :, :] > 0.1)
    print(f"Number of elements greater than 0.1 in s[{i}, :, :, :]: {count}")
