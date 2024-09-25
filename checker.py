import numpy as np

def checker(arr):
    (n, m) = np.shape(arr)
    for i in range(n):
        for j in range(m):
            sv = -np.mod(i+j, 2)
            if sv == 0: sv = 1
            arr[i][j] = sv * arr[i][j]
    return arr