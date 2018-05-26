"""
https://www.kaggle.com/mindcool/unrolling-of-helices-outliers-removal?scriptVersionId=3755863
"""

import numpy as np

if __name__ == "__main__":

    x = np.array([(62, 397, 103), (82, 347, 107), (93, 288, 120),
                  (94, 266, 128), (65, 163, 169), (12, 102, 198),
                  (48, 138, 180), (77, 187, 157), (85, 209, 149), (89, 316, 112)])

    xm = np.mean(x)

    X = (x - xm).T

    v, s, t = np.linalg.svd(X, full_matrices=True)

    sigma1 = s[0]
    sigma2 = s[1]
    sigma3 = s[2]
    v1 = t[0]
    v2 = t[1]
    v3 = t[2]

    Z = np.zeros((x.shape[0], 10), np.float32)
    Z[:, 0] = x[:, 0] ** 2
    Z[:, 1] = 2 * x[:, 0] * x[:, 1]
    Z[:, 2] = 2 * x[:, 0] * x[:, 2]
    Z[:, 3] = 2 * x[:, 0]
    Z[:, 4] = x[:, 1] ** 2
    Z[:, 5] = 2 * x[:, 1] * x[:, 2]
    Z[:, 6] = 2 * x[:, 1]
    Z[:, 7] = x[:, 2] ** 2
    Z[:, 8] = 2 * x[:, 2]
    Z[:, 9] = 1

    v, s, t = np.linalg.svd(Z,full_matrices=True)
    smallest_value = np.min(np.array(s))
    smallest_index = np.argmin(np.array(s))
    T = np.array(t)
    T = T[smallest_index,:]
    S = np.zeros((4,4),np.float32)
    S[0,0] = T[0]
    S[0,1] = S[1,0] = T[1]
    S[0,2] = S[2,0] = T[2]
    S[0,3] = S[3,0] = T[3]
    S[1,1] = T[4]
    S[1,2] = S[2,1] = T[5]
    S[1,3] = S[3,1] = T[6]
    S[2,2] = T[7]
    S[2,3] = S[3,2] = T[8]
    S[3,3] = T[9]
    norm = np.linalg.norm(np.dot(Z,T), ord=2)**2
    print(norm)