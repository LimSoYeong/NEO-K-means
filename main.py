from NEOkmeans import *
from kmeans import *
import numpy as np

# 군집 수 k
def cluster(X, k):
    kmeans = Kmeans(k, max_iter=10, tol=1e-4)
    U, C = kmeans.fit(X)
    C = np.array(C)
    X = np.array(X)
    # initU 설정
    initU = [[0] * k for x in range(len(X))]
    for i in range(len(X)):
        initU[i][U[i]] = 1
    alpha, beta = estimate_alpha_beta(X, C, alpha_delta=1.1, beta_delta=6)
    NeoU, J, J_track = neo_kmeans(X, k, alpha, beta, initU)
    return NeoU