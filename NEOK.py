import numpy as np


def estimate_alpha_beta(X, C, alpha_delta=1.1, beta_delta=6): # X(n*m), C(k*m) : 행렬 /  delta :상수 값
  n = X.shape[0]        # 데이터 개수
  k = C.shape[0]        # centroid 수
  U = np.zeros((n, k))  # 행렬 U 생성 *distnace 저장

  # 거리 계산
  for j in range(k):                              # 모든 (i,j)에 대해 거리가 게산되어 행렬 U 완성
    diff = X - np.tile(C[j, :], (n, 1))           # 센트로이드 c_j의 좌표 C[j, :]가 (n,1)차원만큼 반복하여 쌓는다. -> 모든 데이터 좌표와 거리 계산
    U[:, j] = np.sqrt(np.sum(diff**2, axis=1))    # 유클리디안 거리 행렬에 삽입

  dist = np.min(U, axis=1)      # 각 행(데이터)별로 가장 가까운 클러스터 중심과의 거리를 가져와서 dist 배열에 저장 / (N, 1) 형태의 2차원 배열
  ind = np.argmin(U, axis=1)    # 각 행(데이터)별로 가장 가까운 클러스터의 인덱스를 가져와서 ind 배열에 저장 / (N, 1) 형태의 2차원 배열

  # beta 추정
  betaN = np.count_nonzero(dist > (np.mean(dist) + beta_delta * np.std(dist)))  # 이상치로 간주되는 데이터 개수
  beta = betaN / n    # 이상치 개수 / 전체 데이터 개수

  # alpha 추정 - 겹치는 부분이 작은 경우일 때의 임계치를 기준으로 추정
  overlap = 0     # 겸치는 개수

  for j in range(k) :
    if np.count_nonzero(ind == j) > 0:    # 클러스터에 할당되어 있다면,
      cdist = dist[ind == j]              # 중심과 클러스터에 속한 인스턴스들 사이의 거리만 가져오기
      threshold = np.mean(cdist) + alpha_delta * np.std(cdist)
      v = (U[ind != j, j] <= threshold)   # 클러스터 j에 속하지 않는 임의의 인스턴스 v와 클러스터 J 사이의 거리 < 임계치 -> True -> v는 overlapped region에 위치한 것으로 간주
      overlap += np.count_nonzero(v)

  alpha = overlap / n    # 겹치는 개수 / 전체 데이터 개수
  return alpha, beta


def neo_kmeans(X, k, alpha, beta, initU):
    # NEO-K-Means (Non-exhaustive and Overlapping clustering)

    N = X.shape[0]   # no. of data points
    dim = X.shape[1]  # dimension
    U = np.array(initU.copy())  # initial U
    t = 0
    t_max = 200
    alphaN = int(round(alpha * N))
    betaN = int(round(beta * N))
    J = float('inf')
    oldJ = 0
    epsilon = 0
    J_track = []

    D = np.zeros((N, k))
    M = np.zeros((dim, k))

    while abs(oldJ - J) > epsilon and t <= t_max:
        oldJ = J
        J = 0

        for j in range(k):
            ind = U[:, j].astype(bool)
            M[:, j] = np.mean(X[ind, :], axis=0)

        for j in range(k):
            diff = X - np.tile(M[:, j], (N, 1))
            D[:, j] = np.sum(diff**2, axis=1)

        ind = np.argpartition(D, N - betaN, axis=None)[:N - betaN]
        ind_flat = np.unravel_index(ind, D.shape)
        sorted_d = D[ind_flat]
        sorted_n, sorted_k = ind_flat
        num_assign = N - betaN
        J += np.sum(sorted_d[:num_assign])
        U = np.zeros((N, k))
        U[sorted_n[:num_assign], sorted_k[:num_assign]] = 1
        D[sorted_n[:num_assign], sorted_k[:num_assign]] = np.inf

        n = 0
        while n < (alphaN + betaN):
            min_d = np.min(D)
            J += min_d
            i_star, j_star = np.unravel_index(np.argmin(D), D.shape)
            U[i_star, j_star] = 1
            D[i_star, j_star] = np.inf
            n += 1

        t += 1
        J_track.append(J)
        print(f'***** iteration: {t}, objective: {J:.6f}')

    print(f'***** No. of iterations done: {t}')

    # display results
    print(f'***** Total no. of data points: {N}')
    print(f'***** alpha: {alpha:.3f}, alphaN: {alphaN}')
    print(f'***** beta: {beta:.3f}, betaN: {betaN}')

    return U, J, J_track