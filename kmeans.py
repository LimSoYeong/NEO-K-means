import random
import copy
import math

def euclidean_dist(A, B):
    assert len(A) == len(B) # 길이가 맞는지 확인
    sum = 0
    for a, b in zip(A, B):
        sum += (a-b)**2
    return math.sqrt(sum)

def calc_vector_mean(array):
    summation = [0] * len(array[0])
    for row in array:
        for d in range(len(row)):
            summation[d] += row[d]
    for s in range(len(summation)):
        summation[s] = summation[s]/len(array)
    return summation

def calc_diff(A, B):
    tol = 0
    for a, b in zip(A,B):
        tol += euclidean_dist(a, b)
    return tol

class Kmeans(object):
    def __init__(self, k, max_iter=10, tol=1e-4):
        '''
        입력
        - k : 클러스터 수
        - max_iter = iteration 최대 수
        - tol = Update Tolerance, centroid 업데이트가 tol보다 작아지면 fit 중단
        '''
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def random_init(self, array):
        '''
        K-means ++ 사용
        입력 : 전체 데이터
        '''
        M = [] # Indices of initial points
        indices = []
        i = random.randint(0, len(array)-1)
        print(i)
        print(array[i])
        M.append(array[i])
        indices.append(i)
        while len(M) < self.k:
            max_dist = - float('inf')
            max_index = -1
            for i in range(len(array)):
                avg_dist = 0
                if i in indices:
                    continue
                for j in range(len(M)):
                    dist = euclidean_dist(array[i], array[j])
                    avg_dist += dist
                avg_dist /= len(M)
                if max_dist < avg_dist:
                    max_dist = avg_dist
                    max_index = i

            M.append(array[max_index])
            indices.append(max_index)
        return M

    def fit(self, X):
        '''
        입력 :  Array of [10000, 16]
        assignments : 각 벡터의 클러스터 할당
        centroids
        '''
        self.centroids = self.random_init(X)
        for iter in range(self.max_iter):
            print(f'{iter+1} iteration..')
            ## Assign Cluster
            self._assign_cluster(X)
            ## Update Centroids
            self._update_centroids(X)

            if calc_diff(self.prev_centroids, self.centroids) < self.tol:
                break
        return self.assignments, self.centroids

    def _assign_cluster(self, X):
        '''데이터를 군집 번호로 할당'''
        self.assignments = []
        for d in X:
            min_dist = float('inf')
            min_index = -1
            for i, centroid in enumerate(self.centroids):
                dist = euclidean_dist(d, centroid)
                if dist < min_dist:
                    min_dist = dist
                    min_index = i
            self.assignments.append(min_index)

    def _update_centroids(self, X):
        self.prev_centroids = copy.deepcopy(self.centroids)
        for i in range(self.k):
            data_indices = list(filter(lambda x: self.assignments[x] == i, range(len(self.assignments))))

            if len(data_indices) == 0:
                r = random.randint(0, len(X))
                self.centroids[i] = X[r]
                continue

            cluster_data = []
            for index in data_indices:
                cluster_data.append(X[index])
            self.centroids[i] = calc_vector_mean(cluster_data)