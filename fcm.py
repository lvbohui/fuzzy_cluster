import numpy as np
from sklearn import datasets

def fcm(data, category_num, m=2, thresh=100):
    membership = np.random.random((len(data), category_num))
    membership = np.divide(membership, np.sum(membership, axis=1)[:, np.newaxis])

    while True:
        working_membership = membership ** m
        centroids = np.divide(np.dot(working_membership.T, data), 
                              np.sum(working_membership.T, axis=1)[:, np.newaxis])
        
        n_c_distance = np.zeros((len(data), category_num))

        for i, x in enumerate(data):
            for j, c  in enumerate(centroids):
                n_c_distance[i][j] = np.linalg.norm(data-c, 2)

        new_membership = np.zeros((len(data), category_num))
        for i, x in enumerate(data):
            for j, c in enumerate(centroids):
                new_membership[i][j] = 1./(np.sum((n_c_distance[i][j] / n_c_distance[i]) ** (2/(m-1))))
        if np.sum(abs(new_membership - membership)) < thresh:
            break
        membership = new_membership
    return centroids


if __name__ == "__main__":
    iris = datasets.load_iris()

    print(fcm(iris.data, 3))