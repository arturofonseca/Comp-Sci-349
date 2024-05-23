from pprint import pprint
from math import sqrt
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# returns Euclidean distance between vectors a dn b
def euclidean(a: list[str], b: list[str]) -> float:
    sum_squared: int = 0
    for a_i, b_i in zip(a, b):
        a_i, b_i = int(a_i), int(b_i)
        # mapping grayscale to binary
        if a_i > 0:
            a_i = 1
        if b_i > 0:
            b_i = 1
        sum_squared += (a_i - b_i) ** 2
    return sum_squared ** 0.5


# returns Cosine Similarity between vectors a dn b
def cosim(a: list[str], b: list[str]) -> float:
    dot, mag_a, mag_b = 0, 0, 0
    for a_i, b_i in zip(a, b):
        a_i, b_i = int(a_i), int(b_i)
        # grayscale (0-255) to binary (0, 1) reduction
        if a_i > 0:
            a_i = 1
        if b_i > 0:
            b_i = 1
        # summing dot products and magnitudes
        dot += a_i * b_i
        mag_a += a_i
        mag_b += b_i
    # magnitude of a and b
    mag_a **= 0.5
    mag_b **= 0.5
    return dot / (mag_a * mag_b)


# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "Euclidean" or "cosim".
# all hyperparameters should be hard-coded in the algorithm.
def knn(train: list[list], query: list[list], metric: str = "euclidean", k: int = 1) -> list[list]:
    """
    training data format:
    [
      [label, [attribute1,..., attribute_n]]
    ]
    query format:
    [
      [attribute1,..., attribute_n]
    ]
    metric:
    - 'euclidean' or 'cosim'
    output format:
    [
        [label, [attribute1,..., attribute_n]]
    ]
    """
    # distance function
    if metric == 'euclidean':
        distance = euclidean
    else:
        distance = cosim
    # hyper-parameter k
    K = k
    # return query list
    query_predictions = []

    # function for predicting label of one example given a dataset
    def predict(query_point: list[int]) -> int:  # i ∈ {0, 1,..., 9}
        vec_dist_list = []
        # iterate through every point and find its distance to query_point
        # training_point = [label, [attr1,...,attr_n]]
        for train_point_label, train_point_vector in train:
            dist = distance(query_point, train_point_vector)  # want only the vector (index 1)
            vec_dist_list.append((train_point_label, dist))  # appending tuple of [label, distance]

        # if cosim, sort in decreasing order: cos(θ)=1 is closer than cos(θ)=-1 where θ = angle b/w a and b
        vec_dist_list.sort(key=lambda x: x[1], reverse=(metric == 'cosim'))  # sort by distance

        k_labels = Counter([point[0] for point in vec_dist_list[:K]])  # extract k labels
        return k_labels.most_common(1)[0][0]  # return most common

    for query_point in query:
        predicted_query_label = predict(query_point)
        # append list [label, attributes vector]
        query_predictions.append([predicted_query_label, query_point])

    # return updated query points
    return query_predictions


# returns a list of labels for the query dataset based upon observations in the train dataset
# labels should be ignored in the training set
# metric is a string specifying either "Euclidean" or "cosim".
# All hyperparameters should be hard-coded in the algorithm.
def kmeans(train: list[list], query: list[list], metric: str = "euclidean") -> list[list]:
    # distance function
    if metric == 'euclidean':
        distance: function = euclidean
    else:
        distance: function = cosim
    # hyper-parameter k
    K = 10  # 10 digits
    # initialize the means to be first K training attribute vectors
    means = [train[i][1] for i in range(K)]
    # return query list
    query_predictions = []

    # calculate the distance to means from every point
    for query_point in query:
        pass
    '''
    ALGORITHM
    1. Choose the number of clusters(K) and obtain the data points 
    2. Place the centroids c_1, c_2, ..... c_k randomly 
    3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
    4. for each data point x_i:
       - find the nearest centroid(c_1, c_2 .. c_k) 
       - assign the point to that cluster 
    5. for each cluster j = 1..k
       - new centroid = mean of all points assigned to that cluster
    6. End 
    '''
    ...
    return query_predictions


def read_data(file_name):
    data_set = []
    with open(file_name, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i + 1])
            data_set.append([label, attribs])
    return data_set


def show(file_name, mode):
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ', end='')
                else:
                    print('*', end='')
            else:
                print('%4s ' % data_set[obs][1][idx], end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0], end='')
        print(' ')


def main():
    show('valid.csv', 'pixels')


if __name__ == "__main__":
    size = 500
    k = 10
    train = read_data("train.csv")[:size]
    query = [point[1] for point in train][:size]

    # K NEAREST NEIGHBORS -------------------------------
    # TODO: Remove k parameter from knn once done testing
    for metric in ['euclidean', 'cosim']:
        print()
        print(metric.upper())
        print('-' * 15)
        for i in range(1, k + 1):
            accurate = 0
            predicted_query = knn(train, query, metric, i)
            for j in range(size):
                if train[j][0] == predicted_query[j][0]:
                    accurate += 1
            print(f"{accurate / size:.0%} accuracy (k={i}, size={size})")

    # K MEANS -------------------------------------------
    ...
