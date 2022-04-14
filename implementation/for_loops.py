import numpy as np
import random
from scipy.spatial import distance
import time


def initialize_centroids(n_features, n_clusters, init):
    initial_centroids = np.empty((n_clusters, n_features))
    if init is None:
        initial_centroids.fill(0.0)
    return initial_centroids


def initialize_partition_matrix(n_samples, n_clusters, init):
    initial_partition_matrix = np.empty((n_samples, n_clusters))
    if init is None:
        for i in range(n_samples):
            for j in range(n_clusters):
                initial_partition_matrix[i][j] = random.uniform(0, 1)
        for row in range(n_samples):
            sum_row = np.sum(initial_partition_matrix[row])
            for column in range(n_clusters):
                initial_partition_matrix[row][column] = initial_partition_matrix[row][column] / sum_row
    return initial_partition_matrix


def compute_centroids(partition_matrix, data, fuzzy_value):
    samples = np.shape(partition_matrix)[0]
    cluster = np.shape(partition_matrix)[1]
    features = np.shape(data)[1]
    centroids = np.empty(shape=(cluster, features), dtype=float)
    for c in range(cluster):
        numerator_matrix = np.empty(shape=(samples, features), dtype=float)
        denominator = 0.0
        for i in (range(samples)):
            partition_object = pow(partition_matrix[i][c], fuzzy_value)
            numerator_matrix[i] = np.array(partition_object * data[i:i + 1, :])[0]
            denominator += partition_object
        numerator = np.sum(numerator_matrix, axis=0)
        centroids[c] = np.divide(numerator, denominator)
    return centroids


def update_partition_matrix(centroids, data, dist, b, F, a):
    clusters = np.shape(centroids)[0]
    samples = np.shape(data)[0]
    partition_matrix = np.empty(shape=(samples, clusters), dtype=float)
    for k in range(clusters):
        for i in range(samples):
            outer_fraction = 1 / (1 + a)
            summation_innest_element = 0.0
            numerator = 0.0
            denominator = 0.0
            for l in range(clusters):
                summation_innest_element += F[i][l]
                ratio_distance = 0.0
                if dist == "euclidean":
                    ratio_distance = (distance.euclidean(data[i:i + 1, :], centroids[k])) / (
                        distance.euclidean(centroids[l], data[i:i + 1, :]))
                denominator += pow(ratio_distance, 2)
            numerator += (1 + a * (1 - b[i] * summation_innest_element))
            partition_matrix[i][k] = outer_fraction * (numerator / denominator + (a * F[i][k] * b[i]))
    # # ---- Normalize U
    # for row in range(samples):
    #     sum_row = np.sum(partition_matrix[row])
    #     for column in range(clusters):
    #         partition_matrix[row][column] = partition_matrix[row][column] / sum_row
    # # ----
    return partition_matrix


def evaluate_objective_functions(data, centroid, partition_matrix, fuzzy_value, dist, a, b, F):
    J = 0.0
    samples = np.shape(data)[0]
    cardinality_clusters = np.shape(centroid)[0]
    for k in range(cardinality_clusters):
        for i in range(samples):
            if dist == "euclidean":
                J += (pow(distance.euclidean(data[i:i + 1, :], centroid[k]), 2) * (
                    pow(partition_matrix[i][k], fuzzy_value)))
    for k in range(cardinality_clusters):
        for i in range(samples):
            if dist == "euclidean":
                inner_term = partition_matrix[i][k] - (F[i][k] * b[k])
                J += a * (pow(distance.euclidean(data[i:i + 1, :], centroid[k]), 2) * (
                    pow(inner_term, fuzzy_value)))
    return J


def ssfcm(X, number_of_clusters, fuzziness_coefficient, b, F, alpha, max_iter=100,
          stop_condition=('obj_delta', 0.001),
          init=None, distance='euclidean'):
    start = time.time()
    cardinality_samples = np.shape(X)[0]
    number_features = np.shape(X)[1]
    v = initialize_centroids(n_features=number_features, n_clusters=number_of_clusters, init=init)
    U = initialize_partition_matrix(n_samples=cardinality_samples, n_clusters=number_of_clusters, init=init)
    obj_functions = []
    cont = 0
    while (cont < max_iter and (len(obj_functions) <= 2 or np.abs(obj_functions[-1] - obj_functions[-2])) >=
           stop_condition[1]):
        v = compute_centroids(partition_matrix=U, data=X, fuzzy_value=fuzziness_coefficient)
        U = update_partition_matrix(centroids=v, data=X, dist=distance, b=b, F=F,
                                    a=alpha)
        obj_f_value = evaluate_objective_functions(data=X, centroid=v, partition_matrix=U,
                                                   fuzzy_value=fuzziness_coefficient, dist=distance, a=alpha, b=b,
                                                   F=F)
        obj_functions.append(obj_f_value)
        cont += 1
    end = time.time()
    eta = np.round(end - start, decimals=5)
    print("Time elapsed: {} [sec]".format(eta))
    # print("#Iterations:\t", cont - 1, "\t\tObj Func value:\t", obj_functions[-1] - obj_functions[-2])
    # -------------------- DEBUGGING LOGS --------------------
    # print("Check sum of rows of partition matrix:\n")
    # for row in range(cardinality_samples):
    #     sum_row = np.sum(U[row])
    #     print("Sample #", str(row), "\t\t", str(sum_row))
    # print("----------------------------------------")
    # --------------------------------------------------------
    return U, v, eta
