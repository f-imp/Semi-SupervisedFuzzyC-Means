import cupy as cp
import time
import numpy as np


def initialize_centroids_new_cp(n_features, n_clusters, init):
    initial_centroids = cp.empty((n_clusters, n_features))
    if init is None:
        initial_centroids.fill(0.0)
    return initial_centroids


def initialize_partition_matrix_new_cp(n_samples, n_clusters, init):
    initial_partition_matrix = cp.empty((n_samples, n_clusters))
    if init is None:
        initial_partition_matrix = cp.random.uniform(low=0.0, high=1.0, size=(n_samples, n_clusters))
        sum_row = cp.sum(a=initial_partition_matrix, axis=1)
        divisor_matrix = cp.transpose(cp.tile(sum_row, reps=(n_clusters, 1)))
        initial_partition_matrix = cp.divide(initial_partition_matrix, divisor_matrix)
    return initial_partition_matrix


def compute_centroids_new_cp(partition_matrix, data, fuzzy_value):
    cluster = cp.shape(partition_matrix)[1]
    features = cp.shape(data)[1]
    centroids = cp.empty(shape=(cluster, features), dtype=float)
    partition_matrix_of_objects = cp.power(partition_matrix, fuzzy_value)
    for c in range(cluster):
        numerator_matrix = partition_matrix_of_objects[:, c:c + 1] * data[:, :]
        denominator = cp.sum(partition_matrix_of_objects[:, c:c + 1], axis=0)
        numerator = cp.sum(numerator_matrix, axis=0)
        centroids[c] = cp.divide(numerator, denominator)
    return centroids


def update_partition_matrix_new_cp(centroids, data, dist, b, F, a):
    clusters = cp.shape(centroids)[0]
    samples = cp.shape(data)[0]
    outer_fraction = 1 / (1 + a)

    """
      Compute the distance matrix DM of shape=(samples, clusters) in which 
      DM_ij is the euclidean distance among sample_i and centroid_j

      Let's assume to have 3 features:
        sample_i   = [s_11, s_12, s_13]
        centroid_j = [c_11, c_12, c_13]
      The euclidean distance is computed as follows: 
        distance(sample_i, centroid_j) = square_root((s11-c11)^2 + (s12-c12)^2 + (s13-c13)^2)

    """

    distance_matrix = cp.zeros(shape=(samples, clusters), dtype=float)

    if dist == "euclidean":
        centroids_flatten = centroids.flatten()
        centroids_flatten_tile = cp.tile(centroids_flatten, reps=(samples, 1))
        data_tile = cp.tile(data, reps=(1, clusters))
        subtraction = data_tile - centroids_flatten_tile
        subtraction_pow = cp.power(subtraction, 2)
        # Split the subtraction_pow matrix into (#clusters)-array, creating a tensor
        subtraction_pow_tensor = cp.array(cp.hsplit(ary=subtraction_pow, indices_or_sections=clusters))
        # Sum the rows of each array in the tensor to get the sum (s11-c11)^2 + (s12-c12)^2 + (s13-c13)^2
        sum_row = cp.sum(a=subtraction_pow_tensor, axis=2)
        sum_row_transpose = cp.transpose(a=sum_row)
        # Take the square-root to get in each cell the value of distance(•,•)
        distance_matrix = cp.sqrt(sum_row_transpose)

    """
      Now we need to compute the ratio_distance_matrix of shape=(sample, clusters)
      RDM_ij = distance(sample_i, centroid_j) / (Sigma|k from 1 to clusters : distance(sample_i, centroid_k))

      Hence we need to repeat every column for #(clusters) times,
      then reshape this vector to be of shape(sample, clusters*clusters),
      then create a tensor of shape=(clusters, sample, clusters),
      divide each multidimensional array by the original distance_matrix (element-wise division),
      take the power of 2 of all values, 
      sum each row of every multidimensional array and in the end get the transpose to have
      a multidimensional array of shape=(sample, clusters) in which 
      the cell (i,j) is the sum, of the power of 2, of distance(sample_i, cluster_j)/(Sigma|k from 1 to clusters : distance(sample_i, centroid_k))

    """

    distance_matrix_repeated = cp.repeat(a=distance_matrix, repeats=clusters)
    distance_matrix_repeated_right_shape = cp.reshape(a=distance_matrix_repeated,
                                                      newshape=(samples, clusters * clusters))
    distance_matrix_tensor = cp.array(cp.hsplit(ary=distance_matrix_repeated_right_shape, indices_or_sections=clusters))
    ratio_distance_matrix = cp.divide(distance_matrix_tensor, distance_matrix)
    ratio_distance_matrix_pow = cp.power(ratio_distance_matrix, 2)
    denominator = cp.sum(a=ratio_distance_matrix_pow, axis=2)
    denominator_matrix = cp.transpose(a=denominator)

    """
      Now we need to compute the numerator_matrix (shape=(samples, clusters)).
      The numerator value is computed, in a for-oriented algorithm as follows:

        numerator =  (1 + a * (1 - b[i] * summation_innest_element)), 
                      where: a                       : is a constant value (alpha)
                             i                       : is the sample index,
                             b[i]                    : is the labelled-flag (1 if the sample_i is assumed to be labelled, 0 otherwise)
                             summation_innest_element: is the sum of all membership values F for the sample i over all the clusters

      While, the equivalent solution, in a matrix-oriented way can be:

        numerator_matrix = 1_array + a * (1_array - b * summation_innest_vector)
                           where: 1_array                : shape=(samples, 1)
                                  a                      : scalar (float)
                                  1_array                : shape=(samples, 1)
                                  b                      : shape=(samples, 1)
                                  summation_innest_vector: shape=(samples, 1)

      And then repeat this vector column #(clusters) times
    """

    summation_innest_vector = cp.sum(F, axis=1)
    numerator_vector = cp.ones(shape=(samples,)) + a * (cp.ones(shape=(samples,)) - b * summation_innest_vector)
    numerator_vector = cp.reshape(a=numerator_vector, newshape=(numerator_vector.shape[0], 1))
    numerator_matrix = cp.tile(numerator_vector, reps=(1, clusters))

    """
      Now we need to compute the partition_matrix (shape=(samples, clusters))

      The partition_matrix in a for-oriented algorithm is computed as follows:

        partition_matrix[i][k] = outer_fraction * (numerator / denominator + (a * F[i][k] * b[i]))
                                 where: outer_fraction: is a constant value (float)
                                        numerator     : is the i-th cell of numerator_matrix
                                        denominator   : is the cell i of the denominator_matrix
                                        a             : is a constant value (called alpha, float)
                                        F[i][k]       : the membership degree of sample_i to cluster_k
                                        b[i]          : is the labelled-flag (1 if the sample_i is assumed to be labelled, 0 otherwise)

      While, the equivalent solution, in a matrix-oriented way can be:

        partition_matrix = outer_fraction * (numerator_matrix_tile/denominator_matrix_tile + (a * F * b_tile))
                           where: outer_fraction         : is a constant value (float)
                                  numerator_matrix_tile  : is the i-th cell of numerator_matrix (shape=(sample, clusters))
                                  denominator_matrix_tile: is the cell i of the denominator_matrix (shape=(sample, clusters))
                                  a                      : is a constant value (called alpha, float)
                                  F                      : is the membership matrix (shape=(sample, clusters))
                                  b_tile                 : is the labelled-flag vector replicated #(clusters)-time (shape=(sample, clusters))

    """
    b_tile = cp.tile(cp.reshape(a=b, newshape=(b.shape[0], 1)), reps=(1, clusters))

    partition_matrix = outer_fraction * (cp.divide(numerator_matrix, denominator_matrix) + (a * F * b_tile))
    return partition_matrix


def evaluate_objective_functions_new_cp(data, centroids, partition_matrix, fuzzy_value, dist, a, b, F):
    J = 0.0
    samples = cp.shape(data)[0]
    cardinality_clusters = cp.shape(centroids)[0]
    distance_matrix = cp.zeros(shape=(samples, cardinality_clusters), dtype=float)

    if dist == "euclidean":
        centroids_flatten = centroids.flatten()
        centroids_flatten_tile = cp.tile(centroids_flatten, reps=(samples, 1))
        data_tile = cp.tile(data, reps=(1, cardinality_clusters))
        subtraction = data_tile - centroids_flatten_tile
        subtraction_pow = cp.power(subtraction, 2)
        # Split the subtraction_pow matrix into (#cardinality_clusters)-array, creating a tensor
        subtraction_pow_tensor = cp.array(cp.hsplit(ary=subtraction_pow, indices_or_sections=cardinality_clusters))
        # Sum the rows of each array in the tensor to get the sum (s11-c11)^2 + (s12-c12)^2 + (s13-c13)^2
        sum_row = cp.sum(a=subtraction_pow_tensor, axis=2)
        sum_row_transpose = cp.transpose(a=sum_row)
        # Take the square-root to get in each cell the value of distance(•,•)
        distance_matrix = cp.sqrt(sum_row_transpose)

    J += cp.sum(cp.divide(cp.power(distance_matrix, 2), cp.power(partition_matrix, 2)))

    b_tile = cp.tile(cp.reshape(a=b, newshape=(b.shape[0], 1)), reps=(1, cardinality_clusters))
    J += cp.sum(a * distance_matrix * (partition_matrix - F * b_tile))

    return J


def ssfcm_new_cp(X, number_of_clusters, fuzziness_coefficient, b, F, alpha, max_iter=100,
                 stop_condition=('obj_delta', 0.001),
                 init=None, distance='euclidean'):
    start = time.time()
    cardinality_samples = cp.shape(X)[0]
    number_features = cp.shape(X)[1]
    v = initialize_centroids_new_cp(n_features=number_features, n_clusters=number_of_clusters, init=init)
    U = initialize_partition_matrix_new_cp(n_samples=cardinality_samples, n_clusters=number_of_clusters, init=init)
    obj_functions = []
    cont = 0
    while (cont < max_iter and (len(obj_functions) <= 2 or cp.abs(obj_functions[-1] - obj_functions[-2])) >=
           stop_condition[1]):
        v = compute_centroids_new_cp(partition_matrix=U, data=X, fuzzy_value=fuzziness_coefficient)
        U = update_partition_matrix_new_cp(centroids=v, data=X, dist=distance, b=b, F=F,
                                           a=alpha)
        obj_f_value = evaluate_objective_functions_new_cp(data=X, centroids=v, partition_matrix=U,
                                                          fuzzy_value=fuzziness_coefficient, dist=distance, a=alpha,
                                                          b=b,
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
    #     sum_row = cp.sum(U[row])
    #     print("Sample #", str(row), "\t\t", str(sum_row))
    # print("----------------------------------------")
    # --------------------------------------------------------
    return U, v, eta
