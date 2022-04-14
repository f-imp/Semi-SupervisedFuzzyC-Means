import numpy as np


def get_b_F(y, n_clusters):
    samples = np.shape(y)[0]
    # create a vector of shape (samples,) with 1 where there is a labelled sample and 0 otherwise (unlabelled sample)
    b = np.where(y != -1, 1, 0)
    # create an F matrix with one more column to handle unlabelled samples (it will be removed in the final step)
    F = np.zeros(shape=(samples, n_clusters + 1), dtype=float)
    # create an index assuming that F is a flatten matrix (horizontally stacked)
    support_for_index = np.reshape(np.arange(start=0, stop=(samples * (n_clusters + 1)), step=n_clusters + 1),
                                   newshape=(samples,))
    # add 1 to y to handle the unlabelled samples (-1)
    # add the support in order to handle the different rows and to make the output (y_indexes_flattened)
    # compliant with the np.put operation
    y_indexes_flattened = y + 1 + support_for_index
    # put value 1.0 in the indexes provided through by the array 'y_indexes_flattened'
    np.put(a=F, ind=y_indexes_flattened, v=1.0)
    # delete the first column that is not useful
    F = np.delete(arr=F, obj=0, axis=1)
    return b, F


def evaluation_alpha(percentage, cardinality_dataset, n_labelled):
    if percentage == 0:
        # alpha = np.shape(dataset)[0] / (cont_labelled + 1)
        alpha = 0.0
    else:
        alpha = cardinality_dataset / n_labelled
    return alpha


def create_dataset_partially_labelled_new(X, y, percentage, random_seed):
    np.random.seed(seed=random_seed)
    new_X, new_y = [], []
    cont_labelled = 0
    cont_unlabelled = 0
    for i in range(np.shape(X)[0]):
        if percentage == 100:
            item = X[i:i + 1, :][0]
            new_X.append(item)
            new_y.append(y[i:i + 1])
            cont_labelled += 1
        else:
            r = np.random.binomial(n=1, p=percentage / 100, size=1, )
            if r > (percentage / 100):
                item = X[i:i + 1, :][0]
                new_X.append(item)
                new_y.append(y[i:i + 1])
                cont_labelled += 1
            else:
                item = X[i:i + 1, :][0]
                new_X.append(item)
                new_y.append(-1)
                cont_unlabelled += 1

    alpha = 0.0
    alpha = evaluation_alpha(percentage, np.shape(X)[0], cont_labelled)
    print("Labelled Samples: #{} \t\t Unlabelled Samples: #{}\n Percentage: {}% \t\t Alpha: {}".format(cont_labelled,
                                                                                                       cont_unlabelled,
                                                                                                       percentage,
                                                                                                       alpha))

    return np.array(new_X).astype('float32'), np.array(new_y).astype('int'), alpha, cont_labelled
