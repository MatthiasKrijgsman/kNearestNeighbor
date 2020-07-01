import numpy as np
import math

def classify(row, features, labels, k=None):

    """
    Returns the predicted label of a feature vector given a set of features and labels.

    Parameters
    ----------
    row : array_like
        Row vector of input features. Must only contain numerical values.
    features : array_like
        An array of column vectors describing the model features. Must only contain numerical values.
    labels : numpy.ndarray
        A column vector describing the model labels. Can contain any type.
    k : int, optional
        The number of neighbors the model will use to make a prediction. If value is left empty, the
        default recommended k is used. k == math.floor(math.sqrt(n)), where n is the number of entries
        in labels, and k is odd.

    Returns
    -------
    greatest_label_count : any type
        The model's predicted label

    """

    # Set recommended k if none is given
    if k is None:

        k = math.floor(math.sqrt(len(labels)))

        if k % 2 is 0 and k > 1:
            k -= 1

    # Calculate distances to row features
    distances = np.full(labels.shape, 0)

    for i in range(len(features)):
        distances += ( features[i] - row[i] )**2

    distances = np.sqrt(distances)

    # Sort distances and search most common neighboring label
    distances_sorted_indices = np.argsort(distances)

    distances_labels = labels[distances_sorted_indices]

    labels_by_count = np.unique(distances_labels[0:k], return_counts=True)

    greatest_label_count = labels_by_count[0][np.argmax(labels_by_count[1])]

    # Return result
    return greatest_label_count