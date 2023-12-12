"""Baseline classification system.

Solution outline for the COM2004/3004 assignment.

This solution will run but the dimensionality reduction and
the classifier are not doing anything useful, so it will
produce a very poor result.

version: v1.0
"""
from typing import List

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter



N_DIMENSIONS = 10

def reclassify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray,label):
    correct_label = label
    k = 3
    #reclassifying while the label is still misclassified
    #the loop will terminate as soon as the class is labelled as something other than the wrong label
    while correct_label == label:
        for test_vector in test:
            distances = np.linalg.norm(train - test_vector, axis=1)
            nearest_k = np.argsort(distances)[:k]
            nearest_labels = train_labels[nearest_k]
        label_counts = {}
        for l in nearest_labels:
            if l not in label_counts and l != label:
                label_counts[l] = nearest_labels.tolist().count(l)
        print(label_counts)
        closest_label = max(label_counts, key=label_counts.get)
        # label_counts_without_max = label_counts.copy()
        # label_counts_without_max.pop(closest_label)
        # closest_label = max(label_counts_without_max, key=label_counts_without_max.get) #giving me an error because the nearest k classes are same as wrong label and removing that gives an empty dictionary
        # # Find the second maximum label from the modified label_counts
        correct_label = closest_label
        k+=2 #increasing k by 2 to use more distances in the computation in the hopes of getting another label
    return correct_label

def classify(train: np.ndarray, train_labels: np.ndarray, test: np.ndarray) -> List[str]:
    """Classify a set of feature vectors using a training set.

    This dummy implementation simply returns the empty square label ('.')
    for every input feature vector in the test data.

    Note, this produces a surprisingly high score because most squares are empty.

    Args:
        train (np.ndarray): 2-D array storing the training feature vectors.
        train_labels (np.ndarray): 1-D array storing the training labels.
        test (np.ndarray): 2-D array storing the test feature vectors.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    output_labels = []
    for test_vector in test:
        distances = np.linalg.norm(train - test_vector, axis=1)
        nearest_k = np.argsort(distances)[:3]
        nearest_labels = train_labels[nearest_k]
        label_counts = {}
        for label in nearest_labels:
            if label not in label_counts:
                label_counts[label] = nearest_labels.tolist().count(label)
        closest_label = max(label_counts, key=label_counts.get)
        output_labels.append(closest_label)

    return output_labels

# The functions below must all be provided in your solution. Think of them
# as an API that it used by the train.py and evaluate.py programs.
# If you don't provide them, then the train.py and evaluate.py programs will not run.
#
# The contents of these functions are up to you but their signatures (i.e., their names,
# list of parameters and return types) must not be changed. The trivial implementations
# below are provided as examples and will produce a result, but the score will be low.


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The feature vectors are stored in the rows of 2-D array data, (i.e., a data matrix).
    The dummy implementation below simply returns the first N_DIMENSIONS columns.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    #take the data from the model

    mean_vector = np.mean(data, axis=0)
    centered_data = data - mean_vector

    projected_data = np.dot(centered_data, model["eigenvectors"])
    
    reduced_data = projected_data[:, :N_DIMENSIONS]
    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Note, the contents of the dictionary are up to you, and it can contain any serializable
    data types stored under any keys. This dictionary will be passed to the classifier.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    # The design of this is entirely up to you.
    # Note, if you are using an instance based approach, e.g. a nearest neighbour,
    # then the model will need to store the dimensionally-reduced training data and labels.
    model = {}
    model["labels_train"] = labels_train.tolist()
    cov_matrix = np.cov(fvectors_train, rowvar=False)
    N = cov_matrix.shape[0]
    eigenvalues, eigenvectors = scipy.linalg.eigh(cov_matrix, eigvals=(N - N_DIMENSIONS, N - 1))
    eigenvectors = np.fliplr(eigenvectors)
    model["eigenvectors"] = eigenvectors.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)

    model["fvectors_train"] = fvectors_train_reduced.tolist()

    
    return model


def images_to_feature_vectors(images: List[np.ndarray]) -> np.ndarray:
    """Takes a list of images (of squares) and returns a 2-D feature vector array.

    In the feature vector array, each row corresponds to an image in the input list.

    Args:
        images (list[np.ndarray]): A list of input images to convert to feature vectors.

    Returns:
        np.ndarray: An 2-D array in which the rows represent feature vectors.
    """
    h, w = images[0].shape
    n_features = h * w
    fvectors = np.empty((len(images), n_features))
    for i, image in enumerate(images):
        image = gaussian_filter(image, sigma=1)
        fvectors[i, :] = image.reshape(1, n_features)

    return fvectors


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in an arbitrary order.

    Note, the feature vectors stored in the rows of fvectors_test represent squares
    to be classified. The ordering of the feature vectors is arbitrary, i.e., no information
    about the position of the squares within the board is available.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """

    # Get some data out of the model. It's up to you what you've stored in here
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # Call the classify function.
    labels = classify(fvectors_train, labels_train, fvectors_test)

    return labels


def classify_boards(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Run classifier on a array of image feature vectors presented in 'board order'.

    The feature vectors for each square are guaranteed to be in 'board order', i.e.
    you can infer the position on the board from the position of the feature vector
    in the feature vector array.

    In the dummy code below, we just re-use the simple classify_squares function,
    i.e. we ignore the ordering.

    Args:
        fvectors_test (np.ndarray): An array in which feature vectors are stored as rows.
        model (dict): A dictionary storing the model data.

    Returns:
        list[str]: A list of one-character strings representing the labels for each square.
    """
    labels_list = classify_squares(fvectors_test, model)
    model_list = []
    fvectors_train = np.array(model["fvectors_train"])
    labels_train = np.array(model["labels_train"])

    # print(labels_list)

    for i in range(0,len(labels_list),64):

        #first separate out the boards into individual boards
        model_list.append(labels_list[i:i + 64])

        #go through each board in model_list and implement these:

    # print(model_list)
    for board in range(len(model_list)):
        k_count = 0
        q_count = 0
    #1) if there's pawns in the back or front rows, impossible since pawns can be anywhere but last and first row
        for piece in range(8):
            # print(model_list[board][piece])

            if model_list[board][piece] == "p" or model_list[board][piece] == "P":
                model_list[board][piece] = reclassify(fvectors_train, labels_train, fvectors_test,model_list[board][piece])    
                # print(model_list[board][piece])
            
        for piece in range(-1,-9,-1):
            if model_list[board][piece] == "p" or model_list[board][piece] == "P":
                model_list[board][piece] = reclassify(fvectors_train, labels_train, fvectors_test,model_list[board][piece])

    #2) if there's 2 kings, one of them is definitely a queen - same color as king
        for piece in range(len(model_list[board])):
            if model_list[board][piece] == "k" or model_list[board][piece] == "K":
                k_count += 1
                if k_count == 2:
                    if model_list[board][piece] == "k":
                        model_list[board][piece] = "q"
                    else:
                        model_list[board][piece] = "Q"

            if model_list[board][piece] == "q" or model_list[board][piece] == "Q":
                q_count += 1
                if q_count == 2:
                    if model_list[board][piece] == "q":
                        model_list[board][piece] = "k"
                    else:
                        model_list[board][piece] = "K"

    # print(model_list)


    return model_list
