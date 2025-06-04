import numpy as np
import pandas as pd

# below we implement a binary classification decision tree with (pre cleaned) discrete parameters

# assume that you have an implementation of tree right now

# mutual information function
# takes in a dataset and a feature column, and calculates the mutual info of that feature column with
def mutual_information(data, col):
    

# the train function trains the decision tree based on mutual information metric
# inputs: pandas Dataframe and max depth for decision tree
# outputs: decision tree
def train(data, depth):
    # assume that the last column of the dataset are the labels
    # check that the depth is less than or equal to the number of features
    assert len(data.column) >= depth, "depth cannot exceed number of features"
    assert depth > 0, "depth cannot be negative"

    # case where depth == 0
    if depth == 0:
        data.iloc[:,-1:].mode()

    if depth == 1:

# the test function tests the decision tree
# inputs: decision tree, metric to score decision tree
# outputs: score

# predict function gives a prediction based on the input parameters and given decision tree
# inputs: input parameters + decision tree
# output: prediction value

