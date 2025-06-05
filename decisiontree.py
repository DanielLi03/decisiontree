import numpy as np
import pandas as pd
import math

# below we implement a binary classification decision tree with (pre cleaned) discrete parameters

# assume that you have an implementation of tree right now
DATA = pd.read_csv('cleandata.csv', header = None)
split = np.random.rand(len(DATA)) < 0.8
TRAINDATA = DATA[split]
TESTDATA = DATA[~split]

#calculates the entropy of the outcome varaible (aka label)
def entropy(data):
    labels = data.iloc[:,-1:]
    total = len(labels)
    positives = labels.sum()
    negatives = total - labels.sum()
    positive_entropy = 0
    negative_entropy = 0
    if float(positives) > 0:
        positive_entropy = (positives / total) * math.log((positives / total) , 2)
    if float(negatives) > 0:
        negative_entropy = (negatives / total) * math.log((negatives / total) , 2)
    return -1 * float(positive_entropy + negative_entropy)


# mutual information function
# takes in a dataset and a column index, and calculates the mutual info of that feature column with the labels column
def mutual_information(data, col):
    global_entropy = entropy(data)
    conditional_entropies = []
    for i in data[col].unique():
        probability = len(data[data[col] == i]) / len(data)
        conditional_entropy = entropy(data[data[col] == i])
        conditional_entropies.append(probability * conditional_entropy)
    
    return global_entropy - sum(conditional_entropies)



# the train function trains the decision tree based on mutual information metric
# inputs: pandas Dataframe and max depth for decision tree
# outputs: decision tree
def train(data, depth):
    # assume that the last column of the dataset are the labels
    # check that the depth is less than or equal to the number of features
    assert len(data.columns) >= depth, "depth cannot exceed number of features"
    assert depth >= 0, "depth cannot be negative"

    # case where depth == 0
    infos = []
    if depth == 0:
        return int(TRAINDATA.iloc[:,-1:].mode().iloc[0])

    for i in range(len(data.columns) - 1):
        infos.append((mutual_information(data, i), i))
    infos.sort(key = lambda x: x[0])
    return infos[:(depth)]

# predict function gives a prediction based on the input parameters and given decision tree
# inputs: input parameters + decision tree
# output: prediction value
def predict(tree, x):
    if len(tree) ==  0:
        return int(TRAINDATA.iloc[:,-1:].mode().iloc[0])

    print(list(x))
    pointer = 0
    data = TRAINDATA
    output = int(TRAINDATA.iloc[:,-1:].mode().iloc[0])
    while (len(data) > 0) and (pointer < len(x)):
        output = int(data.iloc[:,-1:].mode().iloc[0])
        data = data[data[tree[pointer][1]] == x[tree[pointer - 1][1]]]

    return output

# the test function tests the decision tree
# inputs: decision tree, metric to score decision tree
# outputs: score
def test(data, tree):
    results = data.iloc[:,-1:]
    predictions = data.apply(lambda x: predict(tree,x), axis = 1)
    return (len(results) - ((abs(results - predictions)).sum())) / len(results)

print(test(TESTDATA, train(TRAINDATA, 1)))