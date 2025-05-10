#202135835 정지호
import pandas as pd
import numpy as np
from collections import Counter
import math

# entropy calculate
def entropy(column):
    counts = Counter(column) # save the counts for the columns
    total = len(column)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


# Information Gain calculate
def info_gain(data, split_attribute, target_attribute='Outcome'):

    total_entropy = entropy(data[target_attribute])

    # weighted average entropy
    values = data[split_attribute].unique()
    weighted_entropy = 0
    for val in values:
        subset = data[data[split_attribute] == val]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset[target_attribute])

    # Information Gain
    return total_entropy - weighted_entropy


# tree building
def build_tree(data, features, target_attribute='Outcome'):
    target_values = list(data[target_attribute])

    # if all samples have the same result
    if len(set(target_values)) == 1:
        return target_values[0]

    # if no longer have attributes to divide
    if not features:
        return Counter(target_values).most_common(1)[0][0]

    # select attributes with the greatest information gain
    gains = [info_gain(data, feature, target_attribute) for feature in features]
    best_feature_index = np.argmax(gains)
    best_feature = features[best_feature_index]

    tree = {best_feature: {}}

    # create subtrees
    for value in data[best_feature].unique():
        sub_data = data[data[best_feature] == value]
        subtree = build_tree(sub_data, [f for f in features if f != best_feature], target_attribute)
        tree[best_feature][value] = subtree

    return tree

# prediction : with recursion
def predict(tree, sample):
    if not isinstance(tree, dict):
        return tree  # leaf node

    attribute = next(iter(tree))
    value = sample[attribute]

    if value in tree[attribute]:
        return predict(tree[attribute][value], sample)
    else:
        return "Unknown"  # value that wasn't in the training


if __name__ == '__main__':
    # create the dataset
    data = {
        'District': ['Suburban', 'Suburban', 'Rural', 'Urban', 'Urban', 'Urban', 'Rural', 'Suburban', 'Suburban',
                     'Urban',
                     'Suburban', 'Rural', 'Rural', 'Urban'],
        'House Type': ['Detached', 'Detached', 'Detached', 'Semi-detached', 'Semi-detached', 'Semi-detached',
                       'Semi-detached', 'Terrace', 'Semi-detached', 'Terrace', 'Terrace', 'Terrace', 'Detached',
                       'Terrace'],
        'Income': ['High', 'High', 'High', 'High', 'Low', 'Low', 'Low', 'High', 'Low', 'Low', 'Low', 'High', 'Low',
                   'High'],
        'Previous Customer': ['No', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes'],
        'Outcome': ['Not responded', 'Not responded', 'Responded', 'Responded', 'Responded', 'Not responded',
                    'Responded',
                    'Not responded', 'Responded', 'Responded', 'Responded', 'Responded', 'Responded', 'Not responded']
    }

    df = pd.DataFrame(data)


    # make tree with this attributes
    features = ['District', 'House Type', 'Income', 'Previous Customer']

    # make tree
    decision_tree = build_tree(df, features)

    # predict input
    new_customer = {
        'District': 'Suburban',
        'House Type': 'Detached',
        'Income': 'Low',
        'Previous Customer': 'Yes'
    }

    # predict output
    prediction = predict(decision_tree, new_customer)

    print("Decision Tree:")
    print(decision_tree)
    print("\nPrediction:", prediction)
