import pandas as pd
import numpy as np

# calculate euclidean distance
def euclidean(df, new_data):
    return np.sqrt((df["HEIGHT(cm)"] - new_data[0])**2 + (df["WEIGHT(kg)"] - new_data[1])**2)

def knn(df, k):
    nn = df.sort_values("Distance").head(k)
    predict= nn["T SHIRT SIZE"].mode()[0]
    return predict, nn


if __name__ == "__main__":
    # set the dataset, predict data
    data = {
        "HEIGHT(cm)": [158, 158, 158, 160, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170],
        "WEIGHT(kg)": [58, 59, 63, 59, 60, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68],
        "T SHIRT SIZE": ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
    }
    new_data = np.array([161, 61])
    df = pd.DataFrame(data)
    df["Distance"] = euclidean(df, new_data)

    # predict when k = 3
    pred_3, neighbors_3 = knn(df, 3)
    print("Nearest neighbors (k=3):")
    print(neighbors_3)
    print("k=3 Prediction:", pred_3, '\n')

    # predict when k = 5
    pred_5, neighbors_5 = knn(df, 5)
    print("Nearest neighbors (k=5):")
    print(neighbors_5)
    print("k=5 Prediction:", pred_5)