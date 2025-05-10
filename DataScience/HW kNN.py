import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


# calculate euclidean distance
def euclidean(df, new_data):
    return np.sqrt((df.iloc[:, 0] - new_data[0]) ** 2 + (df.iloc[:, 1] - new_data[1]) ** 2)


def knn(df, k, z=0):
    if z == 1:
        nn = df.sort_values("Z_Distance").head(k)
    elif z == 2:
        nn = df.sort_values("R_Distance").head(k)
    else:
        nn = df.sort_values("Distance").head(k)

    predict = nn["T SHIRT SIZE"].mode()[0]
    return predict, nn


def z_score(column):
    return (column - column.mean()) / column.std()


def z_score_predict(df, z_new_data):
    # add columns that normalized(z_score) value, and with that distance
    df["Z_HEIGHT"] = z_score(df["HEIGHT(cm)"])
    df["Z_WEIGHT"] = z_score(df["WEIGHT(kg)"])

    df["Z_Distance"] = euclidean(df[["Z_HEIGHT", "Z_WEIGHT"]], z_new_data)

    print("----Z-Score Scaled prediction----")
    # predict when k = 3
    pred_3, neighbors_3 = knn(df, 3, z=1)
    print("Nearest neighbors (k=3):")
    print(neighbors_3)
    print("k=3 Prediction:", pred_3, '\n')

    # predict when k = 5
    pred_5, neighbors_5 = knn(df, 5, z=1)
    print("Nearest neighbors (k=5):")
    print(neighbors_5)
    print("k=5 Prediction:", pred_5, '\n')


def predict(df, new_data):
    df["Distance"] = euclidean(df[["HEIGHT(cm)", "WEIGHT(kg)"]], new_data)

    print("----Non-Scaled prediction----")
    # predict when k = 3
    pred_3, neighbors_3 = knn(df, 3, z=0)
    print("Nearest neighbors (k=3):")
    print(neighbors_3)
    print("k=3 Prediction:", pred_3, '\n')

    # predict when k = 5
    pred_5, neighbors_5 = knn(df, 5, z=0)
    print("Nearest neighbors (k=5):")
    print(neighbors_5)
    print("k=5 Prediction:", pred_5, '\n')


if __name__ == "__main__":
    # set the dataset, predict data
    data = {
        "HEIGHT(cm)": [158, 158, 158, 160, 160, 160, 163, 163, 160, 163, 165, 165, 165, 168, 168, 168, 170, 170, 170],
        "WEIGHT(kg)": [58, 59, 63, 59, 60, 60, 60, 61, 64, 64, 61, 62, 65, 62, 63, 66, 63, 64, 68],
        "T SHIRT SIZE": ['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L']
    }
    new_data = np.array([161, 61])
    z_new_data = z_score(new_data)

    df = pd.DataFrame(data)

    z_score_predict(df.copy(), z_new_data)
    predict(df.copy(), new_data)
