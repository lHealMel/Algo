import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import math


def idx_noise(df):
    cr_df = df.copy()
    threshold = 3
    cr_df[['col1', 'col2']] = StandardScaler().fit_transform(cr_df[['col1', 'col2']])
    # print(cr_df.max(), cr_df.mean(), cr_df.min())
    cr_df = cr_df.loc[(cr_df['col1'] > threshold) | (cr_df['col1'] < -threshold) | (cr_df['col2'] > threshold) | (
                cr_df['col2'] < -threshold)]
    idx = np.array(cr_df.index)

    # plt.scatter(cr_df['col1'], cr_df['col2'], s=50, c='black')
    # plt.show()

    return idx

def noise_centroid(kmeans, df):
    center = kmeans.cluster_centers_
    label = kmeans.labels_.astype(float)
    print(center, len(center))
    print(label, len(label))
    threshold = .5
    k = 0
    noise_idx  = []
    for i in range(len(center)):
        if i == label[k]:
            if math.sqrt((center[i][0]-df.iloc[i][0])**2 + (center[i][1]-df.iloc[i][1])**2) > threshold:
                noise_idx.append(df[i].index)





if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.read_csv('data/mouse.csv', names=['col1', 'col2'])
    plt.scatter(df['col1'], df['col2'], s=50, c='red', alpha=0.5)
    index = idx_noise(df)
    noise = df.iloc[index]



    plt.show()



    cluster = [2, 3, 4, 5, 6]
    iteration = [50, 100, 200, 300]

    kmeans = KMeans(n_clusters=3, max_iter=100).fit(df)
    noise_centroid(kmeans, df)

    # for i in cluster:
    #     for j in iteration:
    #         kmeans = KMeans(n_clusters=i, max_iter=j).fit(df)
    #         centroids = kmeans.cluster_centers_
    #         plt.scatter(df['col1'], df['col2'], c=kmeans.labels_.astype(float),
    #                     s=50, alpha=0.5)
    #         plt.scatter(noise['col1'], noise['col2'], s=50, c='black')
    #         plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    #         plt.title('K-means clustering with K = %d and max_iter = %d' % (i, j))
    #         plt.legend()
    #         plt.show()
