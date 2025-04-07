import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, minmax_scale, MinMaxScaler
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA


def std_scaler(x):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    pd_x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    return pd_x_scaled


def minmax_scaler(x):
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    pd_x_scaled = pd.DataFrame(x_scaled, columns=x.columns)
    return pd_x_scaled


def pca_95(x):
    pca = PCA(.95)
    pca_x = pca.fit_transform(x)
    return pca_x


def pca_2d(x):
    pca = PCA(n_components=2)
    pca_x = pca.fit_transform(x)
    return pca_x


def sh_pca(df):
    y_seg = pd.qcut(df['target'], q=9, labels=False)

    plt.figure(figsize=(20, 20))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=y_seg, cmap='tab10', s=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title('2 component PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.show()


def main():
    # pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)

    dataset = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # target: median house value of california district
    df['target'] = dataset.target

    x, y = df.iloc[:, 0:8], df.iloc[:, -1]  # Similar to x, y = dataset.data, dataset.target

    # std scaler
    x_std_scaled = std_scaler(x)
    print("\nStd Scaled X\n", x_std_scaled)

    # minmax scaler
    x_minmax_scaled = minmax_scaler(x)
    print("\nMinMax Scaled X\n", x_minmax_scaled)

    """
    With std scaling
    """
    pca_95_x = pca_95(x_std_scaled)
    print("\n.95 pca\n", pca_95_x, '\n')  # 8 columns -> 6 columns

    pca_2d_x = pca_2d(x_std_scaled)
    print("\n2d pca\n", pca_2d_x, '\n')  # 8 columns -> 2 columns

    pd_pca_x = pd.DataFrame(pca_2d_x, columns=['principal component 1', 'principal component 2'])
    df = pd.concat([pd_pca_x, y], axis=1)
    print(df)

    """
    With minmax scaling
    """
    # pca_95_x = _95pca(x_minmax_scaled)
    # print("\n.95 pca\n",pca_95_x, '\n')  # 8 columns -> 6 columns
    #
    # pca_2d_x = _2d_pca(x_minmax_scaled)
    # print("\n2d pca\n",pca_2d_x, '\n') # 8 columns -> 2 columns
    #
    # pd_pca_x = pd.DataFrame(pca_2d_x, columns=['principal component 1', 'principal component 2'])
    # df = pd.concat([pd_pca_x, y], axis=1)
    # print(df)

    sh_pca(df)


if __name__ == '__main__':
    main()
