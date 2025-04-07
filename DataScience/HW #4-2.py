import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import seaborn as sns


def scaling(x):
    # standardization with MinMaxScaler
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
    return x_scaled_df


def sh_selectKBest(x, y):
    plt.figure(figsize=(20, 20))
    plt.scatter(x['MedInc'], y, c=y)
    plt.title("between MedInc and Target", fontsize=12)
    plt.xlabel('MedInd')
    plt.ylabel('target')
    plt.show()


def sh_importance_scoring(x, y):
    model = ExtraTreesClassifier()
    model.fit(x, y)
    print('\nFeature importance\n', model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=x.columns)
    feat_importances.plot(kind='barh')
    plt.show()


def sh_corr(df):
    plt.figure(figsize=(20, 20))
    # plot the heat map
    g = sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")
    plt.show()


def pr_best(x, y, k):
    bestfeatures = SelectKBest(score_func=chi2, k=k)
    fit = bestfeatures.fit(x, y)

    dfcolumns = pd.DataFrame(x.columns)
    dfscores = pd.DataFrame(fit.scores_)

    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Features', 'Score']
    print("\nFeature score\n",
          featureScores.nlargest(k, 'Score'))


def main():
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.width', None)

    dataset = sklearn.datasets.fetch_california_housing()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

    # target: median house value of california district
    df['target'] = dataset.target

    x, y = df.iloc[:, 0:8], df.iloc[:, -1]  # Similar to x, y = dataset.data, dataset.target

    # As the target type is continuous, we may divide the sections.
    y_seg = pd.qcut(y, q=4, labels=False)

    # As Longitude feature is negative, we use min-max scaling.
    # Or, If deemed unnecessary, you can delete the longitude and latitude values.
    x_scaled = scaling(x)

    pr_best(x_scaled, y_seg, 3)
    sh_importance_scoring(x_scaled, y_seg)
    sh_corr(df)


if __name__ == '__main__':
    main()
