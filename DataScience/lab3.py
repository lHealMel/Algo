# 202135835 정지호
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder, OneHotEncoder, KBinsDiscretizer


def kBin():
    kbin = KBinsDiscretizer()
    score = np.linspace(0, 10, 1).reshape(-1, 1)
    score_fit = kbin.fit_transform(score)
    print(score_fit)


def ordinal_encode():
    ordE = OrdinalEncoder()
    x = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox'],
         ['male', 'from Asia', 'uses Chrome']]
    x_en = ordE.fit_transform(x)
    print(x_en)


def onehot_encode():
    oneE = OneHotEncoder()
    x = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox'],
         ['male', 'from Asia', 'uses Chrome']]
    x_one = oneE.fit_transform(x)
    print(x_one.toarray())


def polynomial_reg():
    x = np.arange(1, 11)
    y = np.array([20.6, 30.8, 55.0, 71.4, 97.3, 131.8, 156.3, 197.3, 238.7, 291.7])
    lr = LinearRegression(fit_intercept=False)
    pf = PolynomialFeatures(degree=2)
    model = Pipeline([('poly', pf), ('linear', lr)])

    model.fit(x.reshape(-1, 1), y)

    plt.figure(figsize=(10, 10))
    plt.plot(x, model.predict(x.reshape(-1, 1)), color='red')
    plt.scatter(x, y)
    plt.grid(True)
    plt.show()


def linear_reg():
    hu = np.array([46, 53, 29, 61, 36, 39, 47, 49, 52, 38, 55, 32, 57, 54, 44])
    mo = np.array([12, 15, 7, 17, 10, 11, 11, 12, 14, 9, 16, 8, 18, 14, 12])

    # Calculate
    lr = LinearRegression()
    lr.fit(hu.reshape(-1, 1), mo)

    x = np.array([hu.min() - 1, hu.max() + 1])
    y = lr.predict(x.reshape(-1, 1))
    plt.figure(figsize=(12, 8))
    plt.scatter(hu, mo, color='blue')
    plt.plot(x, y)
    plt.grid(True)
    plt.show()

    """
        Just using sns.regplot.
    plt.figure(figsize=(12, 8))
    sns.regplot(x=hu, y=mo, ci=None, color='b', line_kws={'color': 'red'}, scatter_kws={'color': 'blue', 's': 10})
    plt.grid(True)
    plt.show()
    """

def main():
    # ordinal_encode()
    # onehot_encode()
    # kBin()
    print()


if __name__ == '__main__':
    main()
