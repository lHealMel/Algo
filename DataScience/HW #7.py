import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def bmi_sh(df):
    bmi_val = np.array([0., 1.0, 2.0, 3.0, 4.0])
    fig, ax = plt.subplots(2, 5, figsize=(12, 10))

    for i, bmi in enumerate(bmi_val):
        subset = df[df["BMI"] == bmi]
        ax[0][i].hist(subset["Height (Inches)"], bins=10)
        ax[0][i].set_title(f"Height (BMI={bmi})")
        ax[0][i].set_xlabel("Height (Inches)")
        ax[0][i].set_ylabel("Frequency")

    for i, bmi in enumerate(bmi_val):
        subset = df[df["BMI"] == bmi]
        ax[1][i].hist(subset["Weight (Pounds)"], bins=10)
        ax[1][i].set_title(f"Weight (BMI={bmi})")
        ax[1][i].set_xlabel("Weight (Pounds)")
        ax[1][i].set_ylabel("Frequency")

    plt.tight_layout()
    # weight & height histogram per bmi : [0., 1., 2., 3., 4.]
    plt.show()

    stdsc = StandardScaler()
    mnsc = MinMaxScaler()
    rbsc = RobustScaler()
    df_cp = df[['Height (Inches)', 'Weight (Pounds)']].copy()
    df_std = stdsc.fit_transform(df_cp)
    df_min = mnsc.fit_transform(df_cp)
    df_rb = rbsc.fit_transform(df_cp)

    fit, ax1 = plt.subplots(3, figsize=(12, 10))

    ax1[0].scatter(df_std[:, 0], df_std[:, 1])
    ax1[0].set_title("Standard scaling")

    ax1[1].scatter(df_min[:, 0], df_min[:, 1])
    ax1[1].set_title("Min-max scaling")

    ax1[2].scatter(df_rb[:, 0], df_rb[:, 1])
    ax1[2].set_title("Robust scaling")

    plt.tight_layout()
    plt.show()  # plot scaled scatter per scaling methods


# Data exploration
def exploration(df):
    print(df.describe())  # statistical data
    print(df.columns)  # feature names
    print(df.dtypes)  # data types, == df.info()
    bmi_sh(df)  # plot histograms, scaling results


def main():
    # Data assumed no missing or wrong values
    df = pd.read_excel("data/bmi_data_phw3.xlsx")

    pd.set_option('display.max_rows', None)

    exploration(df)


if __name__ == '__main__':
    main()
