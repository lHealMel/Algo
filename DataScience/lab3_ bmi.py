import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def bmi_sh(df):
    bmi_val = np.array([1.0, 2.0, 3.0])
    fig, ax = plt.subplots(2, 3, figsize=(12, 10))

    for i, bmi in enumerate(bmi_val):
        subset = df[df["BMI"] == bmi]
        ax[0][i].hist(subset["Height (Inches)"], bins=10)
        ax[0][i].set_title(f"Height (BMI={int(bmi)})")
        ax[0][i].set_xlabel("Height (Inches)")
        ax[0][i].set_ylabel("Frequency")

    # 몸무게(Weight) 히스토그램 그리기
    for i, bmi in enumerate(bmi_val):
        subset = df[df["BMI"] == bmi]
        ax[1][i].hist(subset["Weight (Pounds)"], bins=10)
        ax[1][i].set_title(f"Weight (BMI={int(bmi)})")
        ax[1][i].set_xlabel("Weight (Pounds)")
        ax[1][i].set_ylabel("Frequency")
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
    ax1[1].scatter(df_min[:, 0], df_min[:, 1])
    ax1[2].scatter(df_rb[:, 0], df_rb[:, 1])
    plt.show()


#  linear regression equation : e
def e(df):
    x = df['Height (Inches)'].sum()
    y = df['Weight (Pounds)'].sum()

    xy = (df['Height (Inches)'] * df['Weight (Pounds)']).sum()

    x_df = pd.DataFrame([x * x for x in df['Height (Inches)']]).sum()
    x_squared = x_df.values

    n = len(df)
    m = ((n * xy) - (x * y)) / ((n * x_squared) - (x * x))
    b = (y - (m * x)) / n

    return m, b


"""
    P.51
    Peek into the dataset (data exploration)
"""


def exploration(df):
    print(df.describe())  # statistical data
    print(df.columns)  # feature names
    print(df.dtypes)  # data types, == df.info()
    bmi_sh(df)  # plot histograms, scaling results


""" 
    P.52
    Missing value manipulation (simple)
"""


def missing_values(df):
    # make all likely-wrong values, missing values to np.nan
    df.loc[(df['Height (Inches)'] > 400) | (df['Height (Inches)'] <= 0), 'Height (Inches)'] = np.nan
    df.loc[(df['Weight (Pounds)'] > 500) | (df['Weight (Pounds)'] <= 0), 'Weight (Pounds)'] = np.nan

    print(df.isna().sum())  # Print sum of rows with NAN, and sum of NAN for each column
    print(df[df.isna().any(axis=1)])
    df_no_nan = df.dropna()  # Extract all rows without NAN, df_no_nan is dropped nan values from df.

    df.iloc[:, 1:5] = df.iloc[:, 1:5].fillna(df.iloc[:, 1:5].median())  # nan values replaced to median value.

    # df : replaced, df_no_nan : dropped
    return df_no_nan


"""
    Missing value manipulation (more elaborate)
    Identify all dirty records with likely-wrong or missing height or weight values (by eye inspection)
    Clean the dirty values using linear regression (see the next page)
     Draw a scatter plot of (height, weight) in the clean dataset emphasizing previously dirty records with a different color
"""


def linear_treat(df):
    # set the values to np.nan which has wrong height values, but has weight values, vice versa.
    df.loc[(df['Height (Inches)'] <= 0) | (df['Height (Inches)'] > 119) & (
        df['Weight (Pounds)'].notna()), 'Height (Inches)'] = np.nan
    df.loc[(df['Weight (Pounds)'] <= 0) | (df['Weight (Pounds)'] > 600) & (
        df['Height (Inches)'].notna()), 'Weight (Pounds)'] = np.nan

    # if weight and height is both nan, drop it
    df = df.dropna(subset=['Height (Inches)', 'Weight (Pounds)'], how='all')


    # calculate with data which has complete values
    m, b = e(df.dropna(how='any'))

    # save the imputed data's index
    height_missing = df['Height (Inches)'].isna()
    weight_missing = df['Weight (Pounds)'].isna()

    # impute the values.
    df.loc[weight_missing, 'Weight (Pounds)'] = df['Height (Inches)'] * m + b
    df.loc[height_missing, 'Height (Inches)'] = (df['Weight (Pounds)'] - b) / m


    # calculate BMI which BMI values is NaN.
    bmi_mask = df['BMI'].isna()
    bins = [0, 16, 18.5, 24.9, 29.9, 45]
    df.loc[bmi_mask, 'BMI'] = np.digitize(
        (df.loc[bmi_mask, 'Weight (Pounds)'] / (df.loc[bmi_mask, 'Height (Inches)'] ** 2)) * 703, bins)

    # set the mask, which means imputed value
    filled_mask = height_missing | weight_missing


    plt.figure(figsize=(10, 10))
    plt.scatter(df.loc[filled_mask, 'Height (Inches)'],df.loc[filled_mask, 'Weight (Pounds)'], c="green", label="Imputed Data")
    plt.scatter(df.loc[~filled_mask, 'Height (Inches)'], df.loc[~filled_mask, 'Weight (Pounds)'], c="red", label="Original Data")

    plt.plot(df['Height (Inches)'], df['Height (Inches)'] * m + b, c="blue", label="Linear Regression Line", linewidth=1)
    plt.xlabel('Height (Inches)')
    plt.ylabel('Weight (Pounds)')
    plt.legend()
    plt.show()



def main():
    df = pd.read_csv("data/bmi_data_lab3.csv")
    # exploration(df)
    # missing_values(df)
    linear_treat(df)


if __name__ == '__main__':
    main()
