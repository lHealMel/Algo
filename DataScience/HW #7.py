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
    plt.show()  # weight & hegith histogram per bmi : 0.0 ~ 4.0

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


# P.57, Data exploration
def exploration(df):
    print(df.describe())  # statistical data
    print(df.columns)  # feature names
    print(df.dtypes)  # data types, == df.info()
    bmi_sh(df)  # plot histograms, scaling results


# P.58
def find_outlier(df):
    m, b = e(df)  # y = mx + b
    error = df['Weight (Pounds)'] - (df['Height (Inches)'] * m + b)  # y - y^hat
    z_e = (error - error.mean()) / np.std(error)

    # check z_scores when bmi = 0, 4
    # mask_zero = df["BMI"] == 0.0
    # mask_four = df["BMI"] == 4.0
    # print(z_e[mask_zero], '\n', z_e[mask_four])

    plt.figure()
    plt.hist(z_e, bins=10, color='blue')
    # plt.hist(z_e[mask_zero], bins=10, color= 'red')
    # plt.hist(z_e[mask_four], bins=10, color= 'green')
    plt.xlabel('Z_score')
    plt.ylabel('Counts')
    plt.title("distribution of z_e")
    plt.tight_layout()
    plt.show()

    return z_e
    # with inspection from histogram & actual values,
    # assume that a = 1.8 for when z_e < -α, BMI = 0, when z_e>α, set BMI = 4; However not that accurate


def female_male(df):
    # for Female
    d_female = df[df['Sex'] == 'Female']
    exploration(d_female)
    z_female = find_outlier(d_female)
    mask_zero_f = d_female["BMI"] == 0.0
    mask_four_f = d_female["BMI"] == 4.0
    print(z_female[mask_zero_f], '\n', z_female[mask_four_f])

    # for Male
    d_male = df[df['Sex'] == 'Male']
    exploration(d_male)
    z_male = find_outlier(d_male)
    mask_zero_m = d_male["BMI"] == 0.0
    mask_four_m = d_male["BMI"] == 4.0
    print(z_male[mask_zero_m], '\n', z_male[mask_four_m])

    plt.figure(figsize=(12, 10))
    plt.axvline(-1.8, color='red', linestyle='--')
    plt.axvline(1.8, color='red', linestyle='--')
    plt.hist(pd.concat([z_female[mask_zero_f], z_female[mask_four_f]], axis=0), bins=10, color='green', label='Female',
             alpha=0.5)
    plt.hist(pd.concat([z_male[mask_zero_m], z_male[mask_four_m]], axis=0), bins=10, color='blue', label='Male',
             alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    # Data assumed no missing or wrong values
    df = pd.read_excel("data/bmi_data_phw3.xlsx")

    pd.set_option('display.max_rows', None)

    exploration(df)
    find_outlier(df)
    female_male(df)


if __name__ == '__main__':
    main()
