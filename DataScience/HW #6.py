#202135835 정지호
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

def hand_LLS_Reg(df):
    x = df['spends'].values.sum()
    y = df['income'].values.sum()

    xy = (df['spends'].values * df['income'].values).sum()

    x_squared = sum([x * x for x in df['spends'].values])
    print(x, y, xy, x_squared)

    n = len(df)
    m = ((n * xy) - (x * y)) / ((n * x_squared) - (x * x))
    b = (y - (m * x)) / n
    print(n, m, b)
    print(m * 3500 + b, '\n', m * 5300 + b)

    sns.regplot(x=df['spends'].values, y=(m * df['spends'].values) + b, line_kws={'color': 'red'},
                scatter_kws={'color': 'r', 's': 0.1})
    sns.scatterplot(x=df['spends'].values, y=df['income'].values, color='blue')
    plt.grid(True)
    plt.show()


df = pd.DataFrame(np.array(
    [[2400, 41200], [2650, 50100], [2350, 52000],
     [4950, 66000], [3100, 44500], [2500, 37700],
     [5106, 73500], [3100, 37500], [2900, 56700], [1750, 35600]]))
df.columns = ['spends', 'income']

# Calculate
lr = LinearRegression()
lr.fit(df['spends'].values.reshape(-1, 1), df['income'].values.reshape(-1, 1))


x = np.linspace(1500, 5500, 100)
y = lr.coef_[0][0] * x + lr.intercept_[0]
ex_df = pd.DataFrame({'spends': x, 'income': y})


"""
    With calculated coef_, intercpet_ from 'sklearn.linear_model.LinearRegression'
    and seaborn.
    1. Draw regression line
    2. Draw scatter plot
"""
# sns.regplot(x='spends', y='income', data=ex_df, ci=None, line_kws={'color': 'red'},
#             scatter_kws={'color': 'red', 's': 0.1})
# sns.scatterplot(x='spends', y='income', data=df, color='b')


"""
    Just using sns.regplot.
"""
sns.regplot(x='spends', y='income', data=df, ci=None, color='b', line_kws={'color': 'red'},
            scatter_kws={'color': 'blue', 's': 10})
plt.grid(True)
plt.show()
