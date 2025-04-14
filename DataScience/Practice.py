# 202135835 정지호
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(
    np.array(
    [[2400, 41200],
     [2650, 50100],
     [2350, 52000],
     [4950, 66000],
     [3100, 44500],
     [2500, 37700],
     [5106, 73500],
     [3100, 37500],
     [2900, 56700],
     [1750, 35600]]
    ))
df.columns = ['spends', 'income']

x = df['spends'].values.sum()
y = df['income'].values.sum()

xy = (df['spends'].values * df['income'].values).sum()

x_squared = sum([x * x for x in df['spends'].values])
print(x, y, xy, x_squared)

n = len(df)
m = ((n * xy) - (x * y)) / ((n * x_squared) - (x * x))
b = (y - (m * x)) / n
print(n, m, b)
print( m * 3500 + b, '\n', m * 5300 + b)

sns.regplot(x=df['spends'].values, y=(m * df['spends'].values) + b, line_kws={'color': 'red'},
            scatter_kws={'color': 'r', 's': 0.1})
sns.scatterplot(x=df['spends'].values, y=df['income'].values, color = 'blue')
plt.grid(True)
plt.show()
