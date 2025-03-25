import numpy as np
import pandas as pd

arr = np.array([3.0, '?', 2.0, 5.0, '*', 4.0, 5.0, 6.0, '+', 3.0, 2.0, '&', 5.0, '?', 7.0, '!'])
arr.shape = (4, 4)
df = pd.DataFrame(arr)

df1 = pd.DataFrame([[3.0, '?', 2.0, 5.0], ['*', 4.0, 5.0, 6.0], ['+', 3.0, 2.0, '&'], [5.0, '?', 7.0, '!']])
df.columns = ['A', 'B', 'C', 'D']

#  Display the DataFrame
print('Original DataFrame\n',df, '\n', df1)

#  Replace any non-numeric value with NaN.
df = df.replace(to_replace='[^a-zA-z0-9\.]', value = np.    nan, regex=True)

# Replace all columns type into float.
df = df.astype(float)
print("\nReplaced DataFrame\n", df)

#  isna with any, and sum
print('\n.isna().any()\n', df.isna().any())
print('\n.isna().sum()\n', df.isna().sum())

#  dropna with how any, how all, thresh 1, thresh 2
print('\nhow any\n', df.dropna(how='any'))
print('\nhow all\n', df.dropna(how='all'))
print('\nthresh1\n', df.dropna(thresh=1))
print('\nthresh2\n', df.dropna(thresh=2))
print('\nthresh3\n', df.dropna(thresh=3))

#  fillna with 100, mean, median
print('\nfillna 100\n', df.fillna(100))
print('\nfillna mean\n', df['A'].fillna(df['A'].mean()))
print('\nfillna median\n', df['B'].fillna(df['B'].median()))

#  ffill, bfill
print('\nffill\n', df.ffill())
print('\nbfill\n', df.bfill())

