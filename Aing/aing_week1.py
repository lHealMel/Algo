# 1) pandas, numpy, seaborn, matplotlib을 불러오세요
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 3) Seaborn의 load_dataset 함수를 활용하여 tips 데이터를 불러와 변수 df에 저장하세요
df = pd.read_csv('tips.csv')

df.head(5)

# [퀴즈 1-1] df의 행과 열 크기를 출력해보세요
print("데이터프레임 크기:", df.shape)

# [퀴즈 1-2] df의 컬럼명을 출력해보세요
print("컬럼명:", df.columns)

# [퀴즈 1-3] df의 기본 정보를 확인해보세요
print(df.info)

# [퀴즈 1-4] 기술통계도 확인해보세요 (문자열 포함 전체 보려면 include='all' 사용)
print(df.describe(include='all'))

# [퀴즈 1-5] 결측치(null)가 있는지 확인해보세요
print(df.isna().sum())

#######################################################
# Step 2. 전처리 - 문자열 처리, 범주형 변환, 이상치 처리
#######################################################

# [퀴즈 2-1] 수치형 컬럼(total_bill, tip)에서  '$' 제거하고 float으로 변환

# df['total_bill'] = df['total_bill'].str.replace('$', '', regex=False).astype(float)
# df['tip'] = df['tip'].str.replace('$', '', regex=False).astype(float)
df['total_bill'] = df['total_bill'].apply(lambda x: float(x[1:]))
df['tip'] = df['tip'].apply(lambda x: float(x[1:]))

# 이것도 가능
# df['total_bill'] = df['total_bill'].str.replace('$', '', regex=False).astype(float)
# df['tip'] = df['tip'].str.replace('$', '', regex=False).astype(float)

print(df[['total_bill', 'tip']].head())
print(df[['total_bill', 'tip']].info())

# [퀴즈 2-2] 범주형 컬럼(day, time)을 category 타입으로 변환
df['sex'] = df['sex'].astype('category')
df['smoker'] = df['smoker'].astype('category')
df['day'] = df['day'].astype('category')
df['time'] = df['time'].astype('category')

print(df[['sex', 'smoker', 'day', 'time']].info())

#######################################################
# Step 3. 단변량(수치형) 분석
#######################################################

# [퀴즈 3-1] total_bill의 히스토그램을 그려보세요
plt.hist(df['total_bill'], bins=10, edgecolor='k')
plt.xlabel("Total Bill")
plt.ylabel("Count")
plt.title("Histogram of total_bill")
plt.show()

# [퀴즈 3-2] tip의 히스토그램을 그려보세요
# (빈칸을 채워서 직접 작성해보세요)
plt.hist(df['tip'], bins=10, edgecolor='r')
plt.xlabel("tip")
plt.ylabel("Count")
plt.title("Histogram of tip")
plt.show()

# [퀴즈 3-3] customer_age의 히스토그램을 그려보세요
plt.hist(df['customer_age'], bins=10, edgecolor='r')
plt.xlabel("customer_age")
plt.ylabel("Count")
plt.title("Histogram of customer_age")
plt.show()

# [퀴즈 3-4] size의 히스토그램을 그려보세요
plt.hist(df['size'], bins=10, edgecolor='r')
plt.xlabel("size")
plt.ylabel("Count")
plt.title("Histogram of size")
plt.show()

# (2) 기본 통계 다시 한 번!
print(df.describe(include='all'))  # 문자열 컬럼도 함께 보려면 include='all'

#######################################################
# Step 4. 범주형 변수 탐색
#######################################################

# Tips 데이터의 범주형 칼럼 : sex, smoker, day, time 등
# 이 중 day와 time의 unique 값을 직접 확인해보세요

# [퀴즈 4-1] 'sex' 컬럼의 중복을 고려한 고유값들을 출력해보세요
print(df["sex"].unique())

# [퀴즈 4-2] 'smoker' 컬럼의 중복을 고려한 고유값들을 출력해보세요
print(df["smoker"].unique())

# [퀴즈 4-3] 'day' 컬럼의 중복을 고려한 고유값들을 출력해보세요
print(df["day"].unique())

# [퀴즈 4-4] 'time' 컬럼의 중복을 고려한 고유값들을 출력해보세요
print(df["time"].unique())

# [퀴즈 4-5] 각 범주의 개수를 확인해보세요
print(df["sex"].value_counts())
print(df["smoker"].value_counts())
print(df["day"].value_counts())
print(df["time"].value_counts())

# [퀴즈 4-6] Bar chart를 사용하여 각 범주별 데이터 개수를 시각화해 보세요
test = df["sex"].value_counts()

plt.bar(test.index, test.values, edgecolor='k')
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Bar chart of Sex")
plt.show()

#######################################################
# Step 5. 이변량(두 변수) 분석 & 시각화
#######################################################

# [퀴즈 5-1] total_bill과 tip의 관계를 scatterplot으로 살펴보기
# seaborn의 scatterplot(또는 relplot)을 사용해보세요
sns.scatterplot(x="total_bill", y="tip", data=df)
plt.show()

# 3) day별, time별로 주문 건수가 어떻게 다른지 시각화하기
# seaborn의 countplot을 사용하면 범주별 데이터 건수를 한눈에 볼 수 있음
sns.countplot(x="day", hue="time", data=df)
plt.show()

#######################################################
# Special step. 이상치 제거
#######################################################

# (C) IQR 방법으로 이상치 제거 (예: customer_age에서 이상치 제거)
Q1 = df['customer_age'].quantile(0.25)
Q3 = df['customer_age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

mask_age = (df['customer_age'] >= lower_bound) & (df['customer_age'] <= upper_bound)
df_clean = df[mask_age].copy()

print("\n--- customer_age 이상치 제거 전후 데이터 크기 비교 ---")
print("제거 전:", df.shape, "/ 제거 후:", df_clean.shape)

plt.hist(df_clean['customer_age'], bins=10, edgecolor='k')
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Histogram of age")
plt.show()

#######################################################
# Step 6. 그룹 분석, Pivot Table, 상관분석
#######################################################

# [퀴즈 6-1] 'sex'(성별)와 'smoker'(흡연 여부) 조합에 따른 tip 평균을 pivot_table 형태로 구해보세요
pivot_tip = pd.pivot_table(
    data=df,
    index="sex",
    columns="smoker",
    values="tip",
    aggfunc="mean",
    observed=True
)
print(pivot_tip)

# [퀴즈 6-1] day별 평균 total_bill 비교해보세요
day_mean = df_clean.groupby('day', observed=True)['total_bill'].mean()
day_mean = day_mean.sort_values(ascending=False)
print(day_mean)

# [3] 전체 변수 간 상관계수를 확인하고 heatmap으로 시각화해보세요
numeric_df = df.select_dtypes(include=[np.number])
corr_mat = numeric_df.corr()
print(corr_mat)

sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="GnBu")
plt.show()

#######################################################
# Step 6. 탐색적 분석 퀴즈
#######################################################

# [퀴즈 1] 'female'(여성)과 'male'(남성) 중 전체 평균 팁(tip)이 더 높은 성별은?
pivot_tip = pd.pivot_table(
    data=df,
    index="sex",
    values="tip",
    aggfunc="mean",
    observed=True
)
print(pivot_tip)

# [퀴즈 2] customer_age와 tip의 관계는 어떨까?
# (pearson 상관관계를 계산하거나 시각화 해보자)

pivot_tip = pd.pivot_table(
    data=df_clean,
    index="customer_age",
    values="tip",
    aggfunc="mean",
    observed=True
)

print(df_clean['customer_age'].corr(df_clean['tip']))
print(df_clean.loc[:, ['customer_age', 'tip']].corr(method='pearson'))

plt.scatter(df_clean.customer_age, df_clean.tip)
plt.xlabel("customer_age")
plt.ylabel("tip")
plt.title("Scatter plot of customer_age and tip")
plt.show()

# [퀴즈 3] 흡연 여부(smoker)에 따른 평균 팁 차이는?
pivot_tip = pd.pivot_table(
    data=df,
    index="smoker",
    values="tip",
    aggfunc="mean",
    observed=True
)
print(pivot_tip)

# [퀴즈 4] (응용) '흡연 여부'와 '요일(day)' 별로 사람들은 평균 얼마의 팁을 내는지 확인해보세요
# 결과를 보고, 어떤 요일에 흡연자가 팁을 많이 주는지, 비흡연자가 팁을 많이 주는지 등을 파악
pivot_tip = pd.pivot_table(
    data=df,
    index="smoker",
    columns="day",
    values="tip",
    aggfunc="mean",
    observed=True
)
smoke_day = df_clean['smoker'].groupby(df_clean['day'])['tip'].mean()
print(df_clean)
print(pivot_tip)
