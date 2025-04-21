# # NumPy Exercise-1
# import numpy as np
#
# np.random.seed(42)
# wt = np.random.uniform(40.0, 90.0, size=100)
# ht = np.random.randint(140, 201, size=100)
#
# ht = ht / 100.
# bmi = wt / (ht * ht)
#
# # NumPy Exercise-2
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import Counter
#
# bins = [0, 18.5, 24.9, 29.9, bmi.max() + 1]
# labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
#
# bmi_categories = np.digitize(bmi, bins)
#
# freq = [np.sum(bmi_categories == i + 1) for i in range(len(labels))]
# plt.bar(labels, freq)
# plt.show()
#
# plt.hist(bmi, bins)
# plt.xticks(bins)
# plt.show()
#
# plt.pie(freq, labels=labels, autopct='%.2f%%')
# plt.show()
#
# colors = ['blue', 'green', 'orange', 'red']
# markers = ['o', 's', '^', 'D']
#
# for i in range(len(labels)):
#     idx = bmi_categories == i + 1
#     plt.scatter(ht[idx], wt[idx], color=colors[i], marker=markers[i], label=labels[i], alpha=0.7, edgecolor='black')
#
# plt.xlabel("Height (m)")
# plt.ylabel("Weight (kg)")
# plt.title("Scatter Plot of Height vs Weight by BMI Category")
# plt.legend(title="BMI Category", loc="best")
# plt.show()
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성
np.random.seed(42)
wt = np.random.uniform(40.0, 90.0, size=100)
ht = np.random.randint(140, 201, size=100) / 100.
bmi = wt / (ht * ht)

# BMI 구간 정의
bins = [0, 18.5, 24.9, 29.9, bmi.max() + 1]
bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese']
colors = ['blue', 'green', 'orange', 'red']
bmi_categories = np.digitize(bmi, bins)
print(bmi_categories)
# subplot 구성
plt.figure(figsize=(14, 10))

# 1. Bar Plot
plt.subplot(2, 2, 1)
freq = [np.sum(bmi_categories == i + 1) for i in range(len(bmi_labels))]
plt.bar(bmi_labels, freq, color=colors, edgecolor='black')
plt.title("BMI Category Bar")
plt.ylabel("Count")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 2. Histogram
plt.subplot(2, 2, 2)
plt.hist(bmi, bins=bins, color='gray', edgecolor='black')
plt.xticks(bins)
plt.title("BMI Histogram")
plt.xlabel("BMI")
plt.ylabel("Frequency")

# 3. Pie Chart
plt.subplot(2, 2, 3)
plt.pie(freq, labels=bmi_labels, autopct='%.2f%%', colors=colors)
plt.title("BMI Pie Chart")


points = np.column_stack((ht, wt))
# 4. Scatter Plot
plt.subplot(2, 2, 4)
for i in range(1, len(bmi_labels) + 1):
    cluster_points = points[bmi_categories == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                color=colors[i - 1], label=bmi_labels[i - 1],
                edgecolor='black', alpha=0.7)
plt.title("Height vs Weight by BMI")
plt.xlabel("Height (m)")
plt.ylabel("Weight (kg)")
plt.legend()

plt.tight_layout()
plt.show()

