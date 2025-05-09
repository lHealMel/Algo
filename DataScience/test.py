import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.DataFrame([[-1, 0, 1],[-1, 0, 1]])
print(df.dot(df.T))