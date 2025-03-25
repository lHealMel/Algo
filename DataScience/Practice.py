import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# return (-1, col) shaped np array
def col_based_array(col, tmp_list):
  list_len = len(tmp_list)
  if list_len % col != 0:
    for i in range(col - (list_len%col)):
      tmp_list.append(np.nan)
  arr_result = np.array(tmp_list).reshape(-1, 6)
  return arr_result

str1 = """Dear Sir, SEEKING YOUR IMMEDIATE ASSISTANCE. Please permit 
me to make your acquaintance in so informal a manner. My name 
is. DAN PATRICK of the Democratic Republic of Congo and One of 
the close aides to the former President of the Democratic Republic 
of Congo LAURENT KABILA of blessed memory, may his soul rest in 
peace."""

str1_token = word_tokenize(str1)
stop_words = set(stopwords.words('english'))

result = []
for word in str1_token:
    if word not in stop_words and len(word)>2: # remove stop words and words len < 2
        result.append(word.lower())

shaped_result = col_based_array(6, result)
print('After tokenization, lowercase remove stopwords, shortwords:\n',shaped_result, '\n', shaped_result.shape)