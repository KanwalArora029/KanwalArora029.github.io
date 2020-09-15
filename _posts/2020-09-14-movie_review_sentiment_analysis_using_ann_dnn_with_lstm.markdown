---
layout: post
title:      "Movie Review Sentiment Analysis Using (ANN, DNN with LSTM)"
date:       2020-09-14 15:41:10 -0400
permalink:  movie_review_sentiment_analysis_using_ann_dnn_with_lstm
---

```
# to parse HTML contents
from bs4 import BeautifulSoup

# for numerical analysis
import numpy as np 
# to store and process in a dataframe
import pandas as pd 

# for ploting graphs
import matplotlib.pyplot as plt
# advancec ploting
import seaborn as sns
# to create word clouds
from wordcloud import WordCloud, STOPWORDS 

# To encode values
from sklearn.preprocessing import LabelEncoder
# Convert a collection of text documents to a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
# confusion matrix
from sklearn.metrics import confusion_matrix
# train test split
from sklearn.model_selection import train_test_split

# for deep learning 
import tensorflow as tf
# to tokenize text
from tensorflow.keras.preprocessing.text import Tokenizer
# to pad sequence 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Dropout,Embedding,LSTM,Conv1D,GlobalMaxPooling1D,Flatten,MaxPooling1D,GRU,SpatialDropout1D,Bidirectional
from tensorflow.keras.optimizers import Adam
```

```
# Displaying some records from 
database
df_movie.head()
```

```
# Shape of database. Showing rows and columns
df_movie.shape
```

```
# Description of databse
df_movie.describe()
```

```
# How many values of negetive and positive
df_movie['sentiment'].value_counts()
```

```
# Dividing the data into train and test data frame
test_df = df_movie.drop('sentiment', axis=1)
train_df = df_movie
```

```
# Spliting test database for 
final_test_df = test_df[40000:]
```

```
# Splitting Train data for training and validation
final_train_df = train_df[:40000]
```


```
# Training Data Shape
final_train_df.shape
```

```
# Testing Data Shape
final_test_df.shape
```

```
# converting dataframe to csv dataset
final_test_df.to_csv('Data/testData.csv', index=False )
```

```
# Getting train data values of negetive and positive
final_train_df['sentiment'].value_counts()
```


```
# Importing and initalising LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
```

`y = final_train_df['sentiment'].values`






