---
layout: post
title:      "Movie Review Sentiment Analysis Using (ANN, DNN with LSTM)"
date:       2020-09-14 19:41:09 +0000
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
from tensorflow.keras.optimizers import Adam```

```
## Importing Database from files
df_movie = pd.read_csv('Data/IMDB Dataset.csv');

# Displaying some records from 
```database
df_movie.head()
```

![](https://github.com/KanwalArora029/Sentiment_Analysis_Capstone_Project/blob/master/images/img2.png)










