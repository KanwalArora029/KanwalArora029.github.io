---
layout: post
title:      "Movie Review Sentiment Analysis Using (ANN)"
date:       2020-09-14 15:41:10 -0400
permalink:  movie_review_sentiment_analysis_using_ann_dnn_with_lstm
---

In today's age where we think twice to buy or consume any product or any service. We check the testimonies or reviews of that particular product service. In other words we check reviews given by previous customer wheater the product is worth spending money or not. From the reviews of that product or service we can get the sentiments of previous buyers. 

Here **Sentiment Analysis** comes to action. Sentiment Analysis can be of different product or of whole website.
in this blog I am trying to analyse the sentiment of a movies listed in IMDB to check how much people like it.

**Here is the code**
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

```
y = final_train_df['sentiment'].values
```

```
# processing sentiment values
y = le.fit_transform(y)
```

```

final_train_df['review']

```


### Importing Pre Processing Liabraries

```
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
```

```
stop = stopwords.words('english')
ps = PorterStemmer()
```

```
def clean_text(sample):
    sample = sample.lower()
    sample = sample.replace("<br /><br />", " ")
    sample = re.sub("[^a-zA-Z ]+", " ", sample)
    
    sample = sample.split()
    sample = [ps.stem(s) for s in sample if s not in stop]
    
    sample = " ".join(sample)
    
    return sample
```

```
final_train_df['cleaned_review'] = final_train_df['review'].apply(clean_text)
```

```
final_train_df.head()
```

```
corpus = final_train_df['cleaned_review'].values
```

```
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
```

```
cvec = CountVectorizer(max_df = 0.5, max_features=50000)
```

```
X = cvec.fit_transform(corpus)
```

```
print(X[0])
```

```
#Initialising TFIDF transformer
tfidf = TfidfTransformer()
```

```
X = tfidf.fit_transform(X)
print(X[0])
```
(40000, 50000)

```
y.shape
```
(40000,)

## Artificial Nueral Network


```
mport tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import Dense
```

```
model = models.Sequential()
model.add( Dense(16, activation = "relu", input_shape = (X.shape[1],)))
model.add( Dense(16, activation = "relu"))
model.add( Dense(1, activation='sigmoid'))
```

```
model.summary()
```


```
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 16)                800016    
_________________________________________________________________
dense_1 (Dense)              (None, 16)                272       
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 17        
=================================================================
Total params: 800,305
Trainable params: 800,305
Non-trainable params: 0
```

```
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics = ['accuracy'])
```

```
X_val = X[:5000]
X_train = X[5000:]

y_val = y[:5000]
y_train = y[5000:]
```

```
X_train.shape, y_train.shape
```
((35000, 50000), (35000,))







