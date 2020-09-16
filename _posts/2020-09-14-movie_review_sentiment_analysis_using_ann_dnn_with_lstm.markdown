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


```
hist = model.fit(X_train.todense(), y_train, batch_size=128, epochs=15, validation_data=(X_val.todense(), y_val))
```

```
Epoch 1/15
274/274 [==============================] - 17s 63ms/step - loss: 0.4342 - accuracy: 0.8597 - val_loss: 0.2574 - val_accuracy: 0.8988
Epoch 2/15
274/274 [==============================] - 13s 49ms/step - loss: 0.2071 - accuracy: 0.9184 - val_loss: 0.2422 - val_accuracy: 0.8986
Epoch 3/15
274/274 [==============================] - 12s 45ms/step - loss: 0.1645 - accuracy: 0.9378 - val_loss: 0.2562 - val_accuracy: 0.9012
Epoch 4/15
274/274 [==============================] - 11s 42ms/step - loss: 0.1392 - accuracy: 0.9482 - val_loss: 0.2696 - val_accuracy: 0.8986
Epoch 5/15
274/274 [==============================] - 12s 42ms/step - loss: 0.1223 - accuracy: 0.9549 - val_loss: 0.2866 - val_accuracy: 0.8940
Epoch 6/15
274/274 [==============================] - 12s 43ms/step - loss: 0.1103 - accuracy: 0.9611 - val_loss: 0.3017 - val_accuracy: 0.8930
Epoch 7/15
274/274 [==============================] - 10s 36ms/step - loss: 0.0985 - accuracy: 0.9647 - val_loss: 0.3219 - val_accuracy: 0.8888
Epoch 8/15
274/274 [==============================] - 13s 47ms/step - loss: 0.0895 - accuracy: 0.9687 - val_loss: 0.3416 - val_accuracy: 0.8826
Epoch 9/15
274/274 [==============================] - 14s 53ms/step - loss: 0.0810 - accuracy: 0.9724 - val_loss: 0.3553 - val_accuracy: 0.8842
Epoch 10/15
274/274 [==============================] - 14s 49ms/step - loss: 0.0732 - accuracy: 0.9749 - val_loss: 0.3723 - val_accuracy: 0.8820
Epoch 11/15
274/274 [==============================] - 14s 49ms/step - loss: 0.0655 - accuracy: 0.9774 - val_loss: 0.3968 - val_accuracy: 0.8804
Epoch 12/15
274/274 [==============================] - 12s 42ms/step - loss: 0.0585 - accuracy: 0.9807 - val_loss: 0.4206 - val_accuracy: 0.8760
Epoch 13/15
274/274 [==============================] - 13s 49ms/step - loss: 0.0522 - accuracy: 0.9830 - val_loss: 0.4487 - val_accuracy: 0.8750
Epoch 14/15
274/274 [==============================] - 12s 43ms/step - loss: 0.0466 - accuracy: 0.9849 - val_loss: 0.4683 - val_accuracy: 0.8718
Epoch 15/15
274/274 [==============================] - 13s 46ms/step - loss: 0.0413 - accuracy: 0.9866 - val_loss: 0.5017 - val_accuracy: 0.8708
```


## Visualization of Model

```
plt.plot(hist.history['loss'], label='Loss (training data)')
plt.plot(hist.history['val_loss'], label='Loss (validation data)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
```

![](data:image/png;base64,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%0A)






