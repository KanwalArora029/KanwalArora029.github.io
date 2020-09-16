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
   review	                                                                  sentiment
0	One of the other reviewers has mentioned that ...	positive
1	A wonderful little production. <br /><br />The...	positive
2	I thought this was a wonderful way to spend ti...	positive
3	Basically there's a family where a little boy ...	negative
4	Petter Mattei's "Love in the Time of Money" is...	positive
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


```
plt.plot(hist.history['accuracy'], label='Accuracy (training data)')
plt.plot(hist.history['val_accuracy'], label='Accuracy (validation data)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()
```

![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAYsAAAEKCAYAAADjDHn2AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDMuMC4zLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvnQurowAAIABJREFUeJzt3Xl4VOXZ+PHvnclOFkjCHiAICLIvEUERURZxF2lValuXVvu2ota+trXVblqrr/Wtleprfy6IWsUdRItaq1BkURJQUMIOgYSwZCNkT2bm/v1xJmEI2YAMk+X+XNe5cubMWe6ZJHPPs5znEVXFGGOMaUxIsAMwxhjT+lmyMMYY0yRLFsYYY5pkycIYY0yTLFkYY4xpkiULY4wxTbJkYYwxpkmWLIwxxjTJkoUxxpgmhQY7gJaSlJSkKSkpwQ7DGGPalHXr1uWpatem9ms3ySIlJYX09PRgh2GMMW2KiOxpzn5WDWWMMaZJliyMMcY0yZKFMcaYJrWbNgtjWovq6mqys7OpqKgIdijG1IqMjCQ5OZmwsLCTOt6ShTEtLDs7m9jYWFJSUhCRYIdjDKpKfn4+2dnZ9O/f/6TOYdVQxrSwiooKEhMTLVGYVkNESExMPKXSriULYwLAEoVpbU71b9KqoYwxpg1RVTxepcrjpdrtpcrjJUSExJiIgF7XShbGtEOLFi1CRNiyZUuwQzkp5eXlXHDBBXg8HjIzM3n11VdP6jznnntuk/v88Ic/JCMj46TO35jf//73PPbYY43us3jx4uOurapUe7yUVro5XFbFoeIK/vjoX3h03t/ZeqCYTTlHyNh/hB2HSthTUMb+ogoKy6pbPP66LFkY0w4tXLiQSZMm8dprrwX0Oh6PJyDnnT9/Ptdccw0ul6vRZOF2uxs9z+rVq5u81nPPPcfQoUNPKs6ToapUub2UVLp54613WJ3+FdkFZezKLWHrgSN8k3OEzfuPsDO3hL0FZRwoqmDm7DksePZpIsNCSOgUTq/4KPoldmJQtxiG9oxjQNdOAY/bkoUx7UxJSQmrVq3i+eefPy5ZPProo4wYMYJRo0Zx7733ArBjxw6mTZvGqFGjGDt2LDt37mT58uVcfvnltcfNnTuXBQsWAM7QOg888ACTJk3izTff5Nlnn+Xss89m1KhRzJ49m7KyMgAOHjzIrFmzGDVqFKNGjWL16tX85je/4Yknnqg973333ce8efOOew2vvPIKV111FQD33nsvn332GaNHj+bxxx9nwYIFfPvb3+aKK65gxowZlJSUMHXqVMaOHcuIESN49913a88TExMDwPLly5kyZQrf+ta3GDJkCDfccAOqCsCUKVNqhwqKiYnhvvvuY9SoUUyYMIGDBw8CsHPnTiZMmMDZZ5/Nb3/729rz1vXQQw8xePBgpk6bRsbmLVRUuzlUXMGjf32KkWPGMWTYCKZfdhVf7j7AOx98wj/ff4/f3f8rpp0/gd27d7Fo4Ut874qL+M4lk7l/7s0kx7oY1iue1AE9GTzwDA7u3ESvzlEkxUYQHxVGVHgooa6Q09JGZm0WxgTQH97bREbOkRY959BecfzuimENPr948WJmzpzJmWeeSUJCAuvXr2fs2LF88MEHLF68mC+++ILo6GgKCgoAuOGGG7j33nuZNWsWFRUVeL1esrKyGo0hMjKSlStXApCfn8+tt94KwP3338/zzz/PHXfcwZ133skFF1zAokWL8Hg8lJSU0KtXL6655hruuusuvF4vr732GmvXrj3m3FVVVezatYuagUEfeeQRHnvsMd5//30AFixYwJo1a9i4cSMJCQm43W4WLVpEXFwceXl5TJgwgSuvvPK4D9Avv/ySTZs20atXL8477zxWrVrFpEmTjtmntLSUCRMm8NBDD/GLX/yCZ599lvvvv5+77rqLu+66izlz5vD3v/8dALfHaS+ocnupdHtJT1/HS/94lVeX/ofKqiquv2QKfQYN40BRBZOmXcY137mR8NAQ/vLwH1j27uvcPncuH195JVdcfjnXXvttAEYP6M29d8+tfS9feWkBd9xxBwCpqal89tlnjB8/vtHfTaBYsjCmnVm4cCE//elPAbj++utZuHAhY8eO5d///jc333wz0dHRACQkJFBcXMy+ffuYNWsW4CSB5rjuuutq17/55hvuv/9+Dh8+TElJCRdffDEAn376KS+99BIALpeL+Ph44uPjSUxM5Msvv+TgwYOMGTOGxMTEY86dl5dH586dG73+9OnTSUhIAJxqnV//+tesWLGCkJAQ9u3bx8GDB+nRo8cxx4wfP57k5GQARo8eTWZm5nHJIjw8vLZENXbsWP718ceUVrpZvWYNf3/xNfbml3HO9Cvw6j1k7D/2S8DKlZ8x49Ir6N4l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%0A)






