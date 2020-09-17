---
layout: post
title:      "Google Stock Price Prediction"
date:       2020-09-17 17:01:22 -0400
permalink:  google_stock_price_prediction
---


```
import math
import pandas as pd
import datetime
import pandas_datareader as web
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Activation, GRU
from plotly.graph_objs import *
from plotly.offline import init_notebook_mode, iplot, iplot_mpl
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('seaborn')
```

```
# Importing the training set
data = pd.read_csv('Data/googl.csv', index_col="Date",parse_dates=True)
data
```

```
data.shape
```
(1361, 6)

```
data1 = data.filter(['Close'])
```

```
training_set = data1.values
training_data_len = math.ceil( len(training_set) * .8 )
```

```
# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
```

```
# Creating a data structure with 60 timesteps and 1 output
X_train_lstm = []
y_train_lstm = []
for i in range(60, 1361):
    X_train_lstm.append(training_set_scaled[i-60:i, 0])
    y_train_lstm.append(training_set_scaled[i, 0])
X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
```

```
# Reshaping
X_train_lstm = np.reshape(X_train_lstm, (X_train_lstm.shape[0], X_train_lstm.shape[1], 1))
```

### Part 2 - Building the RNN

```
# Initialising the RNN
regressor = Sequential()
```

```
# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_lstm.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))
```

```
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
```

```
# Fitting the RNN to the Training set
regressor.fit(X_train_lstm, y_train_lstm, epochs = 100, batch_size = 32)
```

```
Epoch 1/100
41/41 [==============================] - 9s 231ms/step - loss: 0.0276
Epoch 2/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0057
Epoch 3/100
41/41 [==============================] - 9s 208ms/step - loss: 0.0060
Epoch 4/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0054
Epoch 5/100
41/41 [==============================] - 9s 211ms/step - loss: 0.0056
Epoch 6/100
41/41 [==============================] - 9s 208ms/step - loss: 0.0043
Epoch 7/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0045
Epoch 8/100
41/41 [==============================] - 9s 211ms/step - loss: 0.0039
Epoch 9/100
41/41 [==============================] - 9s 208ms/step - loss: 0.0039
Epoch 10/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0041
Epoch 11/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0041
Epoch 12/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0040
Epoch 13/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0037
Epoch 14/100
41/41 [==============================] - 9s 209ms/step - loss: 0.0032
Epoch 15/100
41/41 [==============================] - 9s 211ms/step - loss: 0.0039
Epoch 16/100
41/41 [==============================] - 9s 211ms/step - loss: 0.0032
Epoch 17/100
41/41 [==============================] - 9s 211ms/step - loss: 0.0030
Epoch 18/100
41/41 [==============================] - 9s 213ms/step - loss: 0.0030
Epoch 19/100
41/41 [==============================] - 9s 214ms/step - loss: 0.0027
Epoch 20/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0033
Epoch 21/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0032
Epoch 22/100
41/41 [==============================] - 9s 227ms/step - loss: 0.0031
Epoch 23/100
41/41 [==============================] - 9s 214ms/step - loss: 0.0028
Epoch 24/100
41/41 [==============================] - 9s 213ms/step - loss: 0.0029
Epoch 25/100
41/41 [==============================] - 9s 226ms/step - loss: 0.0029
Epoch 26/100
41/41 [==============================] - 10s 246ms/step - loss: 0.0025
Epoch 27/100
41/41 [==============================] - 9s 228ms/step - loss: 0.0024
Epoch 28/100
41/41 [==============================] - 10s 243ms/step - loss: 0.0024
Epoch 29/100
41/41 [==============================] - 9s 228ms/step - loss: 0.0025
Epoch 30/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0026
Epoch 31/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0026
Epoch 32/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0026
Epoch 33/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0028
Epoch 34/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0022
Epoch 35/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0026
Epoch 36/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0022
Epoch 37/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0023
Epoch 38/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0022
Epoch 39/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0023
Epoch 40/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0023
Epoch 41/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0021
Epoch 42/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0020
Epoch 43/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0020
Epoch 44/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0023
Epoch 45/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0023
Epoch 46/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0022
Epoch 47/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0023
Epoch 48/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0020
Epoch 49/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0018
Epoch 50/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0020
Epoch 51/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0021
Epoch 52/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0018
Epoch 53/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0019
Epoch 54/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0018
Epoch 55/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0017
Epoch 56/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0018
Epoch 57/100
41/41 [==============================] - 9s 220ms/step - loss: 0.0016
Epoch 58/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0018
Epoch 59/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0015
Epoch 60/100
41/41 [==============================] - 9s 230ms/step - loss: 0.0017
Epoch 61/100
41/41 [==============================] - 10s 236ms/step - loss: 0.0018
Epoch 62/100
41/41 [==============================] - 9s 230ms/step - loss: 0.0017
Epoch 63/100
41/41 [==============================] - 9s 217ms/step - loss: 0.0016
Epoch 64/100
41/41 [==============================] - 9s 217ms/step - loss: 0.0017
Epoch 65/100
41/41 [==============================] - 9s 215ms/step - loss: 0.0018
Epoch 66/100
41/41 [==============================] - 9s 218ms/step - loss: 0.0016
Epoch 67/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0017
Epoch 68/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0014
Epoch 69/100
41/41 [==============================] - 9s 218ms/step - loss: 0.0017
Epoch 70/100
41/41 [==============================] - 9s 217ms/step - loss: 0.0014
Epoch 71/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0016
Epoch 72/100
41/41 [==============================] - 9s 214ms/step - loss: 0.0017
Epoch 73/100
41/41 [==============================] - 9s 208ms/step - loss: 0.0014
Epoch 74/100
41/41 [==============================] - 9s 216ms/step - loss: 0.0014
Epoch 75/100
41/41 [==============================] - 9s 223ms/step - loss: 0.0013
Epoch 76/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0014
Epoch 77/100
41/41 [==============================] - 9s 223ms/step - loss: 0.0014
Epoch 78/100
41/41 [==============================] - 9s 224ms/step - loss: 0.0013
Epoch 79/100
41/41 [==============================] - 9s 225ms/step - loss: 0.0014
Epoch 80/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0013
Epoch 81/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0012
Epoch 82/100
41/41 [==============================] - 9s 218ms/step - loss: 0.0015
Epoch 83/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0013
Epoch 84/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0015
Epoch 85/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0012
Epoch 86/100
41/41 [==============================] - 9s 221ms/step - loss: 0.0012
Epoch 87/100
41/41 [==============================] - 9s 225ms/step - loss: 0.0013
Epoch 88/100
41/41 [==============================] - 9s 224ms/step - loss: 0.0011
Epoch 89/100
41/41 [==============================] - 9s 219ms/step - loss: 0.0012
Epoch 90/100
41/41 [==============================] - 9s 218ms/step - loss: 0.0012
Epoch 91/100
41/41 [==============================] - 9s 226ms/step - loss: 0.0013
Epoch 92/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0012
Epoch 93/100
41/41 [==============================] - 9s 223ms/step - loss: 0.0011
Epoch 94/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0013
Epoch 95/100
41/41 [==============================] - 9s 230ms/step - loss: 0.0012
Epoch 96/100
41/41 [==============================] - 9s 225ms/step - loss: 0.0012
Epoch 97/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0012
Epoch 98/100
41/41 [==============================] - 9s 226ms/step - loss: 0.0013
Epoch 99/100
41/41 [==============================] - 10s 235ms/step - loss: 0.0013
Epoch 100/100
41/41 [==============================] - 9s 222ms/step - loss: 0.0011
Out[9]:
<tensorflow.python.keras.callbacks.History at 0x14e18c780>
```




