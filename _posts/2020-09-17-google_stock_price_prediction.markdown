---
layout: post
title:      "Google Stock Price Prediction"
date:       2020-09-17 21:01:21 +0000
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




