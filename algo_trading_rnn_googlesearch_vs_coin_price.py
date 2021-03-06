# LSTM RNN Google Search Terms vs Bitcoin Prices

# Description
# This code will use LSTM RNN to perform a prediction of weekly stock prices (converted from hourly) from search frequency on google (using Google Trends).
# Shifts will be used to see whether one week ahead will make a difference

# #### Assumptions
# - Weekly summary is broad - it is assumed that the effect will last more than one week.
# - For the crypto exchanges, we assume that the popularity of coinbase is steady throughout the timeline, and that impact of this on the final price prediction result is minimal. 

import numpy as np
import pandas as pd   
import hvplot.pandas
import pickle

# Initial imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler #scale

debug = True

## Data Preparation


# Set the random seed - reproduce results
from numpy.random import seed
seed(42)
from tensorflow import random
random.set_seed(42)

### Load all Bitcoin searches

# load tweets from elon
df_btc = pd.read_excel('./retrieve/google/bitcoin_global_search_2017.xlsx')

df_btc.dropna(inplace=True)

df_btc.columns

df_btc.columns

df_btc.columns = ['Week', 'btc_search']

df_btc['Week'] = pd.to_datetime(df_btc['Week'])

df_btc.head()

df_btc.set_index('Week',inplace=True)

df_btc = df_btc[['btc_search']]

df_btc.head()

### Load hourly crypto data

### Load crypto hourly data
df2 = pd.read_pickle("./retrieve/cryptocompare/CRYPTODUMP/hourlyhist_crypto_compare_api_v1_df")

df2.time_convert.min()

df2.dtypes

df2.dropna(inplace=True)

df2 = df2[df2.coin_symbol == 'BTC']

df2 = df2[df2['time_convert'] >= '2017-01-01']

df2.time_convert.min()

df3 = df2.copy()

df3.set_index('time_convert', inplace = True)


set(df3.coin_symbol)

# get the mean closing price for bitcoin weekly 
df3 = df3.resample('W').mean()

df3['WeeklyReturns'] = df3['close'].pct_change()

df3.head()

df3 = df3[['WeeklyReturns']]

df3.head()

### Join 2 dataframes by date index

# Join the data into a single DataFrame - bitcoin search and close price weekly
df_btc_close = df_btc.join(df3, how="inner")
df_btc_close.tail()

df = df_btc_close.copy()

len(df)

df.head()

df.isna().sum()

df.dropna(inplace=True)

df.head()

# shift btc search one step back
df['WeeklyReturnsShift'] = df['WeeklyReturns'].shift(-1)

df.head()

df.dropna(inplace=True)

df = df[['btc_search','WeeklyReturnsShift']]

# Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
df['PositiveReturn'] = np.where(df['WeeklyReturnsShift'] > 0, 1.0, 0.0)
df

### Window function

def window_data(df, window, feature_col_number, target_col_number):
    """
    This function accepts the column number for the features (X) and the target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt.
    It returns two numpy arrays of X and y.
    """
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i : (i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

# Creating the features (X) and target (y) data using the window_data() function.
window_size = 40

feature_column = 0
target_column = 2
X, y = window_data(df, window_size, feature_column, target_column)
print (f"X sample values:\n{X[:5]} \n")
print (f"y sample values:\n{y[:5]}")

### Get train and test split and scale

# Use 70% of the data for training and the remainder for testing
# split = int(0.7 * len(X))
# X_train = X[: split]
# X_test = X[split:]
# y_train = y[: split]
# y_test = y[split:]

X_train = X
y_train = y

X_test = X
y_test = y

### Scaling data with minmaxscaler

from sklearn.preprocessing import MinMaxScaler
x_train_scaler = MinMaxScaler()
x_test_scaler = MinMaxScaler()
y_train_scaler = MinMaxScaler()
y_test_scaler = MinMaxScaler()

# Fit the scaler for the Training Data
x_train_scaler.fit(X_train)
y_train_scaler.fit(y_train)

# Scale the training data
X_train = x_train_scaler.transform(X_train)
y_train = y_train_scaler.transform(y_train)

# Fit the scaler for the Testing Data
x_test_scaler.fit(X_test)
y_test_scaler.fit(y_test)

# Scale the y_test data
X_test = x_test_scaler.transform(X_test)
y_test = y_test_scaler.transform(y_test)

### Reshape the data

# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
print (f"X_train sample values:\n{X_train[:5]} \n")
print (f"X_test sample values:\n{X_test[:5]}")

### Build and Train the LSTM RNN

# Import required Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the LSTM RNN model.
model = Sequential()

number_units = 40
dropout_fraction = 0.2

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )
model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))
# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))
# Output layer
model.add(Dense(1, activation='sigmoid'))

#### compile the model

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Summarize the model
model.summary()

#### train model

# Train the model
model.fit(X_train, y_train, epochs=120, shuffle=True, batch_size=10, verbose=1)

#### Evaluate the model

model.evaluate(X_test, y_test)

#### Make predictions

# Make some predictions
predicted = model.predict_classes(X_test)

# Recover the original closing price instead of the scaled version
predicted_prices = y_test_scaler.inverse_transform(predicted)
real_prices = y_test_scaler.inverse_transform(y_test.reshape(-1, 1))

### Plot Outcome

# Create a DataFrame of Real and Predicted values
result = pd.DataFrame({
    "Actual Positive Return": real_prices.ravel(),
    "Predicted Positive Return": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 

# Show the DataFrame's head
result.head()

# Plot the real vs predicted prices as a line chart
result.hvplot()

# Replace predicted values 0 to -1 to account for shorting
result['Predicted Positive Return'].replace(0, -1, inplace=True)
result

# Add a new column for daily return 
result = result.join(df['WeeklyReturnsShift'])
result

# Save ML model
model.save('model/rnn_lstm_model_btcgoogsearch')

# Save X_test
np.save('data/rnn_X_test_btcgoogsearch.npy', X_test)

# Save y_test
np.save('data/rnn_y_test_btcgoogsearch.npy', y_test)

# Save Bitcoin signal
result.to_pickle('data/rnn_result_btcgoogsearch.plk')

# Evaluate the model
model.evaluate(X_test, y_test, verbose=0)

## References

# ##### Rounding date time to hour
# https://stackoverflow.com/questions/32344533/how-do-i-round-datetime-column-to-nearest-quarter-hour 