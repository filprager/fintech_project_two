# Import libraries
import pandas as pd
import numpy as np

# Set the random seed for reproducibility
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

# -------------------- Dogecoin Section --------------------

# Load Dogecoin data
df_doge = pd.read_pickle('data/elon_doge.plk')
df_doge

# Remove extraneous columns
signals_df_doge = df_doge[["Dogecoin Price", "Does Elon Musk's Tweet Tention the Word DOGE?"]]
signals_df_doge

# Add daily returns column
signals_df_doge['Daily Return'] = signals_df_doge['Dogecoin Price'].pct_change()
signals_df_doge

# Shift DataFrame values by 1
signals_df_doge = signals_df_doge.shift(1)
signals_df_doge

# Drop NAs
signals_df_doge = signals_df_doge.dropna()
signals_df_doge

# Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
signals_df_doge['Positive Return'] = np.where(signals_df_doge['Daily Return'] > 0, 1.0, 0.0)
signals_df_doge

def window_data(df, window, feature_col_number, target_col_number):
    '''
    This function accepts 1 column number for the features (X) and one for target (y).
    It chunks the data up with a rolling window of Xt - window to predict Xt+1.
    It returns two numpy arrays of X and y.
    '''
    X = []
    y = []
    for i in range(len(df) - window):
        features = df.iloc[i : (i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)

# Define the window size
window_size = 100

# Set the index of the feature, target and extra columns
feature_column = 1 # Historical Dogecoin Price
target_column = 3 # Positive Return

# Create the features (X) and target (y) data using the window_data() function.
X, y = window_data(signals_df_doge, window_size, feature_column, target_column)

# Print a few sample values from X and y
print (f"X sample values:\n{X[:3]} \n")
print (f"y sample values:\n{y[:3]}")

# Manually splitting the data
split = int(0.7 * len(X))

# X_train = X[: split]
# X_test = X[split:]

# y_train = y[: split]
# y_test = y[split:]

X_train = X
X_test = X

y_train = y
y_test = y

X_train

# Reshape the features data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Importing required Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# Define the LSTM RNN model.
model = Sequential()

# Initial model setup
number_units = 100
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
# sigmoid
# relu


# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Show the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=100, verbose=1)

# Save ML model
model.save('model/rnn_lstm_model_doge')

# Save X_test
np.save('data/rnn_X_test_doge.npy', X_test)

# Save y_test
np.save('data/rnn_y_test_doge.npy', y_test)

# Save Dogecoin signal
signals_df_doge.to_pickle('data/rnn_signals_doge.plk')

# Evaluate the model
model.evaluate(X_test, y_test, verbose=0)

# Make predictions using the testing data X_test
predicted = model.predict_classes(X_test)

# Create a DataFrame of Real and Predicted values
result = pd.DataFrame({
    "Actual Positive Return": y_test.ravel(),
    "Predicted Positive Return": predicted.ravel()
}, index = signals_df_doge.index[-len(y_test): ]) 

# Show the DataFrame's head
result

import hvplot.pandas
result['Predicted Positive Return'].hvplot()

# Replace predicted values 0 to -1 to account for shorting
result['Predicted Positive Return'].replace(0, -1, inplace=True)
result

# Add a new column for daily return 
result = result.join(signals_df_doge['Daily Return'])
result

# Save the result
result.to_pickle('data/rnn_result_doge.plk')

# Calculate cumulative return of model and plot the result
cumulative_return = (1 + (result['Daily Return'] * result['Predicted Positive Return'])).cumprod() -1

cumulative_return_end_doge = cumulative_return[-1]
cumulative_return_end_doge

cumulative_return_plot_doge = cumulative_return.hvplot(title='Dogecoin Cumulative Returns')
cumulative_return_plot_doge

# -------------------- Bitcoin Section --------------------

# Load Bitcoin data
df_btc = pd.read_pickle('data/elon_btc.plk')
df_btc

# Remove extraneous columns
signals_df_btc = df_btc[["Bitcoin Price", "Does Elon Musk's Tweet Tention the Word Bitcoin or BTC?"]]
signals_df_btc

# Add daily returns column
signals_df_btc['Daily Return'] = signals_df_btc['Bitcoin Price'].pct_change()
signals_df_btc

# Shift DataFrame values by 1
signals_df_btc = signals_df_btc.shift(1)
signals_df_btc

# Drop NAs
signals_df_btc = signals_df_btc.dropna()
signals_df_btc

# Construct the dependent variable where if daily return is greater than 0, then 1, else, 0.
signals_df_btc['Positive Return'] = np.where(signals_df_btc['Daily Return'] > 0, 1.0, 0.0)
signals_df_btc

# Define the window size
window_size = 100

# Set the index of the feature, target and extra columns
feature_column = 1 # Historical Dogecoin Price
target_column = 3 # Positive Return

# Create the features (X) and target (y) data using the window_data() function.
X, y = window_data(signals_df_btc, window_size, feature_column, target_column)

# Print a few sample values from X and y
print (f"X sample values:\n{X[:3]} \n")
print (f"y sample values:\n{y[:3]}")

# Manually splitting the data
split = int(0.7 * len(X))

# X_train = X[: split]
# X_test = X[split:]

# y_train = y[: split]
# y_test = y[split:]

X_train = X
X_test = X

y_train = y
y_test = y

# Reshape the features data
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Importing required Keras modules
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Define the LSTM RNN model.
model = Sequential()

# Initial model setup
number_units = 100
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
# sigmoid
# relu


# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Show the model summary
model.summary()

# Train the model
model.fit(X_train, y_train, epochs=10, shuffle=False, batch_size=100, verbose=1)

# Save ML model
model.save('model/rnn_lstm_model_btc')

# Save X_test
np.save('data/rnn_X_test_btc.npy', X_test)

# Save y_test
np.save('data/rnn_y_test_btc.npy', y_test)

# Save Bitcoin signal
signals_df_doge.to_pickle('data/rnn_signals_btc.plk')

# Evaluate the model
model.evaluate(X_test, y_test, verbose=0)

# Make predictions using the testing data X_test
predicted = model.predict_classes(X_test)

# Create a DataFrame of Real and Predicted values
result = pd.DataFrame({
    "Actual Positive Return": y_test.ravel(),
    "Predicted Positive Return": predicted.ravel()
}, index = signals_df_btc.index[-len(y_test): ]) 

# Show the DataFrame's head
result

import hvplot.pandas

result['Predicted Positive Return'].hvplot()

# Replace predicted values 0 to -1 to account for shorting
result['Predicted Positive Return'].replace(0, -1, inplace=True)
result

# Add a new column for daily return 
result = result.join(signals_df_btc['Daily Return'])
result

# Save the result
result.to_pickle('data/rnn_result_btc.plk')

# Calculate cumulative return of model and plot the result
cumulative_return = (1 + (result['Daily Return'] * result['Predicted Positive Return'])).cumprod() -1

cumulative_return_end_btc = cumulative_return[-1]
cumulative_return_end_btc

cumulative_return_plot_btc = cumulative_return.hvplot(title='Bitcoin Cumulative Returns')
cumulative_return_plot_btc