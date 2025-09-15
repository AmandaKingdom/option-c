# File: stock_prediction.py
# Authors: Bao Vo and Cheong Koo
# Date: 14/07/2021(v1); 19/07/2021 (v2); 02/07/2024 (v3)

# Code modified from:
# Title: Predicting Stock Prices with Python
# Youtuble link: https://www.youtube.com/watch?v=PuZY9q-aKLw
# By: NeuralNine

# Need to install the following (best in a virtual env):
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import tensorflow as tf
import yfinance as yf       # data = web.DataReader(COMPANY, DATA_SOURCE, TRAIN_START, TRAIN_END) # Read data using yahoo
import mplfinance as mpf


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer

#------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
#------------------------------------------------------------------------------
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'     # Start date to read
TRAIN_END = '2023-01-01'       # End date to read

def load_data (
        COMPANY, 
        TRAIN_START, 
        TRAIN_END, 
        scale = True,
        shuffle = True,
        lookup_step = 1, 
        split_by_date = True,
        test_size = 0.2, 
        feature_columns = ['Close', 'Volume', 'Open', 'High', 'Low'],
        nan_statedgy = "drop",
        use_cache = True,
        cache_dir = "data"
        ):
    
    """
    Loads data from Yahoo Finance source, as well as scaling, shuffling, normalizing and splitting.
    Params:
    COMPANY (str/pd.DataFrame): the ticker you want to load, examples include AAPL, TESL, etc.
    TRAIN_START, TRAIN_END: sets start and end date for data gathering.
    scale (bool): whether to scale prices from 0 to 1, default is True
    shuffle (bool): whether to shuffle the dataset (both training & testing), default is True
    lookup_step (int): the future lookup step to predict, default is 1 (e.g next day)
    split_by_date (bool): whether we split the dataset into training/testing by date, setting it 
    to False will split datasets in a random way
    test_size (float): ratio for test data, default is 0.2 (20% testing data)
    feature_columns (list): the list of features to use to feed into the model, default is everything grabbed from yahoo_fin
    nan_stratedgy: sets bhaviour for columns with missing rows
    use_chache: Load stock data from local file if available, if not load from load from web and save as cache
    cach_dir: Sets name of folder to save cache into if use_cache is set to true.
    """

    file_path = os.path.join(cache_dir, f"{COMPANY}_{TRAIN_START}_{TRAIN_END}.csv")     #create cache directory
    if use_cache and os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col = 0,parse_dates = True)              # Load data if file exists, load index column as date in datetime format
    else:
        df = yf.download(COMPANY, TRAIN_START, TRAIN_END, auto_adjust=True)

        # Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.map(lambda x: x[0])  # Keep only the data labels

        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
            df.index = pd.to_datetime(df.index)  # Ensure the index is in datetime format
            df.index.name = "Date"
            df.to_csv(file_path)



    if nan_statedgy == "drop":
        df.dropna(inplace = True)                       # If data is missing remove entire row from dataframe
    elif nan_statedgy == "fill":
        df.fillna(method = "ffill", inplace = True)     # If data is missing use foward fill to reapeat last known cell

    scalers = {}        # Initialise an empty dictionary call scalers
    for col in feature_columns:     # loop through columns
        scaler = MinMaxScaler()     # create new instance of MinMaxScaler for column
        df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))        # Gets values from column as numpy array and reshapes into a 2D column vector for fit_transform  
        scalers[col] = scaler       # Saves scaler object into dictionary

    X = []
    Y = []

    for i in range (lookup_step, len(df)):
        seq = df[feature_columns].iloc[i - lookup_step:i].values
        X.append(seq)
        Y.append(df["Close"].iloc[i])

    X = np.array(X)     # Convert to numpy array
    Y = np.array(Y)

    if split_by_date:                               
        split_index = int(len(X) * (1 - test_size))         # Calculates number of samples to put into training set
        X_train, X_test = X[:split_index], X[split_index:]  # Gets first block from beginning to split index
        y_train, y_test = Y[:split_index], Y[split_index:]  # Gets last block from slpit to end
    else:
        X_train, X_test, y_train, y_test = train_test_split(            # if not split by date 
            X, Y, test_size=test_size, shuffle=shuffle, random_state=42     # set seed for random
        )
    

    last_sequence = df[feature_columns].iloc[-lookup_step:].values

    return {
       "X_train": np.array(X_train),
        "X_test": np.array(X_test),
        "y_train": np.array(y_train),
        "y_test": np.array(y_test),
        "scalers": scalers if scale else None,
        "df": df,
        "last_sequence": last_sequence
    }



#------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
#------------------------------------------------------------------------------
PRICE_VALUE = "Close"
data_dict = load_data(COMPANY, TRAIN_START, TRAIN_END)
data = data_dict["df"]  # the original preprocessed DataFrame

scaler = data_dict["scalers"][PRICE_VALUE] 
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
# Number of days to look back to base the prediction


# To store the training data
x_train = data_dict["X_train"]
y_train = data_dict["y_train"]
# Extract preprocessed training data (already 3D for LSTM)

#------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
#------------------------------------------------------------------------------
model = Sequential() # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For som eadvances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) 
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.
    
# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=10, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

#------------------------------------------------------------------------------
# Test the model accuracy on existing data
#------------------------------------------------------------------------------
x_test = data_dict["X_test"]
y_test = data_dict["y_test"]

actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
#------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
#------------------------------------------------------------------------------

plt.plot(actual_prices, color="black", label=f"Actual {COMPANY} Price")
plt.plot(predicted_prices, color="blue", label=f"Predicted {COMPANY} Price")
plt.title(f"{COMPANY} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{COMPANY} Share Price")
plt.legend()
plt.show()

def plot_candles(data, n_days = 3):           # funtion to plot candlestick chart
    if n_days > 1:                          # if n_days is more than 1, group data into n day intervals, with 1 candle per group
        data = data.resample(f'{n_days}D').agg({    # Use panda to goup data into tme bins of n calendar days
            'Open': 'first',                    # take first opening price on n day group
            'High': 'max',                   # take maximum price in n day group
            'Low': 'min',                 # take minimum price in n day group
            'Close': 'last',              # take last closing price in n day group
            'Volume': 'sum'                # add up all volume in n day group
        }).dropna()                     # remove any rows with missing values

    mpf.plot(data, type='candle', style='charles')

plot_candles(data, n_days=50)  # 5-trading-day candlesticks

def plot_boxplot(data, column='Close', n_days=5):  # function to plot boxplot chart
    
    # Create moving windows
    windows = []
    labels = []

    for i in range(len(data) - n_days + 1):
        window = data[column].iloc[i:i + n_days]
        windows.append(window.values)
        labels.append(data.index[i].strftime('%Y-%m-%d'))  # label by window start date

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.boxplot(windows, patch_artist=True)
    plt.title(f'Moving {n_days}-Day Boxplot for {column} Prices')
    plt.xlabel('Window Start Date')
    plt.ylabel('Price')
    plt.xticks(ticks=np.arange(1, len(labels) + 1, step=max(len(labels)//10, 1)), 
               labels=labels[::max(len(labels)//10, 1)], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_boxplot(data, n_days = 700)  
#------------------------------------------------------------------------------
# Predict next day
#------------------------------------------------------------------------------
real_data = data_dict["last_sequence"]
real_data = np.expand_dims(real_data, axis=0)
prediction = model.predict(real_data)

prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??