# --------------------------------------
#   Ryan Pauly
#   main.py 
#
#   Grab and Analyze Stock Data via API.
#
#
# --------------------------------------

import datetime as dt
from datetime import date
from datetime import timedelta

import matplotlib.pyplot as plt
from matplotlib import style

import pandas as pd
# Grab data from yahoo stock data
import pandas_datareader.data as web

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# Style...
style.use('ggplot')


class user_data:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __str__(self):
        return str(self.start) + " " + str(self.end)


def get_stock_data(symbol, limit):

    """ 
    
    This function will acquire historical information from yahoo finance
    based off the given symbol (ticker) value passed in. 

    """
    
    df = web.DataReader(symbol, 'yahoo', limit.start, limit.end)
    return df
    

def plot_stock_data(data, ticker):

    """ 
    
    This function will take the acquired historical stock data and visualize it into graphs
    as well as show both a moving averages plot and a bollinger bands plot to indiicate potential
    outcomes of the stock price.

    """

    
    # MOVING AVERAGES:
    data["SMA_SHORT"] = data['Close'].rolling(window=50).mean()
    data["SMA_LONG"] = data['Close'].rolling(window=200).mean()
    data["exp_MA"] = data['Close'].ewm(halflife=0.5, min_periods=20).mean()


    plt.figure(figsize=(10,10))
    plt.plot(data['Close'], color='cyan', ls='solid', label="Close")
    plt.plot(data["SMA_SHORT"], color='green', ls='dashed', label="50-day s-moving-average")
    plt.plot(data["SMA_LONG"], color='red', ls='dashed', label="200-day s-moving-average")
    plt.plot(data["exp_MA"], color='blueviolet', ls='dashed', label="20-day exp moving-average")

    plt.legend()
    plt.xlabel("date")
    plt.ylabel("$ price per share")
    plt.title(ticker + " HISTORICAL STOCK PRICE | MOVING AVERAGES")
    
    plt.show()


    # BOLLINGER BANDS --> Note: A pinch between upper and lower bands suggests a spike in value.
    data["UPPER_BAND"] = data['Close'].rolling(window=20).mean() + data['Close'].rolling(window=20).std()*2
    data["MIDDLE_BAND"] = data['Close'].rolling(window=20).mean() 
    data["LOWER_BAND"] = data['Close'].rolling(window=20).mean() - data['Close'].rolling(window=20).std()*2

    plt.figure(figsize=(10,10))
    
    plt.plot(data['Close'].iloc[-200:], color='cyan', ls='solid', label="Close")
    plt.plot(data["UPPER_BAND"].iloc[-200:], color='green', ls='dashed', label="UPPER_BAND")
    plt.plot(data["MIDDLE_BAND"].iloc[-200:], color='blue', ls='dashed', label="MIDDLE_BAND")
    plt.plot(data["LOWER_BAND"].iloc[-200:], color='red', ls='dashed', label="LOWER_BAND")

    plt.legend()
    plt.xlabel("date")
    plt.ylabel("$ price per share")
    plt.title(ticker + " HISTORICAL STOCK PRICE | BOLLINGER BANDS")
    plt.show()



def predict_stock_price(data, symbol):

    # First we'll determine the number of training days based off the shape
    # of the pandas dataframe

    print("Training days: ", data.shape)

    # Predict "X" days in the future: i.e. 30 will mean we're predicting 30 days of stock value
    future_trend_x = 20

    data = data[['Close']]

    data["prediction"] = data[['Close']].shift(-future_trend_x)
    print(data.head())
    print(data.tail())


    x = np.array(data.drop(["prediction"], 1))[:-future_trend_x]

    y = np.array(data["prediction"])[:-future_trend_x]

    #   SPLIT DATA 75% train and 25% test
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.25)

    future_days = data.drop(["prediction"], 1)[:-future_trend_x]
    future_days = future_days.tail(future_trend_x)
    future_days = np.array(future_days)
    
    # LINEAR REGRESSION:
    linear_regression_model = LinearRegression().fit(xtrain, ytrain)
    linear_prediction = linear_regression_model.predict(future_days)

    #DECISION TREE
    decision_tree_model = DecisionTreeRegressor().fit(xtrain, ytrain)
    tree_prediction = decision_tree_model.predict(future_days)

    #VISUALIZE PREDICTIONS:


    # TREE VISUALIZATION
    predictions = tree_prediction
    valid = data[x.shape[0]:]
    valid["prediction"] = predictions

    plt.figure(figsize=(10, 10))
    plt.title(symbol + " Stock Price Prediction with Decision Tree Regressor Model")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD $")
    plt.plot(data['Close'])
    plt.plot(valid[['Close', 'prediction']])
    plt.legend(["Original", "Valid", "Predictions"])
    plt.show()

    # LINEAR REGRESSION VISUALIZATION
    predictions = linear_prediction
    valid = data[x.shape[0]:]
    valid["prediction"] = predictions

    plt.figure(figsize=(10, 10))
    plt.title(symbol + " Stock Price Prediction with LINEAR REGRESSION Model")
    plt.xlabel("Date")
    plt.ylabel("Close Price USD $")
    plt.plot(data['Close'])
    plt.plot(valid[['Close', 'prediction']])
    plt.legend(["Original", "Valid", "Predictions"])
    plt.show()


if __name__ == "__main__":

    today = date.today()
    print("\n\n\n\n****************\nWELCOME\nToday's date is: ", today)

    # We will make end always be the current day according to system clock.

    test = user_data(dt.datetime(2019,1,1), dt.datetime.now())

    ticker = 'AMD'

    # Acquire Historical Stock Data from Yahoo Finance
    historical_stock_data = get_stock_data(ticker, test)
    
    #print(historical_stock_data)

    # Plot the acquired stock data
    plot_stock_data(historical_stock_data, ticker)

    # Predict future stock price
    #predict_stock_price(historical_stock_data, ticker)



