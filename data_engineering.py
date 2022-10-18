import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
import pandas_datareader as pdr
from yahoo_fin import stock_info as si

# Getting the data

def getTickers():
    return si.tickers_sp500()

def downloadReturns(startDate, endDate, tickers):
    prices = pdr.get_data_yahoo(tickers, startDate, endDate)["Close"]
    for stock in prices.columns:
        prices[stock] = prices[stock].pct_change()
    return prices.iloc[1:]

def getData(startDate, endDate, tickers, downloaded = True):
    path = os.getcwd() + "/data/returns_data.csv"
    if downloaded == True:
        if os.path.exists(path) == True:
            print("Data was loaded sucessfully!")
            returns = pd.read_csv(path).set_index("Date")
            returns.index = pd.to_datetime(returns.index)
            return returns.dropna(axis = 1, how = "any")
        else:
            print("File doesnt exist. Please download first.")
            return None
    else:
        if os.path.exists(path) == False:
            print("Download in running....")
            returns = downloadReturns(startDate, endDate, tickers)
            returns.to_csv(path)
            print("Data was downloaded sucessfully!")
            returns.index = pd.to_datetime(returns.index)
            return returns.dropna(axis = 1, how = "any")
        else:
            print("File already exists. Please remove before downloading a new one.")
            return None

def getCorrMatrix(returns):
    return np.corrcoef(returns, rowvar=False)