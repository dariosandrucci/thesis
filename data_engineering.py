#---PACKAGES---#

import pandas as pd
import numpy as np
import os
from datetime import datetime as dt
import pandas_datareader as pdr
from yahoo_fin import stock_info as si

#---FUNCTIONS---#

#SP500 tickers
#Use: Get the tickers for a selected market (This version only SP500)

def getTickers(market = "SP500"):
    if market == "SP500":
        return si.tickers_sp500()
    else:
        pass

#PRICE DOWNLOADER
#Downlads the price data for a given set of securities

def downloadPrices(startDate, endDate, tickers):
    prices = pdr.get_data_yahoo(tickers, startDate, endDate)["Close"]
    return prices

#RETURN CONVERTER
#Calculated a returns dataframe froma set of prices

def getReturns(prices):
    for stock in prices.columns:
        prices[stock] = prices[stock].pct_change()
    return prices.iloc[1:]

#DATASET LOADER
#Combines all functions above to provide a desired dataframe for any required experiments

def getData(startDate, endDate, tickers, format = "returns", downloaded = True):
    path = os.getcwd() + "/data/returns_data.csv"
    if downloaded == True:
        if os.path.exists(path) == True:
            returns = pd.read_csv(path).set_index("Date")
            if format == "returns":
                returns = getReturns(returns)
            else: pass
            returns.index = pd.to_datetime(returns.index)
            print("Data was loaded sucessfully!")
            return returns.dropna(axis = 1, how = "any")
        else:
            print("File doesnt exist. Please download first.")
            return None
    else:
        if os.path.exists(path) == False:
            print("Download in running....")
            returns = downloadPrices(startDate, endDate, tickers)
            returns.to_csv(path)
            print("Data was downloaded sucessfully!")
            if format == "returns":
                returns = getReturns(returns)
            else: pass
            returns.index = pd.to_datetime(returns.index)
            return returns.dropna(axis = 1, how = "any")
        else:
            print("File already exists. Please remove before downloading a new one.")
            return None

#Calculates the empirical covariance matrix from a given dataset
#Downlads the price data for a given set of securities

def getCorrMatrix(returns):
    return np.corrcoef(returns, rowvar = 0)

#TRAIN (VAL) TEST SPLITTER
#Splits datasets along the timeaxis to retrieve train, validation and test sets

def testTrainSplit(df, validation_set = True, w_tt = [0.8, 0.2], w_tvt = [0.4,0.4,0.2]):
    if validation_set == True:
        if sum(w_tvt) == 1: pass
        else: raise NameError("sum of w must be 1")
        
        len_df = len(df.index)
        len_train = round(len_df*w_tvt[0])
        len_val = round(len_df*w_tvt[1])
        len_test = len_df - len_train - len_val

        train_set = df.iloc[0:len_train-1]
        val_set = df.iloc[len_train-1:len_train+len_val-1]
        test_set = df.iloc[len_train+len_val-1:len_train+len_val+len_test-1]

        return train_set, val_set, test_set

    else:
        if sum(w_tt) == 1: pass
        else: raise NameError("sum of w must be 1")
        
        len_df = len(df.index)
        len_train = round(len_df*w_tt[0])
        len_test = len_df - len_train 

        train_set = df.iloc[0:len_train-1]
        test_set = df.iloc[len_train-1:len_train+len_test-1]

        return train_set, test_set