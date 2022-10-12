import pandas as pd
import numpy
import os
from matplotlib import pyplot as plt
import bs4 as bs
import requests
import yfinance as yf
from datetime import datetime as dt
import pandas_datareader as pdr

# Handling data

def getTickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    tickers = [s.replace('\n', '') for s in tickers]
    return tickers

def downloadReturns(startDate, endDate, tickers):
    prices = pdr.get_data_yahoo(tickers, startDate, endDate)["Close"]
    for stock in prices.columns:
        prices[stock] = prices[stock].pct_change()
    return prices

def getData(startDate, endDate, tickers, downloaded = True):
    path = os.getcwd() + "/data/returns_data.csv"
    if downloaded == True:
        if os.path.exists(path) == True:
            return pd.read_csv(path)    
        else:
            print("File doesnt exist. Please download first.")
    else:
        if os.path.exists(path) == False:
            returns = downloadReturns(startDate, endDate, tickers)
            returns.to_csv(path)
        else:
            print("File already exists. Please remove before downloading a new one.")
            return None


# Denoising

