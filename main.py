from functions import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2015-01-01')

print(downloadReturns(start, end, tickers[0:3]))