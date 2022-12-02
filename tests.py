from data_engineering import *
from algorithm import *
from denoising_and_detoning import *
from portfolio_and_backtesting import *
import nasdaqdatalink

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2020-10-15')

df = getData(start, end, tickers, downloaded = True, format = "returns")
df = df.dropna(axis = 0, how = "any")

df = df.loc[:,"AAPL"]

#alpha, beta, p_alpha, p_beta, r2 = reg_analysis(df)
#print(alpha)
#print(beta)

sr = sortino_ratio(df, 252)
print(sr)

