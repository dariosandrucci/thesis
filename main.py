from data_and_denoising import *
from clustering import *
from portfolio_construction import *

tickers = getTickers()
end = dt.now()
start = dt.fromisoformat('2012-10-12')

data = getData(start, end, tickers, downloaded = True)
corr = getCorrMatrix(data)

#method = [1,2]
#for m in method:
#    eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr, method = m)
#    plotEvalDiff(eVal0, eVal1, method = m)

eVal0, eVec0, eVal1, eVec1, corr1, var0 = denoiseMatrix(corr, method = 1)
print("The data was denoised successfully!")

corr2 = detoneMatrix(corr1, eVal1, eVec1, mc = 1)
print("The data was detoned successfully!")
corr2 = pd.DataFrame(corr2)

#corr3, clstrs, silh_coef_optimal = clusterKMeansBase(corr2)
#print("The data was clustered successfully!")

#corr3 = pd.DataFrame(corr3)

nco = optPort_nco(corr2, maxNumClusters = 3)

print(nco)