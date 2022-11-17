import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.optimize import minimize
import warnings

#warning surpressor
def fxn():
    warnings.warn("Runtime", RuntimeWarning)

# Denoising
def mpPDF(var,q,pts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2 
        eVal=np.linspace(eMin,eMax,pts) 
        pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5 
        pdf=pd.Series(pdf,index=eVal)
        return pdf

def getPCA(matrix):
    eVal,eVec=np.linalg.eigh(matrix) 
    indices=eVal.argsort()[::-1] 
    eVal,eVec=eVal[indices],eVec[:,indices] 
    eVal=np.diagflat(eVal)
    return eVal,eVec

def cov2corr(cov):
    std=np.sqrt(np.diag(cov)) 
    corr=cov/np.outer(std,std) 
    corr[corr<-1],corr[corr>1]=-1,1  
    return corr

def corr2cov(corr):
    std = np.random.uniform(.05,.2,corr.shape[0])
    cov = corr*np.outer(std, std)
    return cov

def fitKDE(obs,bWidth=.25,kernel='gaussian',x=None):
    if len(obs.shape)==1:
        obs=obs.reshape(-1,1) 
    kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs) 
    if x is None:
        x=np.unique(obs).reshape(-1,1)
    if len(x.shape)==1:
        x=x.reshape(-1,1) 
    logProb=kde.score_samples(x) 
    pdf=pd.Series(np.exp(logProb),index=x.flatten())
    return pdf

def errPDFs(var,eVal,q,bWidth,pts=1000):
    var = var[0]
    pdf0=mpPDF(var,q,pts) 
    pdf1=fitKDE(eVal,bWidth,x=pdf0.index.values) 
    sse=np.sum((pdf1-pdf0)**2)
    return sse

def findMaxEval(eVal,q,bWidth):
    out=minimize(lambda *x: errPDFs(*x),.5,args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
    if out['success']:
        var=out['x'][0] 
    else:
        var=1 
    eMax=var*(1+(1./q)**.5)**2
    return eMax,var

def denoisedCorr(eVal,eVec,nFacts): 
    eVal_=np.diag(eVal).copy() 
    eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
    eVal_=np.diag(eVal_)
    corr1=np.dot(eVec,eVal_).dot(eVec.T) 
    corr1=cov2corr(corr1)
    return corr1

def denoisedCorr2(eVal,eVec,nFacts,alpha=0):
    eValL,eVecL=eVal[:nFacts,:nFacts],eVec[:,:nFacts] 
    eValR,eVecR=eVal[nFacts:,nFacts:],eVec[:,nFacts:] 
    corr0=np.dot(eVecL,eValL).dot(eVecL.T) 
    corr1=np.dot(eVecR,eValR).dot(eVecR.T) 
    corr2=corr0+alpha*corr1+(1-alpha)*np.diag(np.diag(corr1)) 
    return corr2

def plotEvalDiff(eVal0, eVal1, method = 1):
    if method == 1:
        m = "Constant Residual"
    else:
        m = "Target Shrinkage"
    ax = plt.figure().add_subplot(111)
    plt.plot(np.diagonal(eVal0),label = 'Original eigen-function')
    plt.plot(np.diagonal(eVal1),label = f'Denoised eigen-function ({m})',linestyle = '--')
    ax.legend()
    ax.set_yscale('log')
    ax.set_xlabel('Eigenvalue number')
    ax.set_ylabel('Eigenvalue (log-scale)')
    plt.show()

def denoiseMatrix(matrix, method = 1, alpha = 0):
    T = float(matrix.shape[0])
    N = matrix.shape[1]
    q = T/N
    eVal0, eVec0 = getPCA(matrix)
    eMax0, var0 = findMaxEval(np.diag(eVal0), q, bWidth=.01)
    nFacts0 = eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
    if method == 1:
        corr1 = denoisedCorr(eVal0, eVec0, nFacts0)
    else:
        corr1 = denoisedCorr2(eVal0, eVec0, nFacts0, alpha)
    eVal1, eVec1 = getPCA(corr1)
    return eVal0, eVec0, eVal1, eVec1, corr1, var0

#detoning

def detoneMatrix(corr, eVal, eVec, mc = 1):
    eVal_ = eVal[:mc, :mc]
    eVec_ = eVec[:, :mc]
    corr_ = np.dot(eVec_, eVal_).dot(eVec_.T)
    corr = corr - corr_
    corr = cov2corr(corr)
    return corr